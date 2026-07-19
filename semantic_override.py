"""
semantic_override.py
======================
Mục đích:
    Module "Semantic Override" — sửa lỗi ngữ nghĩa của YOLO-World bằng cách
    kết hợp với GIT (mô hình captioning end-to-end, không bị giới hạn bởi
    danh sách class cố định nên tránh được vấn đề "class competition" đã
    phát hiện qua debug_raw_predict()).

Nguyên lý (Ensemble 2 mô hình chuyên biệt):
    - YOLO-World: mạnh về ĐỊNH VỊ (bounding box chính xác cho từng bộ phận:
      horns, nose, ear, eye...), nhưng YẾU về phân loại loài khi nhiều loài
      có embedding text gần giống nhau (deer/antelope/llama/cow cạnh tranh
      điểm số cho cùng 1 vùng ảnh).
    - GIT: mạnh về NGỮ NGHĨA TOÀN CỤC (sinh đúng "deer" nhờ không bị ép so
      khớp với 1 danh sách cố định), nhưng KHÔNG có bounding box.
    -> Kết hợp: lấy tên loài từ GIT, lấy box từ YOLO-World (hợp nhất các box
       bộ phận thành 1 box đại diện cho toàn bộ con vật).

An toàn (fail-safe):
    Nếu không trích được tên loài hợp lệ từ GIT (không khớp vocab đã train),
    hoặc không có box bộ phận nào để hợp nhất -> trả về detections GỐC,
    không override gì cả. Module này KHÔNG BAO GIỜ được phép làm crash
    pipeline chính hoặc làm mất object hợp lệ đã detect được.
"""

import re
from typing import List, Optional, Tuple

import torch
from PIL import Image

from sgg_lite import DetectedObject, _normalize_text, BACKGROUND_CATEGORIES

# ============================================================
# CONFIG
# ============================================================
GIT_MODEL_NAME = "microsoft/git-base-coco"
GIT_MAX_LENGTH = 50

# Các category "bộ phận cơ thể" -- dùng để (1) xác định box nào cần hợp nhất
# thành "whole-object box", (2) LOẠI KHỎI danh sách ứng viên khi trích tên
# loài từ caption GIT (vì "horns"/"nose" không phải tên loài, dù có trong vocab).
BODY_PART_CATEGORIES = {
    "horns", "horn", "ears", "ear", "eyes", "eye", "nose", "snout", "muzzle",
    "mouth", "head", "leg", "legs", "front leg", "back legs", "tail", "body",
    "belly", "face", "wing", "wings", "paw", "paws", "hoof", "hooves",
    "fur", "tusks", "beak", "feather", "feathers",
}

# Từ nối/stopword phổ biến trong caption tiếng Anh -- loại khỏi ứng viên
# ngay cả khi (hiếm khi) trùng với 1 entry nào đó trong vocab.
COMMON_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "and", "is", "are",
    "standing", "sitting", "running", "walking", "lying", "looking",
    "through", "near", "next", "to", "background", "photo", "picture",
}


def _tokenize(caption: str) -> List[str]:
    """Tách caption thành list từ đơn, đã chuẩn hóa (lowercase, bỏ dấu câu)."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", " ", caption)
    return [w for w in caption.split() if w]


def extract_candidate_species(caption: str, object_to_idx: dict) -> Optional[str]:
    """
    Quét caption GIT, tìm từ/cụm từ đầu tiên khớp với vocab đã train
    (object_to_idx) VÀ không thuộc nhóm bộ phận cơ thể / stopword.

    Ưu tiên cụm 2 từ trước (vd "polar bear") rồi mới đến từ đơn, tại mỗi
    vị trí quét từ trái sang phải -- vì câu GIT thường đặt chủ thể chính
    ở đầu câu (vd "whitetail deer buck running..." -> "deer" xuất hiện sớm).

    Trả về None nếu không tìm được ứng viên hợp lệ nào (fail-safe).
    """
    words = _tokenize(caption)

    for i in range(len(words)):
        # Thử cụm 2 từ trước (đặc thù hơn, vd "front leg" -- dù ít khi là species)
        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i+1]}"
            if (bigram in object_to_idx
                    and bigram not in BODY_PART_CATEGORIES
                    and bigram not in BACKGROUND_CATEGORIES):
                return bigram

        word = words[i]
        if (word in object_to_idx
                and word not in BODY_PART_CATEGORIES
                and word not in BACKGROUND_CATEGORIES
                and word not in COMMON_STOPWORDS):
            return word

    return None

def extract_candidate_context(
    caption: str,
    object_to_idx: dict,
    exclude: set,
    max_candidates: int = 3,
) -> List[str]:
    """
    Mở rộng của extract_candidate_species(): thay vì chỉ lấy 1 từ species
    đầu tiên, quét TOÀN BỘ caption và lấy tối đa max_candidates từ/cụm từ
    khớp vocab -- bao gồm cả từ ngữ cảnh (background) như "zoo", "snow",
    không chỉ tên loài.

    LƯU Ý: các từ ngữ cảnh này sẽ được thêm vào object_names dưới dạng NODE
    CÔ LẬP (không có bounding box, không tham gia triple nào) -- R-GCN vẫn
    encode được bình thường nhờ cơ chế self-loop đã có sẵn.
    """
    words = _tokenize(caption)
    candidates = []
    seen = set(exclude)

    for i in range(len(words)):
        if len(candidates) >= max_candidates:
            break

        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i+1]}"
            if (bigram in object_to_idx and bigram not in seen
                    and bigram not in COMMON_STOPWORDS):
                candidates.append(bigram)
                seen.add(bigram)
                continue

        word = words[i]
        if (word in object_to_idx and word not in seen
                and word not in COMMON_STOPWORDS):
            candidates.append(word)
            seen.add(word)

    return candidates

def _union_box(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Hợp nhất nhiều box thành 1 box bao trọn tất cả (min x1,y1 - max x2,y2)."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


class SemanticOverrideEngine:
    """
    Load GIT model 1 lần (giống pattern InferenceService load các model khác),
    expose 2 hàm chính: get_caption() và apply_override().
    """

    def __init__(self, device: torch.device, model_name: str = GIT_MODEL_NAME):
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.device = device
        print(f"[SemanticOverride] Đang load GIT ({model_name}) ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        print("[SemanticOverride] GIT sẵn sàng.")

    @torch.no_grad()
    def get_caption(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=GIT_MAX_LENGTH)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def apply_override(
            self,
            detections: List[DetectedObject],
            object_to_idx: dict,
            image: Optional[Image.Image] = None,
            git_caption: Optional[str] = None,
    ) -> Tuple[List[DetectedObject], List[Tuple[str, str, str]], dict]:
        """
        Xử lý THỐNG NHẤT 3 tình huống lỗi của YOLO-World (đã quan sát thực tế qua
        2 pha test khác nhau):

            A. Chỉ detect được bộ phận cơ thể, KHÔNG có whole-object nào
               (vd deer.jpg: chỉ có "horns")
               -> Hợp nhất box các bộ phận thành 1 "whole box", gán species từ GIT.

            B. Detect được whole-object nhưng SAI loài (class competition khiến
               1 loài gần giống "thắng" nhầm, vd lion.jpg -> "bear")
               -> Giữ NGUYÊN bounding box (vẫn đúng vị trí con vật), chỉ THAY
               NHÃN (relabel) từ nhãn sai sang species đúng theo GIT.

            C. Detect đúng ngay từ đầu (whole-object label == species GIT đoán)
               -> Không cần override, giữ nguyên hoàn toàn.

        Nguyên tắc chung: GIT luôn là "trọng tài" quyết định species cuối cùng
        (vì đã xác nhận qua debug_raw_predict + test GIT độc lập: GIT ít bị
        class-competition hơn hẳn YOLO-World). YOLO-World chỉ đóng góp
        BOUNDING BOX -- dù box đó gắn với nhãn đúng hay sai ban đầu.
        """
        debug_info = {
            "git_caption": git_caption,
            "override_applied": False,
            "override_species": None,
            "override_type": None,  # "part_union" | "relabel" | None -- để debug dễ hơn
        }

        if git_caption is None:
            if image is None:
                return detections, [], debug_info
            git_caption = self.get_caption(image)
            debug_info["git_caption"] = git_caption

        species = extract_candidate_species(git_caption, object_to_idx)
        if species is None:
            return detections, [], debug_info

        species_label = _normalize_text(species)

        part_detections = [d for d in detections if d.label in BODY_PART_CATEGORIES]
        whole_candidates = [
            d for d in detections
            if d.label not in BODY_PART_CATEGORIES and d.label not in BACKGROUND_CATEGORIES
        ]

        # ---- Tình huống C: đã đúng sẵn, không cần làm gì ----
        if any(d.label == species_label for d in whole_candidates):
            return detections, [], debug_info

        # ---- Tình huống B: có whole-object nhưng sai loài -> RELABEL ----
        if whole_candidates:
            best_whole = max(whole_candidates, key=lambda d: d.score)  # ưu tiên box có score cao nhất
            synthesized = DetectedObject(label=species_label, score=best_whole.score, box=best_whole.box)

            # Loại bỏ MỌI whole_candidates cũ (nhãn sai), giữ lại bộ phận + background
            kept = [d for d in detections if d.label in BODY_PART_CATEGORIES or d.label in BACKGROUND_CATEGORIES]
            final_detections = kept + [synthesized]

            forced_triples = [(d.label, "part of", species_label) for d in part_detections]

            debug_info["override_applied"] = True
            debug_info["override_species"] = species
            debug_info["override_type"] = "relabel"
            return final_detections, forced_triples, debug_info

        # ---- Tình huống A: chỉ có bộ phận, không có whole-object nào ----
        if part_detections:
            part_boxes = [d.box for d in part_detections]
            whole_box = _union_box(part_boxes)
            synthesized_score = sum(d.score for d in part_detections) / len(part_detections)
            synthesized = DetectedObject(label=species_label, score=synthesized_score, box=whole_box)

            kept = [d for d in detections if d.label in BODY_PART_CATEGORIES or d.label in BACKGROUND_CATEGORIES]
            final_detections = kept + [synthesized]

            forced_triples = [(d.label, "part of", species_label) for d in part_detections]

            debug_info["override_applied"] = True
            debug_info["override_species"] = species
            debug_info["override_type"] = "part_union"
            return final_detections, forced_triples, debug_info

        # ---- Không có gì để override (không bộ phận, không whole-object nào khác) ----
        return detections, [], debug_info