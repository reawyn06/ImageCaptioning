"""
sgg_lite.py
======================
Mục đích:
    Sinh scene graph (objects + triples) ON-THE-FLY cho 1 ảnh BẤT KỲ (không
    thuộc COCO/Visual Genome), để dùng cho web app demo inference với 3
    strategy cần semantic feature (concat, one_directional, bidirectional).

    KHÔNG đụng đến training pipeline / rgcn_encoder.py gốc -- module này chỉ
    là 1 lớp tiền xử lý MỚI, đứng TRƯỚC RGCNEncoder.forward(), sinh ra đúng
    format (object_names: List[str], triples: List[Tuple[str,str,str]]) mà
    RGCNEncoder đã định nghĩa.

CẬP NHẬT (thay DETR bằng YOLO-World để giải quyết vocabulary mismatch):
    1. Object Detector: KHÔNG còn dùng DETR trong file này. Detector giờ là
       YOLOWorldDetector (định nghĩa ở yolo_world_detector.py), được truyền
       vào build_scene_graph_for_image() qua tham số `detector` -- module
       này chỉ yêu cầu detector có method .detect(image) trả về
       List[DetectedObject] (duck-typing), KHÔNG import cụ thể class nào,
       để tránh circular import với yolo_world_detector.py.
    2. Heuristic suy relationship: nâng cấp thêm bước containment-check
       (part of / inside) TRƯỚC bước suy quan hệ không gian cũ (on/above/
       below/next to) -- giải quyết đúng vấn đề "mắt/mũi bị gán next to
       cánh đồng" thay vì "part of khuôn mặt".
    3. VOCAB-SAFE FILTER (bắt buộc, quan trọng nhất): giữ nguyên không đổi
       -- GloveVocab.object_to_idx/predicate_to_idx là dict lookup KHÔNG có
       fallback, bất kỳ tên lạ nào sẽ làm KeyError, sập cả pipeline.

Lưu ý quan trọng:
    Predicate "part of" (dùng cho containment) PHẢI tồn tại trong
    predicate_to_idx đã train thì mới không bị filter_to_known_vocab() lọc
    mất. Cần verify trước khi tin tưởng tính năng này hoạt động (xem ghi
    chú kèm theo khi triển khai).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PIL import Image


# ============================================================
# CONFIG
# ============================================================
MAX_OBJECTS_PER_IMAGE = 12             # tránh quá nhiều node (ảnh phức tạp) làm chậm/ồn

# Ngưỡng hình học cho heuristic suy relationship KHÔNG GIAN (tỉ lệ theo kích thước ảnh)
NEAR_DISTANCE_RATIO = 0.15   # 2 object được coi là "gần nhau" nếu khoảng cách tâm < 15% đường chéo ảnh

# Ngưỡng hình học cho heuristic CONTAINMENT (part of / inside)
CONTAINMENT_OVERLAP_RATIO = 0.7          # box A coi như "bộ phận của" box B nếu >=70% diện tích A nằm trong B
AREA_RATIO_THRESHOLD_FOR_PART_OF = 0.35  # A phải nhỏ hơn B ít nhất theo tỉ lệ này mới coi A là "bộ phận"

BACKGROUND_CATEGORIES = {
    "field", "pasture", "grass", "sky", "ground", "floor", "wall", "ceiling",
    "background", "road", "street", "sidewalk", "water", "sand", "snow",
    "mountain", "forest", "beach", "cloud",
}

@dataclass
class DetectedObject:
    label: str          # tên object đã chuẩn hóa (chữ thường, vd "person", "bicycle")
    score: float
    box: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) theo pixel ảnh gốc


def _normalize_text(s: str) -> str:
    """Chuẩn hóa giống đúng quy ước normalize_text() trong build_scene_graphs.py,
    để tên object detector sinh ra có khả năng cao nhất KHỚP với vocab đã train
    (vocab được build từ Visual Genome qua đúng hàm chuẩn hóa này)."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# BƯỚC 1 — Object Detector
# ============================================================
# ĐÃ LOẠI BỎ class ObjectDetector (DETR) khỏi file này theo quyết định đã
# chốt: bỏ hẳn DETR, dùng 100% YOLO-World.
#
# Detector giờ được khởi tạo từ yolo_world_detector.YOLOWorldDetector và
# truyền vào build_scene_graph_for_image() ở dưới -- module này KHÔNG cần
# biết cụ thể đó là class nào, chỉ cần nó có method .detect(image) trả về
# List[DetectedObject] (xem YOLOWorldDetector.detect() -- interface khớp
# 100% với những gì code bên dưới kỳ vọng).


# ============================================================
# BƯỚC 2A — Heuristic CONTAINMENT (part of / inside) — MỚI
# ============================================================
def _box_center(box):
    xmin, ymin, xmax, ymax = box
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def _box_area(box):
    xmin, ymin, xmax, ymax = box
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _containment_ratio(box_a, box_b) -> float:
    """Tỉ lệ diện tích box_a nằm trong box_b. Gần 1.0 nghĩa là A nằm gần trọn trong B."""
    area_a = _box_area(box_a)
    if area_a == 0:
        return 0.0
    return _intersection_area(box_a, box_b) / area_a


def _infer_containment(box_a, box_b) -> Optional[str]:
    """
    Kiểm tra quan hệ bao hàm giữa 2 box.
    Trả về:
        "a_part_of_b"  nếu A là bộ phận nhỏ nằm trong B (vd mắt trong mặt)
        "b_part_of_a"  nếu ngược lại
        None           nếu không có quan hệ bao hàm rõ ràng

    LƯU Ý: hàm này KHÔNG cần biết label của A/B -- việc chặn containment với
    background category (field, sky, grass...) đã được xử lý ở CẤP VÒNG LẶP
    trong infer_triples() (kiểm tra label_i/label_j in BACKGROUND_CATEGORIES
    và continue sớm), nên hàm này chỉ thuần túy tính hình học.
    """
    area_a, area_b = _box_area(box_a), _box_area(box_b)
    if area_a == 0 or area_b == 0:
        return None

    ratio_a_in_b = _containment_ratio(box_a, box_b)
    if ratio_a_in_b >= CONTAINMENT_OVERLAP_RATIO and (area_a / area_b) <= AREA_RATIO_THRESHOLD_FOR_PART_OF:
        return "a_part_of_b"

    ratio_b_in_a = _containment_ratio(box_b, box_a)
    if ratio_b_in_a >= CONTAINMENT_OVERLAP_RATIO and (area_b / area_a) <= AREA_RATIO_THRESHOLD_FOR_PART_OF:
        return "b_part_of_a"

    return None


# ============================================================
# BƯỚC 2B — Heuristic suy relationship KHÔNG GIAN (giữ nguyên logic gốc)
# ============================================================
def _infer_predicate(box_a, box_b, image_diagonal: float) -> Optional[str]:
    """
    Suy ra predicate KHÔNG GIAN giữa 2 box A, B (chỉ gọi khi _infer_containment
    đã xác nhận KHÔNG có quan hệ bao hàm cho cặp này). Thứ tự ưu tiên:
        1. "on"        -- đáy của A chạm/gần đỉnh của B
        2. "above"     -- tâm A cao hơn tâm B rõ ràng, có overlap ngang
        3. "below"     -- ngược lại "above"
        4. "next to"   -- 2 box gần nhau nhưng không overlap nhiều
        Trả None nếu 2 object quá xa nhau.
    """
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    if area_a == 0 or area_b == 0:
        return None

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    cx_a, cy_a = _box_center(box_a)
    cx_b, cy_b = _box_center(box_b)
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

    if dist > image_diagonal * 0.6:
        return None

    horizontal_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    has_horizontal_overlap = horizontal_overlap > 0.2 * min(ax2 - ax1, bx2 - bx1)

    tolerance = image_diagonal * 0.08
    bottom_touches_top = has_horizontal_overlap and abs(ay2 - by1) < tolerance and cy_a < cy_b
    if bottom_touches_top:
        return "on"

    if has_horizontal_overlap and abs(cy_a - cy_b) > 0.1 * image_diagonal:
        return "above" if cy_a < cy_b else "below"

    if dist < image_diagonal * NEAR_DISTANCE_RATIO:
        return "next to"

    return None


def infer_triples(detections: List[DetectedObject], image_size: Tuple[int, int]) -> List[Tuple[str, str, str]]:
    width, height = image_size
    image_diagonal = (width ** 2 + height ** 2) ** 0.5

    triples = []
    n = len(detections)
    for i in range(n):
        for j in range(i + 1, n):
            label_i, label_j = detections[i].label, detections[j].label

            # ---- Bước 0: chặn background category khỏi MỌI quan hệ ----
            if label_i in BACKGROUND_CATEGORIES or label_j in BACKGROUND_CATEGORIES:
                continue

            box_i, box_j = detections[i].box, detections[j].box

            # ---- Bước 1: containment check (KHÔNG truyền label) ----
            containment = _infer_containment(box_i, box_j)
            if containment == "a_part_of_b":
                triples.append((label_i, "part of", label_j))
                continue
            elif containment == "b_part_of_a":
                triples.append((label_j, "part of", label_i))
                continue

            # ---- Bước 2: fallback về quan hệ không gian ----
            predicate = _infer_predicate(box_i, box_j, image_diagonal)
            if predicate is not None:
                triples.append((label_i, predicate, label_j))

    return triples

# ============================================================
# BƯỚC 3 — Vocab-safe filter (BẮT BUỘC trước khi đưa vào RGCNEncoder)
# ============================================================
def filter_to_known_vocab(
    object_names: List[str],
    triples: List[Tuple[str, str, str]],
    object_to_idx: dict,
    predicate_to_idx: dict,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Lọc bỏ MỌI object/triple có tên KHÔNG nằm trong vocab đã train.
    LƯU Ý: predicate "part of" (sinh ra từ containment heuristic mới) PHẢI
    tồn tại trong predicate_to_idx thì mới không bị lọc mất ở đây -- nếu
    chưa verify điều này, hãy kiểm tra trước khi tin tưởng containment
    edges thực sự xuất hiện trong scene graph cuối cùng.
    """
    safe_objects = [name for name in object_names if name in object_to_idx]

    safe_objects_set = set(safe_objects)
    safe_triples = []
    for s, p, o in triples:
        if s in safe_objects_set and p in predicate_to_idx and o in safe_objects_set:
            safe_triples.append((s, p, o))

    return safe_objects, safe_triples


# ============================================================
# BƯỚC 4 — Hàm tổng hợp: ảnh -> (object_names, triples) đã vocab-safe
# ============================================================
def build_scene_graph_for_image(
    image: Image.Image,
    detector,
    object_to_idx: dict,
    predicate_to_idx: dict,
    detections_postprocess=None,
    triples_postprocess=None,   # THAM SỐ MỚI
) -> dict:
    """
    ... (docstring giữ nguyên, bổ sung) ...

    triples_postprocess (mới): callable(triples, detections) -> triples, gọi
    NGAY SAU infer_triples() -- dùng để chèn các quan hệ ĐÃ BIẾT CHẮC CHẮN
    (forced_triples từ Semantic Override), ưu tiên hơn/thay thế kết quả suy
    luận hình học thuần túy khi có xung đột.
    """
    detections = detector.detect(image)

    if detections_postprocess is not None:
        detections = detections_postprocess(detections, image)

    raw_object_names = [d.label for d in detections]
    raw_triples = infer_triples(detections, image.size)

    if triples_postprocess is not None:
        raw_triples = triples_postprocess(raw_triples, detections)

    safe_objects, safe_triples = filter_to_known_vocab(
        raw_object_names, raw_triples, object_to_idx, predicate_to_idx
    )

    return {
        "raw_detections": detections,
        "object_names": safe_objects,
        "triples": safe_triples,
        "n_objects_dropped_oov": len(set(raw_object_names) - set(safe_objects)),
    }