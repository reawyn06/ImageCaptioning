"""
build_scene_graphs.py
======================
Mục đích:
    Xây dựng scene graph (object-relation-object triples) cho từng ảnh COCO,
    dựa trên dữ liệu Visual Genome (objects.json + relationships.json),
    chỉ xử lý các ảnh đã có mapping hợp lệ trong vg_to_coco_mapping.json
    (Phương án A: ~51,208 ảnh, train 49,038 / val 2,170).

Input:
    - datasets/visual_genome/objects.json
    - datasets/visual_genome/relationships.json
    - datasets/visual_genome/vg_to_coco_mapping.json   (coco_id -> vg_image_id)

Output:
    - features/semantic/train2017/{coco_id}.json
    - features/semantic/val2017/{coco_id}.json
    Mỗi file chứa:
        {
            "coco_id": int,
            "vg_image_id": int,
            "objects": ["man", "bike", "road", ...],          # node list, đã lọc nhiễu, đã khử trùng
            "triples": [["man", "riding", "bike"], ...]        # edge list (subject, predicate, object)
        }

Lưu ý quan trọng:
    - Mỗi ảnh ra 1 file riêng (giống cách lưu visual features) theo yêu cầu.
    - Có bước lọc nhiễu cơ bản (xem hàm is_valid_object_name / is_valid_predicate).
    - KHÔNG giới hạn (cap) số lượng object/triples mỗi ảnh — số lượng giữ biến đổi,
      việc padding/masking sẽ xử lý ở bước dataloader của model (không xử lý ở đây).
"""

import json
import os
import re
import time
from collections import defaultdict

# ============================================================
# CONFIG — chỉnh đường dẫn nếu cấu trúc thư mục của bạn khác
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"

VG_DIR = os.path.join(PROJECT_ROOT, "datasets", "visual_genome")
OBJECTS_PATH = os.path.join(VG_DIR, "objects.json")
RELATIONSHIPS_PATH = os.path.join(VG_DIR, "relationships.json")
MAPPING_PATH = os.path.join(VG_DIR, "vg_to_coco_mapping.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "features", "semantic")
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train2017")
OUTPUT_VAL_DIR = os.path.join(OUTPUT_DIR, "val2017")

# Cần biết coco_id nào thuộc train2017 / val2017 để lưu đúng thư mục.
# Dùng chính file annotation COCO đã tải (instances) để lấy danh sách id mỗi split.
COCO_ANN_DIR = os.path.join(PROJECT_ROOT, "datasets", "coco", "annotations")
COCO_TRAIN_ANN = os.path.join(COCO_ANN_DIR, "instances_train2017.json")
COCO_VAL_ANN = os.path.join(COCO_ANN_DIR, "instances_val2017.json")


# ============================================================
# BƯỚC 1 — Các hàm lọc nhiễu cơ bản
# ============================================================
# Danh sách tên object/predicate mơ hồ, không mang thông tin ngữ nghĩa hữu ích.
# Đây là danh sách tối giản, tập trung vào các trường hợp gây nhiễu rõ ràng nhất
# trong Visual Genome (đã được nhiều paper scene-graph khác cũng loại bỏ tương tự).
VAGUE_OBJECT_NAMES = {
    "thing", "things", "stuff", "object", "objects", "item", "items",
    "part", "parts", "area", "section", "piece",
}

# Predicate quá chung / không mang ý nghĩa quan hệ rõ ràng để dùng cho GCN
# (vẫn giữ lại các quan hệ không gian/hành động phổ biến như "on", "riding", "holding", ...)
VAGUE_PREDICATES = {
    "", "and", "of", "with",  # rác, không phải quan hệ thật
}


def normalize_text(s: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ khoảng trắng dư, bỏ ký tự đặc biệt thừa."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def is_valid_object_name(name: str) -> bool:
    """Trả về False nếu tên object là nhiễu (rỗng, quá mơ hồ, quá dài bất thường)."""
    name = normalize_text(name)
    if not name:
        return False
    if name in VAGUE_OBJECT_NAMES:
        return False
    # Tên object bất thường dài (thường là lỗi annotation, vd cả câu mô tả)
    if len(name.split()) > 4:
        return False
    return True


def is_valid_predicate(predicate: str) -> bool:
    """Trả về False nếu predicate là nhiễu (rỗng hoặc quá chung)."""
    predicate = normalize_text(predicate)
    if not predicate:
        return False
    if predicate in VAGUE_PREDICATES:
        return False
    return True


# ============================================================
# BƯỚC 2 — Load mapping COCO <-> VG và xác định train/val split
# ============================================================
def load_mapping():
    print("Đang đọc vg_to_coco_mapping.json ...")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # mapping: { "coco_id_str": vg_image_id, ... }  (key có thể là string do JSON)
    mapping = {int(k): int(v) for k, v in mapping.items()}
    print(f"  -> {len(mapping)} ảnh COCO có scene graph khớp.")
    return mapping


def load_coco_split_ids():
    """Đọc instances_train2017.json / instances_val2017.json để biết coco_id nào thuộc split nào."""
    print("Đang đọc COCO annotations để xác định train/val split ...")

    with open(COCO_TRAIN_ANN, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_ids = {img["id"] for img in train_data["images"]}

    with open(COCO_VAL_ANN, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_ids = {img["id"] for img in val_data["images"]}

    print(f"  -> train2017: {len(train_ids)} ảnh | val2017: {len(val_ids)} ảnh")
    return train_ids, val_ids


# ============================================================
# BƯỚC 3 — Load objects.json và relationships.json, index theo vg_image_id
# ============================================================
def load_objects_indexed():
    """
    Trả về dict: vg_image_id -> list tên object đã lọc nhiễu (đã khử trùng).
    Cũng trả về dict: (vg_image_id, object_id) -> tên object đã chuẩn hóa,
    dùng để tra cứu nhanh khi build triples từ relationships.json.
    """
    print("Đang đọc objects.json (có thể mất một lúc, file khá lớn) ...")
    t0 = time.time()
    with open(OBJECTS_PATH, "r", encoding="utf-8") as f:
        objects_data = json.load(f)
    print(f"  -> Đọc xong trong {time.time() - t0:.1f}s, {len(objects_data)} ảnh.")

    image_to_object_names = {}      # vg_image_id -> set tên object hợp lệ
    objectid_to_name = {}           # (vg_image_id, object_id) -> tên object chuẩn hóa (kể cả khi nhiễu, để lookup)

    for entry in objects_data:
        vg_id = entry["image_id"]
        valid_names = set()
        for obj in entry.get("objects", []):
            object_id = obj["object_id"]
            # 1 object có thể có nhiều "names" (đồng nghĩa) -> lấy tên đầu tiên làm đại diện
            raw_name = obj["names"][0] if obj.get("names") else ""
            norm_name = normalize_text(raw_name)
            objectid_to_name[(vg_id, object_id)] = norm_name

            if is_valid_object_name(norm_name):
                valid_names.add(norm_name)

        if valid_names:
            image_to_object_names[vg_id] = valid_names

    return image_to_object_names, objectid_to_name


def load_relationships_indexed(objectid_to_name):
    """
    Trả về dict: vg_image_id -> list triples [(subject_name, predicate, object_name), ...]
    đã lọc nhiễu (cả object name và predicate).
    """
    print("Đang đọc relationships.json ...")
    t0 = time.time()
    with open(RELATIONSHIPS_PATH, "r", encoding="utf-8") as f:
        rel_data = json.load(f)
    print(f"  -> Đọc xong trong {time.time() - t0:.1f}s, {len(rel_data)} ảnh.")

    image_to_triples = defaultdict(list)

    for entry in rel_data:
        vg_id = entry["image_id"]
        for rel in entry.get("relationships", []):
            predicate = normalize_text(rel.get("predicate", ""))
            if not is_valid_predicate(predicate):
                continue

            subj = rel.get("subject", {})
            obj = rel.get("object", {})

            # Ưu tiên lấy tên trực tiếp từ relationship (đã có sẵn "name"),
            # nếu thiếu thì fallback tra cứu qua objectid_to_name đã build ở bước objects.json.
            subj_name = normalize_text(subj.get("name", "")) or \
                objectid_to_name.get((vg_id, subj.get("object_id")), "")
            obj_name = normalize_text(obj.get("name", "")) or \
                objectid_to_name.get((vg_id, obj.get("object_id")), "")

            if not is_valid_object_name(subj_name) or not is_valid_object_name(obj_name):
                continue
            if subj_name == obj_name:
                # Quan hệ tự thân (subject == object) thường là lỗi annotation, bỏ qua
                continue

            image_to_triples[vg_id].append([subj_name, predicate, obj_name])

    return image_to_triples


# ============================================================
# BƯỚC 4 — Ghép tất cả lại, lưu mỗi ảnh 1 file theo đúng split
# ============================================================
def build_and_save():
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)

    mapping = load_mapping()
    train_ids, val_ids = load_coco_split_ids()

    image_to_object_names, objectid_to_name = load_objects_indexed()
    image_to_triples = load_relationships_indexed(objectid_to_name)

    print("Đang ghép scene graph và lưu file cho từng ảnh COCO ...")

    n_saved_train = 0
    n_saved_val = 0
    n_skipped_no_data = 0
    n_skipped_no_split = 0

    for coco_id, vg_id in mapping.items():
        object_names = image_to_object_names.get(vg_id, set())
        triples = image_to_triples.get(vg_id, [])

        # Đảm bảo node list bao gồm cả các object xuất hiện trong triples
        # (trường hợp object đó bị lọc khỏi objects.json nhưng vẫn hợp lệ trong relationship)
        for s, p, o in triples:
            object_names.add(s)
            object_names.add(o)

        if not object_names and not triples:
            n_skipped_no_data += 1
            continue

        record = {
            "coco_id": coco_id,
            "vg_image_id": vg_id,
            "objects": sorted(object_names),
            "triples": triples,
        }

        if coco_id in train_ids:
            out_path = os.path.join(OUTPUT_TRAIN_DIR, f"{coco_id}.json")
            n_saved_train += 1
        elif coco_id in val_ids:
            out_path = os.path.join(OUTPUT_VAL_DIR, f"{coco_id}.json")
            n_saved_val += 1
        else:
            # coco_id có trong mapping nhưng không khớp với COCO 2017 train/val đã tải
            n_skipped_no_split += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("HOÀN TẤT XÂY SCENE GRAPH")
    print("=" * 60)
    print(f"Đã lưu train2017: {n_saved_train} file")
    print(f"Đã lưu val2017:   {n_saved_val} file")
    print(f"Bỏ qua (không có object/triple nào sau lọc nhiễu): {n_skipped_no_data}")
    print(f"Bỏ qua (coco_id không khớp split train/val đã tải): {n_skipped_no_split}")
    print(f"Tổng cộng đã xử lý: {n_saved_train + n_saved_val} / {len(mapping)} ảnh trong mapping")


if __name__ == "__main__":
    build_and_save()