"""
build_yolo_vocab.py
======================
Mục đích:
    Xây whitelist vocab cho YOLO-World, dựa trên tần suất object thực tế
    xuất hiện trong scene graph đã clean (features/semantic/train2017/*.json).

Logic:
    1. Quét toàn bộ file JSON, đếm tần suất mỗi object category.
    2. Lấy top-K category phổ biến nhất (đảm bảo tốc độ + độ chính xác YOLO-World).
    3. Bổ sung thêm MANUAL_INCLUDE_LIST — các category quan trọng nhưng hiếm gặp
       trong COCO (ví dụ động vật hoang dã: deer, goat...) mà nếu chỉ lọc theo
       tần suất thuần túy sẽ bị cắt bỏ oan, dẫn đến detector không nhận diện được
       (gây ra lỗi gán nhầm category như "deer" -> "cow"/"sheep" ban đầu).
    4. Xuất ra:
        - yolo_world_vocab.txt              : whitelist cuối cùng (top-K + manual include)
        - yolo_world_vocab_with_counts.json : whitelist cuối cùng kèm tần suất
        - yolo_world_vocab_FULL.json        : TOÀN BỘ category (không cắt), để rà soát long-tail sau này

Output dùng cho:
    Bước tích hợp YOLO-World (yolov8s-worldv2) trong sgg_lite.py — file .txt sẽ
    được load làm text prompt / custom class cho detector.
"""

import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CẤU HÌNH
# ============================================================
SCENE_GRAPH_DIR = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning\features\semantic\train2017"
OUTPUT_DIR = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning\features"
TOP_K = 3500

# Category quá mơ hồ, không mang giá trị phát hiện thực sự -> loại thủ công
MANUAL_EXCLUDE = {"thing", "item", "stuff", "object"}

# ============================================================
# MANUAL INCLUDE LIST
# ============================================================
# Các category quan trọng về mặt domain (thường là động vật hoang dã / vật thể
# đặc thù) nhưng do COCO vốn thiên về ảnh đô thị/sinh hoạt nên tần suất xuất
# hiện trong scene graph rất thấp -> dễ bị cắt oan nếu chỉ lọc theo top-K
# tần suất thuần túy. Những category này được xác nhận CÓ tồn tại trong vocab
# (đã pass qua is_valid_object_name + is_valid_entry ở bước build/clean trước
# đó), chỉ là tần suất thấp -> ép giữ lại bất kể hạng.
#
# Danh sách này nên được cập nhật dần khi phát hiện thêm case tương tự "deer"
# (ví dụ qua việc rà soát yolo_world_vocab_FULL.json ở các lần sau).
MANUAL_INCLUDE_LIST = {
    "deer", "goat", "sheep", "ox", "wolf", "fox", "rabbit", "hare",
    "raccoon", "squirrel", "hedgehog", "llama", "alpaca", "camel",
    "kangaroo", "koala", "panda", "moose", "antelope", "buffalo",
    "donkey", "mule", "boar", "hedgehog",
}


def load_object_labels_from_scene_graph(json_path):
    """
    Đọc 1 file scene graph JSON theo đúng schema thực tế:
    {"coco_id": ..., "vg_image_id": ..., "objects": [...], "triples": [...]}

    Chỉ lấy từ "objects" — mỗi object trong list này chỉ xuất hiện 1 lần/ảnh
    dù thực tế ảnh có bao nhiêu instance, nên tần suất đếm được phản ánh đúng
    "số ảnh có chứa category X".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    return [obj.strip().lower() for obj in objects if obj and obj.strip()]


def build_vocab_whitelist(scene_graph_dir, top_k, manual_exclude):
    counter = Counter()
    json_files = list(Path(scene_graph_dir).glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .json nào trong {scene_graph_dir}."
        )

    print(f"Tổng số file scene graph cần quét: {len(json_files)}")

    corrupted_files = []
    for json_path in tqdm(json_files, desc="Đang đếm tần suất object"):
        try:
            labels = load_object_labels_from_scene_graph(json_path)
            counter.update(set(labels))
        except (json.JSONDecodeError, KeyError) as e:
            corrupted_files.append((json_path.name, str(e)))
            continue

    if corrupted_files:
        print(f"\n[Cảnh báo] {len(corrupted_files)} file lỗi bị bỏ qua:")
        for name, err in corrupted_files[:5]:
            print(f"  - {name}: {err}")

    for excluded in manual_exclude:
        counter.pop(excluded, None)

    most_common = counter.most_common(top_k)
    return most_common, counter


def merge_with_manual_include(most_common, full_counter, manual_include_list):
    """
    Gộp top-K theo tần suất với danh sách manual include.
    - Nếu category trong manual_include_list ĐÃ có trong top-K -> không thêm trùng.
    - Nếu category trong manual_include_list TỒN TẠI trong full_counter nhưng
      bị cắt khỏi top-K -> thêm vào cuối danh sách, kèm tần suất thật.
    - Nếu category trong manual_include_list KHÔNG tồn tại trong full_counter
      (nghĩa là không hề xuất hiện trong scene graph nào) -> bỏ qua, in cảnh báo
      (để tránh add nhầm category không thực sự có trong vocab dữ liệu).
    """
    existing_labels = {label for label, _ in most_common}
    added = []
    not_found = []

    for label in sorted(manual_include_list):
        if label in existing_labels:
            continue  # đã có sẵn trong top-K, không cần thêm
        if label in full_counter:
            added.append((label, full_counter[label]))
        else:
            not_found.append(label)

    final_list = most_common + added
    return final_list, added, not_found


def export_full_list(full_counter, output_dir):
    """Xuất TOÀN BỘ category (không cắt) để rà soát long-tail ở các lần sau."""
    full_json_path = Path(output_dir) / "yolo_world_vocab_FULL.json"
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"label": label, "count": count} for label, count in full_counter.most_common()],
            f, ensure_ascii=False, indent=2
        )
    print(f"  - {full_json_path} (toàn bộ {len(full_counter)} category, để rà soát long-tail)")


def export_whitelist(final_list, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    txt_path = Path(output_dir) / "yolo_world_vocab.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for label, _ in final_list:
            f.write(label + "\n")

    json_path = Path(output_dir) / "yolo_world_vocab_with_counts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"label": label, "count": count} for label, count in final_list],
            f, ensure_ascii=False, indent=2
        )

    print(f"\nĐã xuất whitelist cuối cùng ({len(final_list)} category):")
    print(f"  - {txt_path}")
    print(f"  - {json_path}")


def check_specific_labels(counter, labels_to_check):
    """Kiểm tra nhanh tần suất + hạng của các category cụ thể trong toàn bộ vocab."""
    sorted_items = counter.most_common()
    rank_lookup = {label: rank for rank, (label, _) in enumerate(sorted_items, start=1)}

    print("\n--- Kiểm tra category cụ thể ---")
    for label in labels_to_check:
        if label in counter:
            print(f"  '{label}': tần suất={counter[label]}, hạng={rank_lookup[label]}")
        else:
            print(f"  '{label}': KHÔNG xuất hiện trong bất kỳ scene graph nào")


if __name__ == "__main__":
    most_common, full_counter = build_vocab_whitelist(
        SCENE_GRAPH_DIR, TOP_K, MANUAL_EXCLUDE
    )

    print(f"\nTổng số category duy nhất tìm thấy: {len(full_counter)}")
    print(f"Top {TOP_K} category phổ biến nhất theo tần suất thuần túy.\n")

    print("--- Top 20 category phổ biến nhất ---")
    for label, count in most_common[:20]:
        print(f"  {label:<20} {count}")

    print("\n--- 20 category cuối trong top-K (ngưỡng cắt) ---")
    for label, count in most_common[-20:]:
        print(f"  {label:<20} {count}")

    # Kiểm tra thử các category "khó" trước khi merge — để so sánh trước/sau
    check_specific_labels(full_counter, ["deer", "giraffe", "elephant", "zebra", "goat"])

    # ---------- BƯỚC MERGE MANUAL INCLUDE ----------
    final_list, added, not_found = merge_with_manual_include(
        most_common, full_counter, MANUAL_INCLUDE_LIST
    )

    print("\n--- Manual Include: category được thêm bổ sung (bị cắt khỏi top-K) ---")
    if added:
        for label, count in sorted(added, key=lambda x: -x[1]):
            print(f"  + {label:<20} (tần suất={count})")
    else:
        print("  (Không có category nào cần thêm — tất cả đã nằm trong top-K)")

    if not_found:
        print("\n--- Cảnh báo: category trong MANUAL_INCLUDE_LIST không tồn tại trong dữ liệu ---")
        for label in not_found:
            print(f"  ! {label} — không xuất hiện trong bất kỳ scene graph nào, không thể thêm")

    print(f"\nTổng số category cuối cùng sau merge: {len(final_list)} "
          f"({len(most_common)} từ top-K + {len(added)} từ manual include)")

    # ---------- XUẤT FILE ----------
    export_full_list(full_counter, OUTPUT_DIR)
    export_whitelist(final_list, OUTPUT_DIR)