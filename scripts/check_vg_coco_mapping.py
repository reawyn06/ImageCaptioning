"""
check_vg_coco_mapping.py
==========================
Kiểm tra file image_data.json của Visual Genome:
- Đọc đúng format không
- Bao nhiêu ảnh VG có coco_id khớp với COCO 2017 (train/val) bạn đã tải
- Đây là bước XÁC NHẬN trước khi xây scene graph, để biết chính xác có
  bao nhiêu ảnh COCO sẽ có đủ cả visual feature + semantic feature
  (chỉ những ảnh có coco_id hợp lệ mới dùng được cho 4 thực nghiệm fusion)

Chạy: python check_vg_coco_mapping.py
"""

import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning\datasets")

VG_DIR = BASE_DIR / "visual_genome"
VG_IMAGE_DATA_PATH = VG_DIR / "image_data.json"

COCO_ANNOTATIONS_DIR = BASE_DIR / "coco" / "annotations"


def load_coco_image_ids(split: str) -> set:
    """Lấy toàn bộ image_id từ COCO annotations (train hoặc val)."""
    ann_path = COCO_ANNOTATIONS_DIR / f"captions_{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {img["id"] for img in data["images"]}


def main():
    print("=" * 60)
    print("KIỂM TRA MAPPING VISUAL GENOME <-> COCO")
    print("=" * 60)

    # 1. Đọc image_data.json
    if not VG_IMAGE_DATA_PATH.exists():
        print(f"❌ Không tìm thấy file: {VG_IMAGE_DATA_PATH}")
        print("   Hãy chắc chắn đã giải nén image_data.json vào đúng thư mục.")
        return

    with open(VG_IMAGE_DATA_PATH, "r", encoding="utf-8") as f:
        vg_image_data = json.load(f)

    print(f"\n✅ Đọc thành công image_data.json: {len(vg_image_data)} entries")

    # Kiểm tra schema (xem 1 mẫu)
    sample = vg_image_data[0]
    print(f"\nMẫu entry đầu tiên: {json.dumps(sample, indent=2)[:300]}")

    if "coco_id" not in sample:
        print("❌ Entry không có field 'coco_id' - kiểm tra lại file đã đúng chưa.")
        return

    # 2. Thống kê bao nhiêu ảnh VG có coco_id khác null
    vg_to_coco = {
        entry["image_id"]: entry["coco_id"]
        for entry in vg_image_data
        if entry.get("coco_id") is not None
    }
    print(f"\n--- Thống kê coco_id ---")
    print(f"Tổng số ảnh VG: {len(vg_image_data)}")
    print(f"Số ảnh VG có coco_id (không null): {len(vg_to_coco)} "
          f"({len(vg_to_coco)/len(vg_image_data)*100:.1f}%)")

    # 3. Load COCO image_id (train + val) để so khớp thực tế
    print("\n--- Đối chiếu với COCO 2017 bạn đã tải ---")
    coco_train_ids = load_coco_image_ids("train2017")
    coco_val_ids = load_coco_image_ids("val2017")
    coco_all_ids = coco_train_ids | coco_val_ids

    print(f"COCO train2017: {len(coco_train_ids)} ảnh")
    print(f"COCO val2017: {len(coco_val_ids)} ảnh")

    # coco_id trong VG thực chất CHÍNH LÀ COCO image_id
    matched_coco_ids = set(vg_to_coco.values()) & coco_all_ids
    matched_train = set(vg_to_coco.values()) & coco_train_ids
    matched_val = set(vg_to_coco.values()) & coco_val_ids

    print(f"\n✅ Số ảnh COCO (train+val) CÓ scene graph từ Visual Genome: {len(matched_coco_ids)}")
    print(f"   - Trong train2017: {len(matched_train)} / {len(coco_train_ids)} "
          f"({len(matched_train)/len(coco_train_ids)*100:.1f}%)")
    print(f"   - Trong val2017:   {len(matched_val)} / {len(coco_val_ids)} "
          f"({len(matched_val)/len(coco_val_ids)*100:.1f}%)")

    # 4. Lưu lại mapping đã lọc (chỉ giữ ảnh có match) để dùng cho bước sau
    output_path = VG_DIR / "vg_to_coco_mapping.json"
    # Đảo ngược: coco_id -> vg_image_id (để dễ tra cứu khi xử lý theo COCO image)
    coco_to_vg = {coco_id: vg_id for vg_id, coco_id in vg_to_coco.items() if coco_id in coco_all_ids}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_to_vg, f)

    print(f"\n💾 Đã lưu mapping (coco_id -> vg_image_id) tại: {output_path}")
    print(f"   Tổng số entries: {len(coco_to_vg)}")

    print("\n" + "=" * 60)
    if len(matched_coco_ids) / len(coco_all_ids) < 0.5:
        print("⚠️  CẢNH BÁO: Tỷ lệ khớp dưới 50% - cần lưu ý khi xây dataset,")
        print("   chỉ những ảnh có trong mapping mới dùng được cho semantic branch.")
    else:
        print("🎉 Tỷ lệ khớp tốt, có thể tiếp tục xây scene graph.")


if __name__ == "__main__":
    main()