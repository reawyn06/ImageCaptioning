"""
extract_visual_features.py
============================
Trích xuất Visual Features từ ảnh COCO bằng ViT-B/16 pretrained (HuggingFace).

Ý tưởng:
- ViT chia ảnh thành các patch 16x16, mỗi patch được encode thành 1 vector 768-dim.
- Với input 224x224 -> (224/16)^2 = 196 patch + 1 [CLS] token = 197 token.
- Ta lấy toàn bộ patch embeddings (bỏ [CLS]) làm "visual feature sequence",
  vì Fusion Module (đặc biệt Cross-Attention) cần một SEQUENCE các vector,
  không phải 1 vector duy nhất (khác với cách dùng [CLS] cho classification).

  => Output mỗi ảnh: tensor shape (196, 768)
     Đây chính là "Visual Feature Vector" trong pipeline của bạn, ở dạng sequence.

Lưu kết quả ra .pt (PyTorch tensor) hoặc .npy để dùng lại, KHÔNG cần chạy ViT
lại mỗi lần train -> tiết kiệm thời gian training rất nhiều (vì ViT-B/16 không
cần fine-tune trong đề tài này, ta dùng feature extractor cố định - "frozen").

Chạy: python extract_visual_features.py
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm

# ====================== CẤU HÌNH ======================
BASE_DIR = Path(r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning")
COCO_IMAGES_DIR = BASE_DIR / "datasets" / "coco" / "images"
COCO_ANNOTATIONS_DIR = BASE_DIR / "datasets" / "coco" / "annotations"

# Nơi lưu visual features đã trích xuất (mỗi ảnh 1 file .pt)
OUTPUT_DIR = BASE_DIR / "features" / "visual"

MODEL_NAME = "google/vit-base-patch16-224-in21k"  # ViT-B/16, 768-dim output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # giảm từ 32 -> 8 để tránh ngốn RAM hệ thống khi load PIL Image hàng loạt

SPLITS = ["val2017", "train2017"]  # xử lý val trước (nhỏ) để test pipeline nhanh


def load_image_ids(split: str) -> list[int]:
    """Lấy danh sách image_id từ file annotation captions (đảm bảo khớp với ảnh thật dùng để train/eval)."""
    ann_path = COCO_ANNOTATIONS_DIR / f"captions_{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [img["id"] for img in data["images"]], {img["id"]: img["file_name"] for img in data["images"]}


def extract_features_for_split(split: str, processor, model):
    """Trích xuất visual feature cho toàn bộ ảnh trong 1 split (train2017 hoặc val2017)."""
    image_ids, id_to_filename = load_image_ids(split)

    split_output_dir = OUTPUT_DIR / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = COCO_IMAGES_DIR / split

    # Lọc ra những ảnh CHƯA được xử lý (cho phép resume nếu bị ngắt giữa đường)
    pending_ids = [
        img_id for img_id in image_ids
        if not (split_output_dir / f"{img_id}.pt").exists()
    ]

    print(f"\n[{split}] Tổng số ảnh: {len(image_ids)}, cần xử lý: {len(pending_ids)} "
          f"(đã có sẵn: {len(image_ids) - len(pending_ids)})")

    if not pending_ids:
        print(f"[{split}] Đã xử lý xong toàn bộ, bỏ qua.")
        return

    for i in tqdm(range(0, len(pending_ids), BATCH_SIZE), desc=f"Extracting {split}"):
        batch_ids = pending_ids[i:i + BATCH_SIZE]
        batch_images = []
        valid_ids = []

        for img_id in batch_ids:
            img_path = images_dir / id_to_filename[img_id]
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_ids.append(img_id)
            except Exception as e:
                print(f"⚠️  Lỗi đọc ảnh {img_path}: {e}")

        if not batch_images:
            continue

        # Tiền xử lý: resize, normalize theo chuẩn ViT pretrained
        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state shape: (batch, 197, 768) -> [CLS] + 196 patch tokens
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # bỏ [CLS], giữ 196 patch

        # Lưu từng ảnh ra 1 file riêng (.pt), giúp dễ load lại trong Dataset/DataLoader sau này
        for idx, img_id in enumerate(valid_ids):
            feature = patch_embeddings[idx].cpu()  # shape (196, 768)
            torch.save(feature, split_output_dir / f"{img_id}.pt")

        # Giải phóng RAM/VRAM ngay sau mỗi batch -> tránh tích lũy memory qua nhiều batch
        for img in batch_images:
            img.close()
        del inputs, outputs, patch_embeddings, batch_images
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("TRÍCH XUẤT VISUAL FEATURES (ViT-B/16)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print(f"\nĐang tải pretrained model: {MODEL_NAME} ...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()  # quan trọng: tắt dropout, vì ta chỉ inference, KHÔNG fine-tune ViT

    for split in SPLITS:
        extract_features_for_split(split, processor, model)

    print("\n🎉 Hoàn tất trích xuất visual features!")
    print(f"Kết quả lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()