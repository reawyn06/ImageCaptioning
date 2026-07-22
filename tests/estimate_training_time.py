"""
estimate_training_time.py  (ĐÃ SỬA — xem khối "FIX" bên dưới)
======================
Mục đích:
    Chạy thử vài batch đầu (KHÔNG chạy hết epoch) để đo thời gian thực tế
    trên máy Rea, từ đó ước lượng:
        - Thời gian 1 epoch đầy đủ (48,365 ảnh train)
        - Thời gian cho cả quá trình train 1 strategy (tối đa 20 epoch)
        - Thời gian cho cả 4 strategy

Cách chạy (từ thư mục gốc project):
    python tests\\estimate_training_time.py --strategy baseline

===========================================================================
FIX — THIẾU sys.path.insert(PROJECT_ROOT)
===========================================================================
Bản gốc import trực tiếp `from rgcn_encoder import GloveVocab`, `from
caption_dataset import ...`, `from train import ...` -- các module này nằm
ở THƯ MỤC GỐC project, không nằm trong tests/. Khi chạy `python
tests\\estimate_training_time.py` từ project root, Python chỉ tự thêm thư
mục CHỨA FILE ĐANG CHẠY (tests/) vào sys.path, KHÔNG tự thêm project root
-> ModuleNotFoundError: No module named 'rgcn_encoder'.

Đây là lỗi CÓ SẴN từ trước (không liên quan gì đến việc đổi Transformer
Decoder) -- so sánh evaluate_flickr30k.py và scripts/build_flickr30k_features.py
đã xử lý đúng việc này bằng sys.path.insert(0, PROJECT_ROOT), file này bị
thiếu bước tương tự. Bản vá thêm đúng 3 dòng đó ở đầu file, TRƯỚC các câu
lệnh import module gốc.
"""

import argparse
import os
import sys
import time

# FIX: thêm project root vào sys.path TRƯỚC khi import các module ở thư mục
# gốc (rgcn_encoder, caption_dataset, train) -- bắt buộc vì file này nằm
# trong subfolder tests/, không tự "nhìn thấy" các module ở thư mục cha.
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from rgcn_encoder import GloveVocab
from caption_dataset import CaptionDataset, collate_fn
from train import ImageCaptioningModel, PROJECT_ROOT, GLOVE_VOCAB_PATH, BATCH_SIZE, MAX_EPOCHS

NUM_BATCHES_TO_TEST = 10  # số batch đầu dùng để đo thời gian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["baseline", "concat", "one_directional", "bidirectional"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Strategy: {args.strategy} | Device: {device}")

    print("\nĐang load dataset (chỉ cần load 1 lần) ...")
    train_dataset = CaptionDataset(PROJECT_ROOT, "train2017")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,  # 0 để đo thời gian thuần, tránh nhiễu do prefetch
    )

    total_train_images = len(train_dataset)
    num_batches_per_epoch = total_train_images // BATCH_SIZE

    print(f"\nĐang khởi tạo model '{args.strategy}' ...")
    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    model = ImageCaptioningModel(args.strategy, glove_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    print(f"\nĐang đo thời gian cho {NUM_BATCHES_TO_TEST} batch đầu (batch_size={BATCH_SIZE}) ...")
    print("(Bỏ qua batch đầu tiên khi tính trung bình -- chứa overhead CUDA warm-up)")

    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= NUM_BATCHES_TO_TEST:
            break

        t0 = time.time()

        visual_features = batch["visual_features"].to(device)
        loss = model.compute_loss(
            visual_features, batch["batch_objects"], batch["batch_triples"], batch["captions"]
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()  # đảm bảo đo đúng thời gian GPU thực tế, không chỉ thời gian "submit" kernel

        elapsed = time.time() - t0
        batch_times.append(elapsed)
        print(f"  Batch {i+1}/{NUM_BATCHES_TO_TEST}: {elapsed:.3f}s, loss={loss.item():.4f}")

    # Bỏ batch đầu tiên (CUDA warm-up) khi tính trung bình, nếu có ít nhất 2 batch
    times_for_avg = batch_times[1:] if len(batch_times) > 1 else batch_times
    avg_batch_time = sum(times_for_avg) / len(times_for_avg)

    estimated_epoch_time = avg_batch_time * num_batches_per_epoch
    estimated_total_time = estimated_epoch_time * MAX_EPOCHS

    print("\n" + "=" * 60)
    print("ƯỚC LƯỢNG THỜI GIAN")
    print("=" * 60)
    print(f"Thời gian trung bình/batch (đã loại batch warm-up): {avg_batch_time:.3f}s")
    print(f"Số batch/epoch (toàn bộ {total_train_images} ảnh, batch_size={BATCH_SIZE}): {num_batches_per_epoch}")
    print(f"Ước lượng thời gian 1 epoch: {estimated_epoch_time:.1f}s (~{estimated_epoch_time/60:.1f} phút)")
    print(f"Ước lượng thời gian TỐI ĐA (nếu chạy hết {MAX_EPOCHS} epoch, "
          f"KHÔNG early stop): {estimated_total_time/3600:.2f} giờ")
    print("\n(Lưu ý: đây CHƯA tính thời gian validation mỗi epoch, và thực tế có thể")
    print(" dừng sớm hơn nhờ early stopping patience=3 -- thời gian thực tế thường thấp hơn số trên.)")


if __name__ == "__main__":
    main()