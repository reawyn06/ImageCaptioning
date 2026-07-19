"""
train.py
======================
Mục đích:
    Training script đầy đủ để chạy 1 trong 4 thực nghiệm fusion strategy.
    Ghép toàn bộ pipeline: RGCNEncoder + Fusion Module + CaptionDecoder
    thành 1 model, train bằng teacher forcing, có validation + early
    stopping + lưu checkpoint (best val loss + checkpoint cuối).

Cách chạy (đổi STRATEGY để chạy thực nghiệm khác nhau):
    python train.py --strategy baseline
    python train.py --strategy concat
    python train.py --strategy one_directional
    python train.py --strategy bidirectional

Thiết kế đã chốt:
    - Caption: random 1/5 mỗi epoch (xử lý tự động trong CaptionDataset)
    - Optimizer: AdamW, lr=5e-5
    - Batch size: 8
    - Epoch tối đa: 20, early stopping patience=3 (theo val loss)
    - Checkpoint: lưu best (val loss thấp nhất) + checkpoint cuối cùng
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rgcn_encoder import GloveVocab, RGCNEncoder
from fusion_module import build_fusion_module
from caption_decoder import CaptionDecoder
from caption_dataset import CaptionDataset, collate_fn


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
GLOVE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
MAX_CAPTION_LENGTH = 30
NUM_WORKERS = 2  # DataLoader workers -- giảm xuống 0 nếu gặp lỗi trên Windows


# ============================================================
# Model wrapper — ghép R-GCN + Fusion + Decoder thành 1 nn.Module
# ============================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, strategy: str, glove_vocab: GloveVocab):
        super().__init__()
        self.strategy = strategy
        self.rgcn = RGCNEncoder(glove_vocab)
        self.fusion = build_fusion_module(strategy)
        self.decoder = CaptionDecoder()

    def compute_loss(self, visual_features, batch_objects, batch_triples, caption_texts):
        device = visual_features.device

        # ----- Semantic qua R-GCN (luôn chạy, ngay cả Baseline -- Baseline
        # fusion sẽ tự bỏ qua semantic, nhưng vẫn cần shape hợp lệ để gọi
        # forward() đồng nhất; tránh nhánh if/else rải rác trong code) -----
        semantic_features, semantic_mask = self.rgcn.forward_batch(batch_objects, batch_triples)

        # ----- Fusion -----
        fused_features, fused_mask = self.fusion(visual_features, semantic_features, semantic_mask)

        # ----- Tokenize caption + tính loss -----
        caption_ids, caption_mask = self.decoder.encode_captions(caption_texts, max_length=MAX_CAPTION_LENGTH)
        caption_ids = caption_ids.to(device)
        caption_mask = caption_mask.to(device)

        loss = self.decoder.compute_loss(fused_features, fused_mask, caption_ids, caption_mask)
        return loss


# ============================================================
# Early Stopping helper
# ============================================================
class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Trả về True nếu val_loss này là tốt nhất từ trước đến nay."""
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_best


# ============================================================
# Training / Validation loop cho 1 epoch
# ============================================================
def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    num_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            visual_features = batch["visual_features"].to(device)
            batch_objects = batch["batch_objects"]
            batch_triples = batch["batch_triples"]
            captions = batch["captions"]

            loss = model.compute_loss(visual_features, batch_objects, batch_triples, captions)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping -- quan trọng khi fine-tune GPT-2, tránh
                # exploding gradient làm hỏng pretrained weight.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["baseline", "concat", "one_directional", "bidirectional"])
    args = parser.parse_args()
    strategy = args.strategy

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Strategy: {strategy} | Device: {device}")

    # ----- Load dữ liệu -----
    print("\nĐang load dataset ...")
    train_dataset = CaptionDataset(PROJECT_ROOT, "train2017")
    val_dataset = CaptionDataset(PROJECT_ROOT, "val2017")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
    )

    # ----- Khởi tạo model -----
    print("\nĐang khởi tạo model ...")
    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    model = ImageCaptioningModel(strategy, glove_vocab).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể train: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE)

    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
    last_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_last.pt")

    # ----- Training loop -----
    print(f"\nBắt đầu train (tối đa {MAX_EPOCHS} epoch, early stopping patience={EARLY_STOPPING_PATIENCE}) ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | thời gian={elapsed:.1f}s")

        # Lưu checkpoint cuối (luôn ghi đè mỗi epoch)
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
        }, last_checkpoint_path)

        # Lưu checkpoint tốt nhất nếu cải thiện
        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
            }, best_checkpoint_path)
            print(f"  -> Lưu checkpoint tốt nhất mới (val_loss={val_loss:.4f})")

        if early_stopping.should_stop:
            print(f"\nEarly stopping tại epoch {epoch} "
                  f"(val_loss không cải thiện sau {EARLY_STOPPING_PATIENCE} epoch liên tiếp).")
            break

    print(f"\nHoàn tất train strategy '{strategy}'. "
          f"Best val_loss: {early_stopping.best_loss:.4f}")
    print(f"Checkpoint tốt nhất: {best_checkpoint_path}")
    print(f"Checkpoint cuối: {last_checkpoint_path}")


if __name__ == "__main__":
    main()