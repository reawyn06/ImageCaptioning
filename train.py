"""
train.py  (ĐÃ CẬP NHẬT — chuyển từ GPT-2+prefix injection sang Transformer
Decoder tự huấn luyện, VÀ thêm Noam LR schedule -- xem 2 khối "THAY ĐỔI
QUAN TRỌNG" bên dưới)
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
    - Optimizer: AdamW + Noam LR schedule (xem bên dưới)
    - Batch size: 8
    - Epoch tối đa: 20, early stopping patience=3 (theo val loss)
    - Checkpoint: lưu best (val loss thấp nhất) + checkpoint cuối cùng

===========================================================================
THAY ĐỔI QUAN TRỌNG #1 (so với bản gốc dùng GPT-2 + prefix injection)
===========================================================================
Theo yêu cầu giáo viên hướng dẫn: thay CaptionDecoder (GPT-2 fine-tune +
ClipCap-style prefix injection) bằng CaptionDecoderTransformer (Transformer
Decoder chuẩn Vaswani et al. 2017, huấn luyện TỪ ĐẦU, có cross-attention
THẬT vào fused_features -- xem transformer_caption_decoder.py để biết chi
tiết lý do và kiến trúc).

HỆ QUẢ QUAN TRỌNG: checkpoint cũ (*_best.pt, *_last.pt của bản GPT-2) KHÔNG
CÒN TƯƠNG THÍCH với kiến trúc decoder mới -- bắt buộc phải train lại từ đầu
cho CẢ 4 STRATEGY. Đảm bảo đã chạy `python build_caption_vocab.py` trước khi
chạy file này (decoder mới cần features/caption_vocab.pt để khởi tạo vocab
+ embedding riêng, không dùng tokenizer GPT-2 nữa).

===========================================================================
THAY ĐỔI QUAN TRỌNG #2 (thêm sau lần train đầu tiên -- phát hiện qua kết quả eval)
===========================================================================
NGUYÊN NHÂN: Lần train đầu dùng LEARNING_RATE=5e-5 CỐ ĐỊNH -- mức này ĐÚNG
cho fine-tune model pretrained (GPT-2 cũ), nhưng SAI ngữ cảnh cho Transformer
decoder train TỪ ĐẦU (trọng số random hoàn toàn, cần LR cao hơn để học
nhanh trong ngân sách epoch giới hạn). Kết quả: Baseline (không có semantic
feature hỗ trợ) học rất chậm, sinh caption chung chung/lặp mẫu câu (BLEU-4
chỉ 0.0975, thấp hơn nhiều so với bản GPT-2 cũ 0.2507) -- trong khi 3
strategy có semantic feature (đặc biệt Bidirectional) vẫn học tốt vì có
thêm tín hiệu "dễ học hơn" (node đồ thị rời rạc) bù đắp cho LR chưa tối ưu.

FIX: Thêm Noam Learning Rate Schedule -- ĐÚNG công thức Vaswani et al. 2017,
"Attention Is All You Need", Mục 5.3:

    lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

LR tăng TUYẾN TÍNH trong `warmup_steps` bước đầu (tránh update quá mạnh khi
Adam's second-moment estimate còn chưa ổn định -- lý do gốc bài báo đưa ra
cho warmup), sau đó giảm dần theo `step^-0.5`. warmup_steps=2000 (thấp hơn
4000 mặc định của bài báo gốc, vì bài báo train trên tập WMT rất lớn nhiều
epoch/step hơn nhiều so với ~6,045 batch/epoch của dataset 48,365 ảnh ở đây
-- điều chỉnh cho phù hợp QUY MÔ dataset, có ghi rõ lý do thay vì copy máy
móc số của bài báo gốc).

===========================================================================
THAY ĐỔI QUAN TRỌNG #3 (sau khi thử Noam schedule THUẦN TÚY -- val_loss PHÂN KỲ)
===========================================================================
KẾT QUẢ QUAN SÁT ĐƯỢC: chạy Noam schedule nguyên bản (LR đỉnh ≈ 8.07e-4 tại
step=2000) cho strategy baseline -> val_loss TĂNG DẦN qua từng epoch (7.30 ->
7.54 -> 7.67 -> 7.84) trong khi train_loss vẫn giảm -- dấu hiệu PHÂN KỲ do LR
quá cao, không phải nhiễu ngẫu nhiên đơn thuần.

NGUYÊN NHÂN: công thức Noam gốc (Vaswani 2017) được tinh chỉnh cho batch
size hiệu dụng RẤT LỚN (hàng chục nghìn token/batch, do dùng token-based
batching trên tập WMT khổng lồ) -- gradient ước lượng được ở quy mô đó rất
"mượt" (ít nhiễu), chịu được LR đỉnh cao. Ở đây batch_size=8 nhỏ hơn RẤT
NHIỀU -- gradient nhiễu hơn đáng kể, cùng mức LR đỉnh gây update quá mạnh,
đẩy training ra khỏi vùng hội tụ tốt (val_loss phân kỳ dù train_loss vẫn
giảm -- KHÔNG phải overfitting thông thường, vì overfitting cần nhiều epoch
hơn để bộc lộ, còn đây phân kỳ ngay từ epoch 1).

FIX: thêm hệ số PEAK_LR_SCALE=0.25 nhân vào công thức Noam gốc -- có căn cứ
từ "linear scaling rule" (Goyal et al., "Accurate, Large Minibatch SGD",
2017): learning rate nên tỉ lệ thuận với batch size. Vì batch_size=8 ở đây
nhỏ hơn rất nhiều so với batch hiệu dụng của Transformer gốc, cần hệ số
giảm tương ứng. LR đỉnh mới ≈ 8.07e-4 * 0.25 ≈ 2.0e-4 -- vẫn cao hơn LR cố
định ban đầu (5e-5) để tránh underfitting đã quan sát ở THAY ĐỔI #2, nhưng
thấp hơn nhiều mức đã gây phân kỳ.
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rgcn_encoder import GloveVocab, RGCNEncoder
from fusion_module import build_fusion_module, HIDDEN_DIM  # THÊM: HIDDEN_DIM cho công thức Noam LR
from transformer_caption_decoder import CaptionDecoderTransformer  # THAY ĐỔI: import decoder mới
from caption_dataset import CaptionDataset, collate_fn


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
GLOVE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")
CAPTION_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "caption_vocab.pt")  # THÊM MỚI: vocab riêng cho decoder
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

BATCH_SIZE = 8
# THAY ĐỔI #2: bỏ LEARNING_RATE cố định -- LR thực tế giờ do Noam schedule
# quyết định theo từng training step (xem _noam_lr_factor() bên dưới).
# Optimizer khởi tạo với lr=1.0 làm "hệ số cơ sở", LambdaLR nhân thêm hệ số
# Noam vào -- đây là cách chuẩn để cắm Noam schedule vào PyTorch optimizer.
WARMUP_STEPS = 2000  # xem giải thích lý do chọn 2000 (không phải 4000 gốc) ở docstring đầu file
PEAK_LR_SCALE = 0.25  # THÊM (THAY ĐỔI #3): hạ LR đỉnh -- xem lý do ở docstring đầu file
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
MAX_CAPTION_LENGTH = 30
NUM_WORKERS = 2  # DataLoader workers -- giảm xuống 0 nếu gặp lỗi trên Windows


def _noam_lr_factor(step: int, hidden_dim: int = HIDDEN_DIM, warmup_steps: int = WARMUP_STEPS,
                     peak_scale: float = PEAK_LR_SCALE) -> float:
    """
    Công thức Noam LR schedule, Vaswani et al. 2017 (Mục 5.3), NHÂN THÊM hệ
    số peak_scale (xem THAY ĐỔI #3 ở docstring đầu file -- lý do: batch_size=8
    nhỏ hơn nhiều so với batch hiệu dụng của Transformer gốc, cần LR thấp
    hơn tương ứng theo "linear scaling rule"):
        lr(step) = peak_scale * d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
    step=0 được ép về 1 để tránh chia cho 0 / lũy thừa âm của 0 ở bước đầu.
    """
    step = max(step, 1)
    return peak_scale * (hidden_dim ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


# ============================================================
# Model wrapper — ghép R-GCN + Fusion + Decoder thành 1 nn.Module
# ============================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, strategy: str, glove_vocab: GloveVocab):
        super().__init__()
        self.strategy = strategy
        self.rgcn = RGCNEncoder(glove_vocab)
        self.fusion = build_fusion_module(strategy)
        # THAY ĐỔI: CaptionDecoderTransformer thay cho CaptionDecoder (GPT-2).
        # Cần truyền CAPTION_VOCAB_PATH vì decoder mới có vocab/embedding
        # riêng, học từ đầu -- không có pretrained tokenizer như GPT-2.
        self.decoder = CaptionDecoderTransformer(CAPTION_VOCAB_PATH)

    def compute_loss(self, visual_features, batch_objects, batch_triples, caption_texts):
        device = visual_features.device

        # ----- Semantic qua R-GCN (luôn chạy, ngay cả Baseline -- Baseline
        # fusion sẽ tự bỏ qua semantic, nhưng vẫn cần shape hợp lệ để gọi
        # forward() đồng nhất; tránh nhánh if/else rải rác trong code) -----
        semantic_features, semantic_mask = self.rgcn.forward_batch(batch_objects, batch_triples)

        # ----- Fusion -----
        fused_features, fused_mask = self.fusion(visual_features, semantic_features, semantic_mask)

        # ----- Tokenize caption + tính loss -----
        # encode_captions() ở decoder mới dùng tokenizer word-level riêng
        # (xem transformer_caption_decoder.py), KHÔNG còn dùng BPE của GPT-2,
        # nhưng interface gọi hàm giữ NGUYÊN như cũ -- không cần sửa gì thêm.
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
def run_epoch(model, loader, optimizer, device, train: bool, scheduler=None):
    """
    THAY ĐỔI: thêm tham số scheduler (tùy chọn, None khi validation). Noam
    schedule cập nhật LR theo TỪNG TRAINING STEP (từng batch), không phải
    theo epoch -- đúng công thức Vaswani 2017 lr(step), nên phải gọi
    scheduler.step() ngay sau optimizer.step(), bên trong vòng lặp batch.
    """
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
                # Gradient clipping -- vẫn giữ lại dù decoder mới không còn
                # fine-tune GPT-2 pretrained (rủi ro exploding gradient thấp
                # hơn trước), nhưng vẫn là thực hành an toàn chuẩn khi train
                # Transformer từ đầu (đặc biệt ở vài epoch đầu, embedding và
                # attention weight còn khởi tạo ngẫu nhiên).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

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

    # Kiểm tra sớm caption_vocab.pt đã tồn tại chưa -- báo lỗi rõ ràng ngay
    # từ đầu thay vì để crash giữa chừng lúc khởi tạo model.
    if not os.path.exists(CAPTION_VOCAB_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {CAPTION_VOCAB_PATH}. "
            f"Hãy chạy 'python build_caption_vocab.py' trước khi train."
        )

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
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể train: {num_params:,}")
    print(f"  (trong đó decoder: {num_decoder_params:,} -- so với GPT-2 fine-tune toàn bộ "
          f"trước đây là ~124,000,000)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    # THAY ĐỔI #2: LambdaLR nhân hệ số Noam vào lr=1.0 ở trên -- LR thực tế
    # tại mỗi step = 1.0 * _noam_lr_factor(step). Xem docstring đầu file để
    # biết lý do đổi từ LR cố định sang schedule này.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_noam_lr_factor)
    peak_lr = _noam_lr_factor(WARMUP_STEPS)  # LR đạt đỉnh đúng tại step = warmup_steps
    print(f"Noam LR schedule: warmup_steps={WARMUP_STEPS}, LR đỉnh dự kiến ≈ {peak_lr:.6f} "
          f"(tại step={WARMUP_STEPS}, tương đương ~{WARMUP_STEPS / 6045:.2f} epoch đầu)")

    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE)

    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
    last_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_last.pt")

    # ----- Training loop -----
    print(f"\nBắt đầu train (tối đa {MAX_EPOCHS} epoch, early stopping patience={EARLY_STOPPING_PATIENCE}) ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, scheduler=scheduler)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | LR={current_lr:.6f} | thời gian={elapsed:.1f}s")

        # Lưu checkpoint cuối (luôn ghi đè mỗi epoch)
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # THÊM: lưu kèm trạng thái Noam schedule
            "val_loss": val_loss,
        }, last_checkpoint_path)

        # Lưu checkpoint tốt nhất nếu cải thiện
        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),  # THÊM
                "val_loss": val_loss,
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