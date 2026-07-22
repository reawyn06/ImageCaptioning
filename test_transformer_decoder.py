"""
test_transformer_decoder.py
======================
Mục đích:
    Smoke test nhanh cho CaptionDecoderTransformer -- kiểm tra shape, NaN/Inf,
    và encode/decode round-trip TRƯỚC KHI chạy train.py đầy đủ (mỗi lần train
    1 strategy có thể mất vài giờ -- không muốn phát hiện lỗi import/shape
    sau khi đã chờ hàng giờ).

Điều kiện chạy: đã chạy `python build_caption_vocab.py` trước đó (script này
sẽ tự kiểm tra và báo lỗi rõ ràng nếu chưa có features/caption_vocab.pt).

Cách chạy:
    python test_transformer_decoder.py
"""

import os
import sys

import torch

PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
CAPTION_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "caption_vocab.pt")

if not os.path.exists(CAPTION_VOCAB_PATH):
    print(f"❌ Chưa tìm thấy {CAPTION_VOCAB_PATH}")
    print("   Hãy chạy 'python build_caption_vocab.py' trước.")
    sys.exit(1)

from transformer_caption_decoder import CaptionDecoderTransformer


def main():
    print("=" * 60)
    print("SMOKE TEST — CaptionDecoderTransformer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/5] Đang khởi tạo decoder từ vocab đã build ...")
    decoder = CaptionDecoderTransformer(CAPTION_VOCAB_PATH).to(device)
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  -> Vocab size: {decoder.vocab_size}")
    print(f"  -> Tổng số tham số có thể train: {num_params:,} "
          f"(so sánh: GPT-2 base fine-tune toàn bộ ~124,000,000 tham số)")

    print("\n[2/5] Đang test encode_captions() (tokenize + BOS/EOS/pad) ...")
    sample_captions = [
        "a man riding a bike down the street",
        "a dog running in the park with a ball",
    ]
    caption_ids, caption_mask = decoder.encode_captions(sample_captions)
    print(f"  -> caption_ids shape: {tuple(caption_ids.shape)}")
    print(f"  -> caption_mask shape: {tuple(caption_mask.shape)}")
    print(f"  -> Ví dụ decode ngược lại (round-trip check):")
    for i in range(len(sample_captions)):
        ids = caption_ids[i].tolist()
        decoded = decoder._ids_to_text(ids[1:])  # bỏ BOS, _ids_to_text tự cắt ở EOS
        print(f"     gốc:    '{sample_captions[i]}'")
        print(f"     decode: '{decoded}'  (nếu có <unk> nghĩa là từ đó hiếm/không có trong train2017)")

    print("\n[3/5] Đang test forward() + compute_loss() với dữ liệu giả lập ...")
    caption_ids = caption_ids.to(device)
    caption_mask = caption_mask.to(device)

    # Giả lập fused_features với 2 shape khác nhau (196 cho Baseline/Concat,
    # 196+N cho One-directional/Bidirectional) để đảm bảo decoder hoạt động
    # đúng với MỌI độ dài prefix có thể có từ Fusion Module.
    for L in [196, 196 + 7]:
        fused_features = torch.randn(2, L, 768, device=device)
        fused_mask = torch.ones(2, L, dtype=torch.bool, device=device)

        loss = decoder.compute_loss(fused_features, fused_mask, caption_ids, caption_mask)
        has_nan = torch.isnan(loss).item()
        has_inf = torch.isinf(loss).item()
        print(f"  -> L={L}: loss={loss.item():.4f}, NaN={has_nan}, Inf={has_inf}")
        assert not has_nan and not has_inf, f"Loss có NaN/Inf với L={L}!"

    print("\n[4/5] Đang test generate() — greedy ...")
    fused_features = torch.randn(2, 196, 768, device=device)
    fused_mask = torch.ones(2, 196, dtype=torch.bool, device=device)
    captions_greedy = decoder.generate(fused_features, fused_mask, max_length=20, method="greedy")
    for i, cap in enumerate(captions_greedy):
        print(f"  -> Ảnh {i+1} (chưa train, kỳ vọng caption VÔ NGHĨA vì weight random): '{cap}'")

    print("\n[5/5] Đang test generate() — beam search (batch_size=1) ...")
    fused_features_1 = torch.randn(1, 196, 768, device=device)
    fused_mask_1 = torch.ones(1, 196, dtype=torch.bool, device=device)
    caption_beam = decoder.generate(fused_features_1, fused_mask_1, max_length=20,
                                     method="beam", num_beams=4)
    print(f"  -> Beam search output: '{caption_beam[0]}'")

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT — Decoder chạy đúng shape, không NaN/Inf.")
    print("   (Caption vô nghĩa ở bước 4-5 là BÌNH THƯỜNG vì model chưa train")
    print("    -- mục đích test này chỉ để bắt lỗi shape/import trước khi train thật.)")
    print("=" * 60)


if __name__ == "__main__":
    main()