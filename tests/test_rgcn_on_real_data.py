"""
test_rgcn_on_real_data.py
======================
Mục đích:
    Kiểm tra RGCNEncoder hoạt động đúng với DỮ LIỆU THẬT (không phải dummy vocab):
        - Load glove_vocab.pt thật (44,523 object + 19,670 predicate)
        - Lấy vài file scene graph thật từ features/semantic/train2017/
        - Chạy forward() + forward_batch(), kiểm tra output hợp lý (không NaN,
          không Inf, shape đúng)
        - Đo thời gian chạy 1 batch để ước lượng tốc độ train sau này

Cách chạy:
    python test_rgcn_on_real_data.py
"""

import os
import time
import json

import torch

from rgcn_encoder import GloveVocab, RGCNEncoder, load_scene_graph, HIDDEN_DIM

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
GLOVE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")
TRAIN_SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "features", "semantic", "train2017")

NUM_SAMPLE_IMAGES = 8  # số ảnh thật lấy ra để test


def main():
    print("=" * 60)
    print("TEST RGCNEncoder VỚI DỮ LIỆU THẬT")
    print("=" * 60)

    # ----- Bước 1: Load GloveVocab thật -----
    print("\nĐang load glove_vocab.pt thật ...")
    t0 = time.time()
    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    print(f"  -> Load xong trong {time.time() - t0:.2f}s")
    print(f"  -> Object vocab: {len(glove_vocab.object_vocab)}")
    print(f"  -> Predicate vocab: {len(glove_vocab.predicate_vocab)}")

    # ----- Bước 2: Khởi tạo encoder (dùng GPU nếu có, giống lúc train thật) -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    encoder = RGCNEncoder(glove_vocab).to(device)
    # object_glove/predicate_glove là buffer nên .to(device) đã tự chuyển theo,
    # nhưng object_indices()/predicate_indices() vẫn tạo tensor trên CPU rồi
    # .to(device) lại trong forward() -- không cần xử lý thêm gì ở đây.

    # ----- Bước 3: Lấy vài ảnh thật để test -----
    print(f"\nĐang lấy {NUM_SAMPLE_IMAGES} ảnh thật từ {TRAIN_SEMANTIC_DIR} ...")
    files = [f for f in os.listdir(TRAIN_SEMANTIC_DIR) if f.endswith(".json")][:NUM_SAMPLE_IMAGES]

    if not files:
        print("❌ KHÔNG TÌM THẤY FILE SCENE GRAPH NÀO. Kiểm tra lại đường dẫn TRAIN_SEMANTIC_DIR.")
        return

    batch_objects = []
    batch_triples = []
    for fname in files:
        fpath = os.path.join(TRAIN_SEMANTIC_DIR, fname)
        objects, triples = load_scene_graph(fpath)
        batch_objects.append(objects)
        batch_triples.append(triples)
        print(f"  - {fname}: {len(objects)} object, {len(triples)} triple")

    # ----- Bước 4: Test forward() từng ảnh riêng lẻ -----
    print("\nĐang test forward() từng ảnh riêng lẻ ...")
    for i, (objects, triples) in enumerate(zip(batch_objects, batch_triples)):
        out = encoder(objects, triples)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"  Ảnh {i+1}: output shape={tuple(out.shape)}, "
              f"mean={out.mean().item():.4f}, std={out.std().item():.4f}, "
              f"NaN={has_nan}, Inf={has_inf}")
        assert not has_nan, f"Ảnh {i+1} có NaN trong output!"
        assert not has_inf, f"Ảnh {i+1} có Inf trong output!"
        assert out.shape == (len(objects), HIDDEN_DIM), f"Ảnh {i+1} sai shape!"

    # ----- Bước 5: Test forward_batch() + đo thời gian -----
    print("\nĐang test forward_batch() (đo thời gian) ...")
    t0 = time.time()
    padded, mask = encoder.forward_batch(batch_objects, batch_triples)
    elapsed = time.time() - t0

    print(f"  -> Batch shape: {padded.shape}")
    print(f"  -> Mask shape: {mask.shape}")
    print(f"  -> Thời gian xử lý batch {NUM_SAMPLE_IMAGES} ảnh: {elapsed:.4f}s "
          f"({elapsed/NUM_SAMPLE_IMAGES*1000:.2f} ms/ảnh)")

    has_nan = torch.isnan(padded).any().item()
    has_inf = torch.isinf(padded).any().item()
    print(f"  -> NaN trong batch: {has_nan}, Inf trong batch: {has_inf}")
    assert not has_nan and not has_inf

    # Kiểm tra mask khớp đúng số node thật của từng ảnh
    for i, objects in enumerate(batch_objects):
        expected_count = len(objects)
        actual_count = mask[i].sum().item()
        status = "✅" if expected_count == actual_count else "❌"
        print(f"  {status} Ảnh {i+1}: mask đếm {actual_count} node, thực tế có {expected_count} object")

    # ----- Bước 6: Ước lượng thời gian cho toàn bộ epoch -----
    total_train_images = 48365
    estimated_batch_size = 32
    num_batches = total_train_images // estimated_batch_size
    time_per_image = elapsed / NUM_SAMPLE_IMAGES
    estimated_epoch_time = time_per_image * total_train_images

    print(f"\n--- Ước lượng thời gian (chỉ riêng phần R-GCN, CHƯA gồm visual/decoder) ---")
    print(f"Với batch_size={estimated_batch_size}, ~{num_batches} batch/epoch")
    print(f"Ước lượng thời gian R-GCN cho 1 epoch (48,365 ảnh): {estimated_epoch_time:.1f}s "
          f"(~{estimated_epoch_time/60:.1f} phút)")
    print("(Lưu ý: đây chỉ là ước lượng sơ bộ dựa trên forward_batch() chạy tuần tự "
          "từng ảnh, CHƯA tính backward pass, chưa gồm visual encoder/decoder.)")

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT TEST — RGCNEncoder hoạt động đúng trên dữ liệu thật.")
    print("=" * 60)


if __name__ == "__main__":
    main()