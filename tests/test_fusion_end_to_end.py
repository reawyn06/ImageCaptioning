"""
test_fusion_end_to_end.py
======================
Mục đích:
    Test TÍCH HỢP toàn bộ pipeline đã code đến hiện tại trên dữ liệu thật:
        1. Load visual feature đã trích sẵn (features/visual/train2017/{coco_id}.pt)
        2. Load semantic feature (scene graph) + chạy qua R-GCN encoder
        3. Đưa cả 2 qua Fusion Module (chạy thử cả 4 strategy)
    Mục đích là xác nhận pipeline chạy được END-TO-END, không lỗi shape/device,
    trước khi chuyển sang code Transformer Decoder.

Cách chạy:
    python test_fusion_end_to_end.py
"""

import os
import time

import torch

from rgcn_encoder import GloveVocab, RGCNEncoder, load_scene_graph
from fusion_module import build_fusion_module, HIDDEN_DIM

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
GLOVE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")
SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "features", "semantic", "train2017")
VISUAL_DIR = os.path.join(PROJECT_ROOT, "features", "visual", "train2017")

NUM_SAMPLE_IMAGES = 4  # số ảnh thật lấy ra để test (nhỏ để chạy nhanh, dễ nhìn log)


def main():
    print("=" * 60)
    print("TEST END-TO-END: VISUAL + R-GCN + FUSION MODULE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ----- Bước 1: Load GloveVocab + R-GCN Encoder -----
    print("\n[1/4] Đang load GloveVocab + khởi tạo R-GCN Encoder ...")
    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    rgcn = RGCNEncoder(glove_vocab).to(device)
    rgcn.eval()  # chế độ eval vì đây chỉ là test forward, chưa train
    print("  -> OK")

    # ----- Bước 2: Lấy danh sách ảnh có ĐỦ CẢ visual feature VÀ semantic feature -----
    print(f"\n[2/4] Đang tìm {NUM_SAMPLE_IMAGES} ảnh có đủ cả visual + semantic feature ...")
    semantic_files = {f.replace(".json", "") for f in os.listdir(SEMANTIC_DIR) if f.endswith(".json")}
    visual_files = {f.replace(".pt", "") for f in os.listdir(VISUAL_DIR) if f.endswith(".pt")}
    common_ids = list(semantic_files & visual_files)[:NUM_SAMPLE_IMAGES]

    if not common_ids:
        print("❌ KHÔNG TÌM THẤY ẢNH NÀO CÓ ĐỦ CẢ 2 LOẠI FEATURE. Kiểm tra lại đường dẫn VISUAL_DIR/SEMANTIC_DIR.")
        return

    print(f"  -> Tìm được: {common_ids}")

    # ----- Bước 3: Load visual feature + semantic feature, chạy qua R-GCN -----
    print("\n[3/4] Đang load visual feature + chạy semantic qua R-GCN encoder ...")
    batch_visual = []
    batch_objects = []
    batch_triples = []

    for coco_id in common_ids:
        # Visual feature: đã trích sẵn, shape (196, 768)
        visual_path = os.path.join(VISUAL_DIR, f"{coco_id}.pt")
        visual_feat = torch.load(visual_path, weights_only=False)
        if isinstance(visual_feat, dict):
            # Một số cách lưu có thể bọc trong dict (vd {"features": tensor}) -- xử lý linh hoạt
            visual_feat = visual_feat.get("features", visual_feat.get("embeddings"))
        batch_visual.append(visual_feat)

        # Semantic: đọc scene graph, CHƯA chạy R-GCN ở đây (chạy theo batch ở dưới)
        semantic_path = os.path.join(SEMANTIC_DIR, f"{coco_id}.json")
        objects, triples = load_scene_graph(semantic_path)
        batch_objects.append(objects)
        batch_triples.append(triples)

        print(f"  - coco_id={coco_id}: visual shape={tuple(visual_feat.shape)}, "
              f"{len(objects)} object, {len(triples)} triple")

    visual_features = torch.stack(batch_visual).to(device)  # (B, 196, 768)
    print(f"\n  -> Visual batch shape: {tuple(visual_features.shape)}")

    with torch.no_grad():
        semantic_features, semantic_mask = rgcn.forward_batch(batch_objects, batch_triples)
    print(f"  -> Semantic batch shape: {tuple(semantic_features.shape)}, mask shape: {tuple(semantic_mask.shape)}")

    # ----- Bước 4: Chạy qua cả 4 Fusion Strategy -----
    print("\n[4/4] Đang chạy qua cả 4 Fusion Strategy ...")
    for strategy in ["baseline", "concat", "one_directional", "bidirectional"]:
        fusion = build_fusion_module(strategy).to(device)
        fusion.eval()

        t0 = time.time()
        with torch.no_grad():
            fused, fused_mask = fusion(visual_features, semantic_features, semantic_mask)
        elapsed = time.time() - t0

        has_nan = torch.isnan(fused).any().item()
        has_inf = torch.isinf(fused).any().item()

        print(f"\n  Strategy: {strategy}")
        print(f"    Output shape: {tuple(fused.shape)}, mask shape: {tuple(fused_mask.shape)}")
        print(f"    mean={fused.mean().item():.4f}, std={fused.std().item():.4f}")
        print(f"    NaN={has_nan}, Inf={has_inf}, thời gian={elapsed*1000:.2f}ms")

        assert not has_nan, f"Strategy {strategy} có NaN!"
        assert not has_inf, f"Strategy {strategy} có Inf!"

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT — Pipeline Visual + R-GCN + Fusion chạy đúng END-TO-END.")
    print("=" * 60)


if __name__ == "__main__":
    main()