"""
evaluate.py
======================
Mục đích:
    Đánh giá 4 checkpoint (1 cho mỗi fusion strategy) trên tập validation,
    bằng cách:
        1. Sinh caption (greedy decoding, dùng decoder.generate() đã có sẵn)
           cho toàn bộ ảnh trong val2017 (2,135 ảnh có đủ semantic feature).
        2. Tính 4 chỉ số: BLEU (1-4), CIDEr, METEOR, SPICE -- dùng
           pycocoevalcap (bộ chuẩn của MS-COCO, đúng cách các paper báo cáo).
        3. In bảng so sánh tổng hợp cho cả 4 strategy.

Cách chạy:
    python evaluate.py                     # đánh giá cả 4 strategy
    python evaluate.py --strategy concat   # chỉ đánh giá 1 strategy

Lưu ý quan trọng:
    - LẦN ĐẦU chạy, pycocoevalcap (phần SPICE) sẽ tự tải Stanford CoreNLP
      (~380MB) từ internet -- cần thời gian, chỉ tải 1 lần, lưu cache.
    - Việc đánh giá dùng GROUND TRUTH là TOÀN BỘ 5 caption/ảnh (không phải
      random 1/5 như lúc train) -- đây là cách đánh giá CHUẨN của COCO,
      caption sinh ra được so với ĐẦY ĐỦ các caption tham chiếu để công bằng.
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from meteor_fixed import MeteorFixed  # Dùng bản đã patch lỗi buffering trên Windows
                                        # (xem meteor_fixed.py để biết chi tiết nguyên nhân)

from rgcn_encoder import GloveVocab
from train import ImageCaptioningModel, PROJECT_ROOT, GLOVE_VOCAB_PATH, BATCH_SIZE
from caption_dataset import CaptionDataset, collate_fn


CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MAX_GEN_LENGTH = 30

ALL_STRATEGIES = ["baseline", "concat", "one_directional", "bidirectional"]


# ============================================================
# BƯỚC 1 — Sinh caption cho toàn bộ val set bằng 1 checkpoint
# ============================================================
@torch.no_grad()
def generate_captions_for_strategy(strategy: str, device) -> dict:
    """
    Load checkpoint best của 1 strategy, sinh caption (greedy) cho toàn bộ
    ảnh trong val2017. Trả về dict {coco_id_str: caption_string}.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
    print(f"\n[{strategy}] Đang load checkpoint: {checkpoint_path} ...")

    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    model = ImageCaptioningModel(strategy, glove_vocab).to(device)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  -> Checkpoint từ epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    val_dataset = CaptionDataset(PROJECT_ROOT, "val2017")
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    generated = {}
    total = len(val_dataset)
    done = 0

    for batch in val_loader:
        visual_features = batch["visual_features"].to(device)
        batch_objects = batch["batch_objects"]
        batch_triples = batch["batch_triples"]
        coco_ids = batch["coco_ids"]

        semantic_features, semantic_mask = model.rgcn.forward_batch(batch_objects, batch_triples)
        fused_features, fused_mask = model.fusion(visual_features, semantic_features, semantic_mask)

        captions = model.decoder.generate(fused_features, fused_mask, max_length=MAX_GEN_LENGTH)

        for coco_id, caption in zip(coco_ids, captions):
            generated[str(coco_id)] = caption

        done += len(coco_ids)
        if done % (BATCH_SIZE * 20) == 0 or done == total:
            print(f"  -> Đã sinh caption cho {done}/{total} ảnh")

    return generated


# ============================================================
# BƯỚC 2 — Load TOÀN BỘ 5 ground-truth caption/ảnh (không random như train)
# ============================================================
def load_full_ground_truth() -> dict:
    """
    Đọc captions_val2017.json, trả về dict {coco_id_str: [list 5 caption]}
    CHỈ cho các ảnh có trong val_dataset (đã có đủ visual+semantic feature).
    """
    captions_path = os.path.join(PROJECT_ROOT, "datasets", "coco", "annotations", "captions_val2017.json")
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gts = {}
    for ann in data["annotations"]:
        img_id = str(ann["image_id"])
        gts.setdefault(img_id, []).append(ann["caption"])

    return gts


# ============================================================
# BƯỚC 3 — Tính 4 chỉ số đánh giá
# ============================================================
def compute_metrics(gts: dict, res: dict) -> dict:
    """
    gts: {coco_id: [list caption tham chiếu]}
    res: {coco_id: [1 caption sinh ra]}   -- pycocoevalcap yêu cầu res cũng
         là list (dù chỉ có 1 phần tử), không phải string trực tiếp.

    Trả về dict {metric_name: score}.
    """
    metrics = {}

    print("  Đang tính BLEU ...")
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    metrics["BLEU-1"] = bleu_scores[0]
    metrics["BLEU-2"] = bleu_scores[1]
    metrics["BLEU-3"] = bleu_scores[2]
    metrics["BLEU-4"] = bleu_scores[3]

    print("  Đang tính METEOR (cần Java, dùng bản đã patch lỗi buffering Windows) ...")
    meteor_scorer = MeteorFixed()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    metrics["METEOR"] = meteor_score

    print("  Đang tính CIDEr ...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    metrics["CIDEr"] = cider_score

    print("  Đang tính SPICE (cần Java, lần đầu sẽ tải Stanford CoreNLP ~380MB) ...")
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts, res)
    metrics["SPICE"] = spice_score

    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None,
                        choices=ALL_STRATEGIES,
                        help="Chỉ đánh giá 1 strategy. Bỏ qua để đánh giá cả 4.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES

    print("\nĐang load ground-truth caption (đầy đủ 5 caption/ảnh) ...")
    full_gts = load_full_ground_truth()

    all_results = {}

    for strategy in strategies:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Bỏ qua '{strategy}': không tìm thấy checkpoint tại {checkpoint_path}")
            continue

        # ----- Sinh caption -----
        generated = generate_captions_for_strategy(strategy, device)

        # Lưu caption sinh ra ra file riêng (để xem qua, debug, hoặc đưa vào báo cáo)
        gen_path = os.path.join(RESULTS_DIR, f"{strategy}_generated_captions.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, ensure_ascii=False, indent=2)
        print(f"  -> Đã lưu caption sinh ra tại: {gen_path}")

        # ----- Chuẩn bị gts/res ĐÚNG FORMAT pycocoevalcap, khớp đúng tập ảnh -----
        common_ids = [cid for cid in generated.keys() if cid in full_gts]
        gts = {cid: full_gts[cid] for cid in common_ids}
        res = {cid: [generated[cid]] for cid in common_ids}

        print(f"\n[{strategy}] Đang tính 4 chỉ số trên {len(common_ids)} ảnh ...")
        metrics = compute_metrics(gts, res)
        all_results[strategy] = metrics

        print(f"\n[{strategy}] KẾT QUẢ:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ----- Bảng tổng hợp so sánh cả 4 strategy -----
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("BẢNG SO SÁNH TỔNG HỢP")
        print("=" * 80)
        metric_names = list(next(iter(all_results.values())).keys())
        header = f"{'Strategy':<18}" + "".join(f"{m:>10}" for m in metric_names)
        print(header)
        for strategy, metrics in all_results.items():
            row = f"{strategy:<18}" + "".join(f"{metrics[m]:>10.4f}" for m in metric_names)
            print(row)

    # ----- Lưu kết quả tổng hợp ra JSON -----
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tổng hợp tại: {results_path}")


if __name__ == "__main__":
    main()