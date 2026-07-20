"""
evaluate.py  (ĐÃ VÁ LỖI — xem khối "FIX #1" và "FIX #2" trong file)
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

===========================================================================
GHI CHÚ VỀ 2 LỖI ĐÃ SỬA (SO VỚI BẢN GỐC) -- LÝ DO BLEU/CIDEr THẤP HƠN KỲ VỌNG
===========================================================================

FIX #1 -- THIẾU BƯỚC PTBTokenizer (nguyên nhân chính khiến điểm thấp):
    Toàn bộ pipeline chuẩn của pycocoevalcap (repo gốc salesforce/coco-caption
    -- nơi MỌI paper báo cáo BLEU/CIDEr/SPICE đều dùng) YÊU CẦU chạy qua
    Stanford PTBTokenizer TRƯỚC khi đưa gts/res vào Bleu()/Cider()/Spice().
    Bản gốc của file này bỏ qua bước này, đưa thẳng text thô (còn nguyên dấu
    câu, hoa/thường không chuẩn hóa) vào scorer -- các scorer này chỉ tách
    từ bằng khoảng trắng (.split()), nên "bike." và "bike" bị tính là 2 token
    KHÁC NHAU, làm giảm precision n-gram một cách HỆ THỐNG trên mọi ảnh, mọi
    metric. Đây gần như chắc chắn là lý do BLEU-4/CIDEr thấp hơn nhiều so với
    literature (Bottom-Up Top-Down: BLEU-4 ~36, CIDEr ~113).

FIX #2 -- Đồng bộ tham số decoding với evaluate_flickr30k.py:
    Bản gốc gọi model.decoder.generate(fused_features, fused_mask,
    max_length=MAX_GEN_LENGTH) KHÔNG truyền repetition_penalty/
    no_repeat_ngram_size -- nghĩa là dùng giá trị MẶC ĐỊNH của hàm generate()
    (repetition_penalty=1.3, no_repeat_ngram_size=3), tức là bảng kết quả
    COCO chính thức (Mục 3.2 báo cáo) KHÔNG PHẢI greedy thuần túy, trong khi
    evaluate_flickr30k.py đã cẩn thận tắt hẳn 2 tham số này để đảm bảo "kiểm
    soát biến số". Bản vá này đồng bộ evaluate.py dùng đúng greedy thuần túy
    (giống Flickr30k) để 2 bảng kết quả có thể so sánh công bằng với nhau.

Lưu ý khác giữ nguyên từ bản gốc:
    - LẦN ĐẦU chạy, pycocoevalcap (phần SPICE) sẽ tự tải Stanford CoreNLP
      (~380MB) từ internet -- cần thời gian, chỉ tải 1 lần, lưu cache.
    - PTBTokenizer cũng cần Java -- bạn đã có sẵn Java cho METEOR/SPICE nên
      KHÔNG phát sinh thêm dependency mới.
    - Việc đánh giá dùng GROUND TRUTH là TOÀN BỘ 5 caption/ảnh (không phải
      random 1/5 như lúc train) -- đây là cách đánh giá CHUẨN của COCO.
"""

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # FIX #1: import tokenizer chuẩn
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
    Load checkpoint best của 1 strategy, sinh caption (greedy THUẦN TÚY --
    xem FIX #2) cho toàn bộ ảnh trong val2017. Trả về dict
    {coco_id_str: caption_string}.
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

        # FIX #2: truyền tường minh repetition_penalty=1.0, no_repeat_ngram_size=0
        # để khớp ĐÚNG điều kiện greedy thuần túy đã dùng trong evaluate_flickr30k.py
        # -- đảm bảo 2 bảng kết quả (COCO vs Flickr30k) so sánh công bằng.
        captions = model.decoder.generate(
            fused_features, fused_mask, max_length=MAX_GEN_LENGTH,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )

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
# BƯỚC 3 — Tính 4 chỉ số đánh giá (ĐÃ SỬA: thêm PTBTokenizer -- FIX #1)
# ============================================================
def compute_metrics(gts: dict, res: dict) -> dict:
    """
    gts: {coco_id: [list caption tham chiếu]}
    res: {coco_id: [1 caption sinh ra]}   -- pycocoevalcap yêu cầu res cũng
         là list (dù chỉ có 1 phần tử), không phải string trực tiếp.

    Trả về dict {metric_name: score}.

    FIX #1: Bản gốc đưa gts/res THẲNG vào scorer (text thô, còn dấu câu,
    hoa/thường lẫn lộn). Điều này khiến n-gram precision bị tính sai vì
    Bleu()/Cider()/Spice() chỉ tách từ theo khoảng trắng. Cần chạy qua
    PTBTokenizer trước -- ĐÚNG như coco-caption gốc luôn làm -- để chuẩn hóa
    dấu câu/hoa-thường trước khi so khớp n-gram.
    """
    print("  Đang chuẩn hóa caption qua PTBTokenizer (FIX quan trọng) ...")
    tokenizer = PTBTokenizer()
    # PTBTokenizer yêu cầu format {id: [{"caption": text}, ...]}
    gts_formatted = {k: [{"caption": c} for c in v] for k, v in gts.items()}
    res_formatted = {k: [{"caption": c} for c in v] for k, v in res.items()}
    gts_tokenized = tokenizer.tokenize(gts_formatted)
    res_tokenized = tokenizer.tokenize(res_formatted)

    metrics = {}

    print("  Đang tính BLEU ...")
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics["BLEU-1"] = bleu_scores[0]
    metrics["BLEU-2"] = bleu_scores[1]
    metrics["BLEU-3"] = bleu_scores[2]
    metrics["BLEU-4"] = bleu_scores[3]

    print("  Đang tính METEOR (cần Java, dùng bản đã patch lỗi buffering Windows) ...")
    meteor_scorer = MeteorFixed()
    meteor_score, _ = meteor_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics["METEOR"] = meteor_score

    print("  Đang tính CIDEr ...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics["CIDEr"] = cider_score

    print("  Đang tính SPICE (cần Java, lần đầu sẽ tải Stanford CoreNLP ~380MB) ...")
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts_tokenized, res_tokenized)
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

        print(f"\n[{strategy}] KẾT QUẢ (ĐÃ SỬA LỖI TOKENIZE + DECODING):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ----- Bảng tổng hợp so sánh cả 4 strategy -----
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("BẢNG SO SÁNH TỔNG HỢP (SAU KHI SỬA LỖI)")
        print("=" * 80)
        metric_names = list(next(iter(all_results.values())).keys())
        header = f"{'Strategy':<18}" + "".join(f"{m:>10}" for m in metric_names)
        print(header)
        for strategy, metrics in all_results.items():
            row = f"{strategy:<18}" + "".join(f"{metrics[m]:>10.4f}" for m in metric_names)
            print(row)

    # ----- Lưu kết quả tổng hợp ra JSON -----
    results_path = os.path.join(RESULTS_DIR, "evaluation_results_fixed.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tổng hợp tại: {results_path}")
    print("\n⚠️  LƯU Ý: file lưu là 'evaluation_results_fixed.json' (KHÔNG ghi đè")
    print("   'evaluation_results.json' cũ) để bạn có thể đối chiếu trước/sau khi sửa lỗi.")


if __name__ == "__main__":
    main()