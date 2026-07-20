"""
evaluate_flickr30k.py  (ĐÃ VÁ LỖI — xem khối "FIX" trong file)
======================
Mục đích:
    Đánh giá khả năng tổng quát hóa (generalization) của 4 checkpoint
    (baseline, concat, one_directional, bidirectional) đã train trên COCO+VG,
    bằng cách sinh caption và tính BLEU/METEOR/CIDEr/SPICE trên Flickr30k.

Điểm khác biệt quan trọng so với evaluate.py (COCO):
    - Ground truth caption: đọc từ captions.txt (CSV), không phải JSON COCO.
    - Visual feature: đọc từ features/flickr30k/visual/{image_id}.pt
    - Semantic feature: đọc từ thư mục được chỉ định qua --semantic-dir
    - Chỉ đánh giá trên ảnh có ĐỦ cả 2 feature (visual + semantic) —
      ảnh không có semantic feature vẫn được đưa vào baseline nhưng cũng đưa vào
      3 strategy kia với graph rỗng (RGCNEncoder xử lý đúng trường hợp 0 node).

Cách chạy:
    python evaluate_flickr30k.py                     # Đánh giá cả 4 strategy với DETR mặc định
    python evaluate_flickr30k.py --semantic-dir features/flickr30k/semantic_yoloworld --tag yoloworld
    python evaluate_flickr30k.py --strategy baseline  # Chỉ test 1 strategy

===========================================================================
GHI CHÚ VỀ LỖI ĐÃ SỬA (so với bản gốc)
===========================================================================
FIX -- THIẾU BƯỚC PTBTokenizer (cùng lỗi đã tìm thấy trong evaluate.py):
    Bản gốc đưa gts/res dạng text thô (chưa qua Stanford PTBTokenizer) thẳng
    vào Bleu()/Cider()/Spice(). Điều này làm giảm precision n-gram một cách
    hệ thống (dấu câu dính vào từ cuối không khớp reference đã tokenize).
    Đây là 1 trong 2 lý do khiến điểm BLEU/CIDEr thấp hơn kỳ vọng literature.
    Bản vá này thêm bước tokenize CHUẨN giống coco-caption gốc.

    LƯU Ý: decoding params (repetition_penalty=1.0, no_repeat_ngram_size=0)
    trong hàm generate_captions() của bản gốc ĐÃ ĐÚNG SẴN (bạn đã tự làm tốt
    phần này để đảm bảo "kiểm soát biến số" khi so DETR vs YOLO-World) --
    KHÔNG cần sửa gì thêm ở phần đó, chỉ evaluate.py (COCO) là thiếu đồng bộ.
"""

import os
import sys
import csv
import json
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

# Thêm project root vào sys.path để import module gốc
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
sys.path.insert(0, PROJECT_ROOT)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # FIX: import tokenizer chuẩn
from meteor_fixed import MeteorFixed

from rgcn_encoder import GloveVocab
from train import ImageCaptioningModel, GLOVE_VOCAB_PATH

# ============================================================
# CONFIG
# ============================================================
FLICKR30K_DIR = os.path.join(PROJECT_ROOT, "datasets", "flickr30k")
IMAGES_DIR = os.path.join(FLICKR30K_DIR, "Images")
CAPTIONS_FILE = os.path.join(FLICKR30K_DIR, "captions.txt")

FEATURE_ROOT = os.path.join(PROJECT_ROOT, "features", "flickr30k")
VISUAL_DIR = os.path.join(FEATURE_ROOT, "visual")
SEMANTIC_DIR = os.path.join(FEATURE_ROOT, "semantic")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

ALL_STRATEGIES = ["baseline", "concat", "one_directional", "bidirectional"]
BATCH_SIZE = 8
MAX_GEN_LENGTH = 30


# ============================================================
# BƯỚC 1 — Đọc ground-truth caption từ captions.txt
# ============================================================
def load_flickr30k_captions(captions_file: str) -> dict:
    """
    Đọc captions.txt, trả về dict {image_id: [list 5 captions]}.
    Dùng csv.reader để xử lý đúng caption có dấu phẩy bên trong.
    """
    gts = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # bỏ qua header
        for row in reader:
            if len(row) < 2:
                continue
            img_id = row[0].strip()
            caption = row[1].strip()
            if img_id:
                gts.setdefault(img_id, []).append(caption)
    return gts


# ============================================================
# BƯỚC 2 — Dataset cho Flickr30k
# ============================================================
class Flickr30kDataset(Dataset):
    """
    Dataset tương tự CaptionDataset (COCO) nhưng dành cho Flickr30k.
    Chỉ trả về ảnh có ĐỦ cả visual feature lẫn bộ hành semantic feature được chỉ định.
    """

    def __init__(self, visual_dir: str, semantic_dir: str,
                 captions_file: str, max_images: int = None):
        self.visual_dir = visual_dir
        self.semantic_dir = semantic_dir

        # Lấy danh sách ảnh có visual feature
        visual_ids = {f.replace(".pt", "") + ".jpg"
                      for f in os.listdir(visual_dir) if f.endswith(".pt")}

        # Lấy danh sách ảnh có semantic feature (kể cả rỗng) từ thư mục động
        semantic_ids = {f.replace(".json", "") + ".jpg"
                        for f in os.listdir(semantic_dir) if f.endswith(".json")}

        # Lấy danh sách ảnh có caption
        self.captions = load_flickr30k_captions(captions_file)
        caption_ids = set(self.captions.keys())

        # Giao: chỉ giữ ảnh có đủ cả 3
        valid_ids = visual_ids & semantic_ids & caption_ids
        self.image_ids = sorted(valid_ids)

        if max_images:
            self.image_ids = self.image_ids[:max_images]

        print(f"[Flickr30kDataset] Thư mục semantic hiện tại: {semantic_dir}")
        print(f"[Flickr30kDataset] Visual: {len(visual_ids)}, "
              f"Semantic: {len(semantic_ids)}, Caption: {len(caption_ids)}")
        print(f"[Flickr30kDataset] Dùng được: {len(self.image_ids)} ảnh")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        stem = img_id.replace(".jpg", "")

        # Visual feature
        visual_feat = torch.load(
            os.path.join(self.visual_dir, f"{stem}.pt"), weights_only=False
        )
        if isinstance(visual_feat, dict):
            visual_feat = visual_feat.get("features", visual_feat.get("embeddings"))

        # Semantic feature (đọc từ thư mục chỉ định)
        with open(os.path.join(self.semantic_dir, f"{stem}.json"), "r") as f:
            record = json.load(f)
        objects = record.get("objects", [])
        triples = [tuple(t) for t in record.get("triples", [])]

        # Caption: lấy tất cả phục vụ evaluation
        captions = self.captions.get(img_id, [])

        return {
            "image_id": img_id,
            "visual_feature": visual_feat,
            "objects": objects,
            "triples": triples,
            "captions": captions,
        }


def collate_fn_flickr(batch):
    image_ids = [item["image_id"] for item in batch]
    visual_features = torch.stack([item["visual_feature"] for item in batch])
    batch_objects = [item["objects"] for item in batch]
    batch_triples = [item["triples"] for item in batch]
    captions = [item["captions"] for item in batch]
    return {
        "image_ids": image_ids,
        "visual_features": visual_features,
        "batch_objects": batch_objects,
        "batch_triples": batch_triples,
        "captions": captions,
    }


# ============================================================
# BƯỚC 3 — Sinh caption cho 1 strategy
# ============================================================
@torch.no_grad()
def generate_captions(strategy: str, dataset: Flickr30kDataset,
                      device: torch.device) -> dict:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
    print(f"\n[{strategy}] Đang load checkpoint: {checkpoint_path} ...")

    glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)
    model = ImageCaptioningModel(strategy, glove_vocab).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  -> Checkpoint epoch={checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f} (COCO)")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn_flickr, num_workers=0)

    generated = {}
    total = len(dataset)
    done = 0

    for batch in loader:
        visual_features = batch["visual_features"].to(device)
        batch_objects = batch["batch_objects"]
        batch_triples = batch["batch_triples"]
        image_ids = batch["image_ids"]

        semantic_features, semantic_mask = model.rgcn.forward_batch(
            batch_objects, batch_triples
        )
        fused_features, fused_mask = model.fusion(
            visual_features, semantic_features, semantic_mask
        )
        # Giữ NGUYÊN như bản gốc -- phần này vốn ĐÃ ĐÚNG (greedy thuần túy,
        # đã tắt tường minh repetition_penalty/no_repeat_ngram_size từ đầu).
        captions = model.decoder.generate(
            fused_features, fused_mask, max_length=MAX_GEN_LENGTH,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )

        for img_id, caption in zip(image_ids, captions):
            generated[img_id] = caption

        done += len(image_ids)
        if done % (BATCH_SIZE * 50) == 0 or done == total:
            print(f"  -> Đã sinh caption cho {done}/{total} ảnh")

    return generated


# ============================================================
# BƯỚC 4 — Tính metrics (ĐÃ SỬA: thêm PTBTokenizer)
# ============================================================
def compute_metrics(gts: dict, res: dict) -> dict:
    """
    FIX: thêm bước PTBTokenizer trước khi tính BLEU/METEOR/CIDEr/SPICE --
    cùng lý do đã giải thích trong evaluate.py (chuẩn hóa dấu câu/hoa-thường
    để n-gram precision được tính đúng, khớp cách coco-caption gốc làm).
    """
    print("  Đang chuẩn hóa caption qua PTBTokenizer (FIX quan trọng) ...")
    tokenizer = PTBTokenizer()
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

    print("  Đang tính METEOR ...")
    meteor_scorer = MeteorFixed()
    meteor_score, _ = meteor_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics["METEOR"] = meteor_score

    print("  Đang tính CIDEr ...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics["CIDEr"] = cider_score

    print("  Đang tính SPICE ...")
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
    parser.add_argument("--max-images", type=int, default=None,
                        help="Giới hạn số ảnh (để test nhanh, vd --max-images 500)")
    parser.add_argument("--semantic-dir", type=str, default=SEMANTIC_DIR,
                        help="Thư mục semantic feature -- đổi thành "
                             "features/flickr30k/semantic_yoloworld để dùng pipeline mới")
    parser.add_argument("--tag", type=str, default="detr",
                        help="Nhãn để phân biệt file kết quả xuất ra, vd 'yoloworld'")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for path, name in [(VISUAL_DIR, "Visual features"), (args.semantic_dir, "Semantic features"),
                       (CAPTIONS_FILE, "captions.txt")]:
        if not os.path.exists(path):
            print(f"LỖI: Không tìm thấy {name} tại {path}")
            print(f"Hãy kiểm tra lại đường dẫn pipeline tương ứng.")
            sys.exit(1)

    print("\nĐang load dataset Flickr30k ...")
    dataset = Flickr30kDataset(VISUAL_DIR, args.semantic_dir, CAPTIONS_FILE,
                               max_images=args.max_images)

    if len(dataset) == 0:
        print("LỖI: Không có ảnh nào hợp lệ. Kiểm tra lại features đã build.")
        sys.exit(1)

    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES
    all_results = {}

    for strategy in strategies:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Bỏ qua '{strategy}': không tìm thấy checkpoint.")
            continue

        # Sinh caption
        generated = generate_captions(strategy, dataset, device)

        # Lưu caption sinh ra với tag động để tránh ghi đè
        gen_path = os.path.join(RESULTS_DIR, f"flickr30k_{strategy}_{args.tag}_generated.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, ensure_ascii=False, indent=2)
        print(f"  -> Đã lưu caption tại: {gen_path}")

        # Chuẩn bị gts/res đúng format pycocoevalcap
        common_ids = [img_id for img_id in generated if img_id in dataset.captions]
        gts = {img_id: dataset.captions[img_id] for img_id in common_ids}
        res = {img_id: [generated[img_id]] for img_id in common_ids}

        print(f"\n[{strategy}] Đang tính 4 chỉ số trên {len(common_ids)} ảnh (Tag: {args.tag}) ...")
        metrics = compute_metrics(gts, res)
        all_results[strategy] = metrics

        print(f"\n[{strategy}] KẾT QUẢ (Flickr30k - Tag: {args.tag}, ĐÃ SỬA LỖI TOKENIZE):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Bảng so sánh tổng hợp
    if len(all_results) > 1:
        print("\n" + "=" * 90)
        print(f"BẢNG SO SÁNH TỔNG HỢP — Flickr30k (Tag: {args.tag})")
        print("=" * 90)
        metric_names = list(next(iter(all_results.values())).keys())
        header = f"{'Strategy':<20}" + "".join(f"{m:>10}" for m in metric_names)
        print(header)
        for strategy, metrics in all_results.items():
            row = f"{strategy:<20}" + "".join(f"{metrics[m]:>10.4f}" for m in metric_names)
            print(row)

        # So sánh với kết quả COCO gốc nếu có (ƯU TIÊN file đã sửa lỗi nếu tồn tại)
        coco_results_path_fixed = os.path.join(RESULTS_DIR, "evaluation_results_fixed.json")
        coco_results_path_old = os.path.join(RESULTS_DIR, "evaluation_results.json")
        coco_results_path = coco_results_path_fixed if os.path.exists(coco_results_path_fixed) else coco_results_path_old

        if os.path.exists(coco_results_path):
            with open(coco_results_path) as f:
                coco_results = json.load(f)
            print("\n" + "=" * 90)
            print(f"SO SÁNH COCO vs FLICKR30K ({args.tag.upper()}) (BLEU-4 và CIDEr)")
            print(f"(Đối chiếu với: {coco_results_path})")
            print("=" * 90)
            print(f"{'Strategy':<20} {'BLEU-4 COCO':>12} {'BLEU-4 F30k':>12} "
                  f"{'CIDEr COCO':>11} {'CIDEr F30k':>11}")
            for strategy in all_results:
                if strategy not in coco_results:
                    continue
                b4_coco = coco_results[strategy].get("BLEU-4", 0)
                b4_f30k = all_results[strategy].get("BLEU-4", 0)
                ci_coco = coco_results[strategy].get("CIDEr", 0)
                ci_f30k = all_results[strategy].get("CIDEr", 0)
                delta_b4 = b4_f30k - b4_coco
                delta_ci = ci_f30k - ci_coco
                print(f"{strategy:<20} {b4_coco:>12.4f} {b4_f30k:>12.4f} "
                      f"{ci_coco:>11.4f} {ci_f30k:>11.4f} "
                      f"  (Δ BLEU-4: {delta_b4:+.4f}, Δ CIDEr: {delta_ci:+.4f})")

    # Lưu kết quả tổng hợp với định dạng tên mới
    out_path = os.path.join(RESULTS_DIR, f"flickr30k_evaluation_results_{args.tag}_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tổng hợp tại: {out_path}")


if __name__ == "__main__":
    main()