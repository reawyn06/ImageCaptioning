"""
evaluate.py  (ĐÃ VÁ LỖI — xem khối "FIX #1", "FIX #2", "FIX #3" trong file)
======================
Mục đích:
    Đánh giá 4 checkpoint (1 cho mỗi fusion strategy) trên tập validation,
    bằng cách:
        1. Sinh caption (Beam Search, num_beams=4 -- xem FIX #3) cho toàn bộ
           ảnh trong val2017 (2,135 ảnh có đủ semantic feature).
        2. Tính 4 chỉ số: BLEU (1-4), CIDEr, METEOR, SPICE -- dùng
           pycocoevalcap (bộ chuẩn của MS-COCO, đúng cách các paper báo cáo).
        3. In bảng so sánh tổng hợp cho cả 4 strategy.

Cách chạy:
    python evaluate.py                     # đánh giá cả 4 strategy
    python evaluate.py --strategy concat   # chỉ đánh giá 1 strategy
    python evaluate.py --max-images 200     # test nhanh trên 200 ảnh trước khi
                                             # chạy full 2135 ảnh (KHUYẾN NGHỊ làm
                                             # trước để đo ETA, vì Beam Search +
                                             # không có KV-cache sẽ chậm hơn greedy
                                             # đáng kể -- xem FIX #3 để biết lý do)

===========================================================================
GHI CHÚ VỀ CÁC LỖI ĐÃ SỬA (SO VỚI BẢN GỐC)
===========================================================================

FIX #1 -- THIẾU BƯỚC PTBTokenizer (nguyên nhân chính khiến điểm thấp):
    Toàn bộ pipeline chuẩn của pycocoevalcap (repo gốc salesforce/coco-caption
    -- nơi MỌI paper báo cáo BLEU/CIDEr/SPICE đều dùng) YÊU CẦU chạy qua
    Stanford PTBTokenizer TRƯỚC khi đưa gts/res vào Bleu()/Cider()/Spice().
    Bản gốc của file này bỏ qua bước này, đưa thẳng text thô (còn nguyên dấu
    câu, hoa/thường không chuẩn hóa) vào scorer -- các scorer này chỉ tách
    từ bằng khoảng trắng (.split()), nên "bike." và "bike" bị tính là 2 token
    KHÁC NHAU, làm giảm precision n-gram một cách HỆ THỐNG trên mọi ảnh, mọi
    metric.

FIX #2 -- Đồng bộ tham số decoding với evaluate_flickr30k.py (bản GPT-2 cũ):
    Với decoder GPT-2 cũ, cần truyền tường minh repetition_penalty=1.0,
    no_repeat_ngram_size=0 để đảm bảo greedy thuần túy. Với decoder
    CaptionDecoderTransformer (from-scratch), 2 tham số này KHÔNG còn tác
    dụng gì (bị "nuốt" qua **_ignored_kwargs) -- xem transformer_caption_decoder.py.
    Giữ lại lời gọi này để tương thích ngược nếu bạn quay lại dùng decoder
    GPT-2 cũ cho strategy nào đó.

FIX #3 (MỚI) -- BẬT BEAM SEARCH thay vì greedy:
    CaptionDecoderTransformer.generate() đã hỗ trợ sẵn method="beam" với
    cross-attention THẬT vào fused_features (khác hẳn GPT-2 prefix injection),
    nhưng _generate_beam() bên trong CHỈ HỖ TRỢ batch_size=1 (xem assert
    trong file gốc). Vì DataLoader trả về batch (BATCH_SIZE ảnh/lần), không
    thể gọi generate(method="beam") thẳng trên cả batch.

    Giải pháp: vẫn tính fused_features/fused_mask theo BATCH như cũ (phần
    này không tốn thời gian đáng kể vì chỉ là 1 lần forward qua R-GCN +
    Fusion Module), nhưng LẶP QUA TỪNG ẢNH khi gọi generate() để mỗi lần
    gọi có batch_size=1 -- đúng điều kiện _generate_beam() yêu cầu.

    LƯU Ý VỀ THỜI GIAN: CaptionDecoderTransformer không dùng KV-cache (tính
    lại toàn bộ sequence mỗi bước sinh từ, xem comment trong generate()) --
    kết hợp với num_beams=4 (gấp 4 lần số forward pass so với greedy) và xử
    lý tuần tự từng ảnh (không còn batch song song ở bước generate), tổng
    thời gian đánh giá sẽ TĂNG ĐÁNG KỂ so với bản greedy trước đó. BẮT BUỘC
    chạy thử với --max-images 100~200 trước để đo ETA, tránh mất hàng giờ
    mới phát hiện thời gian không kịp deadline.

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
import time

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

# ----- Tham số Beam Search (FIX #3) -- có thể tune lại trên 1 tập con nhỏ
# TÁCH RIÊNG khỏi 2135 ảnh sẽ báo cáo chính thức, để tránh "overfit" tham số
# decode lên chính tập test (xem giải thích trong hội thoại). -----
NUM_BEAMS = 4
LENGTH_PENALTY = 0.8

ALL_STRATEGIES = ["baseline", "concat", "one_directional", "bidirectional"]


# ============================================================
# BƯỚC 1 — Sinh caption cho toàn bộ val set bằng 1 checkpoint (Beam Search)
# ============================================================
@torch.no_grad()
def generate_captions_for_strategy(strategy: str, device, max_images: int = None, image_offset: int = None) -> dict:
    """
    Load checkpoint best của 1 strategy, sinh caption bằng BEAM SEARCH
    (num_beams=NUM_BEAMS -- xem FIX #3) cho toàn bộ ảnh trong val2017, hoặc
    1 lát cắt [image_offset : image_offset+max_images] nếu max_images được
    truyền -- dùng để (a) test nhanh/đo ETA trước khi chạy full, hoặc (b) lấy
    1 tập DEV RIÊNG BIỆT (qua image_offset) để tune length_penalty/num_beams
    mà không "nhìn trộm" đúng những ảnh sẽ dùng để báo cáo kết quả chính thức.
    Trả về dict {coco_id_str: caption_string}.
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

    if max_images:
        # FIX (MỚI): thêm image_offset để lấy 1 LÁT CẮT RIÊNG BIỆT (khác tập
        # 100 ảnh đầu đã dùng để test tốc độ trước đó), tránh "nhìn trộm"
        # tập test khi tune length_penalty/num_beams -- nếu tune trực tiếp
        # trên đúng ảnh sẽ báo cáo, đó là học thuộc tập test, đúng loại lỗi
        # "kết luận chủ quan" giáo viên đã phê bình.
        start = image_offset or 0
        end = start + max_images
        val_dataset.image_ids = val_dataset.image_ids[start:end]
        print(f"  -> [TEST MODE] Dùng ảnh [{start}:{end}], còn {len(val_dataset)} ảnh.")

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    generated = {}
    total = len(val_dataset)
    done = 0
    t0 = time.time()

    for batch in val_loader:
        visual_features = batch["visual_features"].to(device)
        batch_objects = batch["batch_objects"]
        batch_triples = batch["batch_triples"]
        coco_ids = batch["coco_ids"]

        # ----- Phần tính fused_features vẫn chạy THEO BATCH như cũ (nhanh,
        # không đổi so với bản trước) -----
        semantic_features, semantic_mask = model.rgcn.forward_batch(batch_objects, batch_triples)
        fused_features, fused_mask = model.fusion(visual_features, semantic_features, semantic_mask)

        # ----- FIX #3: Beam Search -- _generate_beam() chỉ hỗ trợ batch_size=1,
        # nên LẶP QUA TỪNG ẢNH trong batch khi gọi generate(). -----
        captions = []
        for i in range(fused_features.size(0)):
            single_fused = fused_features[i : i + 1]
            single_mask = fused_mask[i : i + 1]
            cap = model.decoder.generate(
                single_fused, single_mask, max_length=MAX_GEN_LENGTH,
                method="beam", num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY,
            )
            captions.append(cap[0])

        for coco_id, caption in zip(coco_ids, captions):
            generated[str(coco_id)] = caption

        done += len(coco_ids)
        if done % (BATCH_SIZE * 5) == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  -> Đã sinh caption cho {done}/{total} ảnh | "
                  f"{speed:.2f} ảnh/s | ETA: {eta/60:.1f} phút")

    total_elapsed = time.time() - t0
    print(f"  -> [{strategy}] Hoàn tất sinh caption cho {total} ảnh trong "
          f"{total_elapsed/60:.1f} phút ({total_elapsed/max(total,1):.2f}s/ảnh).")

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
# BƯỚC 3 — Tính 4 chỉ số đánh giá (FIX #1: có PTBTokenizer)
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
    # FIX (SyntaxError: "name 'NUM_BEAMS' is used prior to global declaration"):
    # Python yêu cầu dòng `global X` phải nằm TRƯỚC mọi chỗ dùng X trong cùng
    # hàm -- kể cả khi X chỉ được ĐỌC (vd làm default= cho argparse), không
    # phải chỉ khi GÁN giá trị mới. Đưa khai báo global lên đầu hàm để sửa.
    global NUM_BEAMS, LENGTH_PENALTY

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None,
                        choices=ALL_STRATEGIES,
                        help="Chỉ đánh giá 1 strategy. Bỏ qua để đánh giá cả 4.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Giới hạn số ảnh -- dùng để test nhanh/đo ETA Beam Search "
                             "trước khi chạy full 2135 ảnh (vd --max-images 200). "
                             "KHÔNG dùng số liệu chạy giới hạn này để báo cáo chính thức.")
    parser.add_argument("--image-offset", type=int, default=None,
                        help="Bỏ qua N ảnh đầu trước khi lấy --max-images ảnh tiếp theo "
                             "(vd --image-offset 300 --max-images 200 -> dùng ảnh [300:500]). "
                             "Dùng để lấy 1 tập DEV RIÊNG BIỆT (khác 100 ảnh đầu đã test) "
                             "khi tune length_penalty/num_beams, tránh tune trên đúng ảnh "
                             "sẽ dùng để báo cáo chính thức (data leakage).")
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS,
                        help=f"Số beam cho Beam Search (mặc định {NUM_BEAMS}).")
    parser.add_argument("--length-penalty", type=float, default=LENGTH_PENALTY,
                        help=f"Length penalty cho Beam Search (mặc định {LENGTH_PENALTY}). "
                             "Thử các giá trị 0.6/0.8/1.0/1.2 trên 1 tập con nhỏ "
                             "TÁCH RIÊNG khỏi tập báo cáo chính thức để chọn giá trị tốt nhất.")
    args = parser.parse_args()

    # Cho phép override NUM_BEAMS/LENGTH_PENALTY qua CLI mà không cần sửa code
    NUM_BEAMS = args.num_beams
    LENGTH_PENALTY = args.length_penalty

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Beam Search: num_beams={NUM_BEAMS}, length_penalty={LENGTH_PENALTY}")

    strategies = [args.strategy] if args.strategy else ALL_STRATEGIES

    print("\nĐang load ground-truth caption (đầy đủ 5 caption/ảnh) ...")
    full_gts = load_full_ground_truth()

    all_results = {}

    for strategy in strategies:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Bỏ qua '{strategy}': không tìm thấy checkpoint tại {checkpoint_path}")
            continue

        # ----- Sinh caption (Beam Search) -----
        generated = generate_captions_for_strategy(
            strategy, device, max_images=args.max_images, image_offset=args.image_offset
        )

        # Lưu caption sinh ra ra file riêng (để xem qua, debug, hoặc đưa vào báo cáo)
        tag = "_testrun" if args.max_images else ""
        gen_path = os.path.join(RESULTS_DIR, f"{strategy}_generated_captions_beam{tag}.json")
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

        print(f"\n[{strategy}] KẾT QUẢ (Beam Search num_beams={NUM_BEAMS}, "
              f"length_penalty={LENGTH_PENALTY}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ----- Bảng tổng hợp so sánh cả 4 strategy -----
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print(f"BẢNG SO SÁNH TỔNG HỢP (Beam Search num_beams={NUM_BEAMS}, "
              f"length_penalty={LENGTH_PENALTY})")
        print("=" * 80)
        metric_names = list(next(iter(all_results.values())).keys())
        header = f"{'Strategy':<18}" + "".join(f"{m:>10}" for m in metric_names)
        print(header)
        for strategy, metrics in all_results.items():
            row = f"{strategy:<18}" + "".join(f"{metrics[m]:>10.4f}" for m in metric_names)
            print(row)

    # ----- Lưu kết quả tổng hợp ra JSON -----
    tag = "_testrun" if args.max_images else ""
    results_path = os.path.join(RESULTS_DIR, f"evaluation_results_beam{tag}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả tổng hợp tại: {results_path}")
    if args.max_images:
        print("\n⚠️  ĐÂY LÀ KẾT QUẢ TEST TRÊN TẬP CON NHỎ (--max-images) -- KHÔNG dùng")
        print("   số liệu này để báo cáo chính thức. Chạy lại KHÔNG có --max-images")
        print("   để đánh giá trên toàn bộ 2135 ảnh val2017.")


if __name__ == "__main__":
    main()