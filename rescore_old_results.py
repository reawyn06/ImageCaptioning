"""
rescore_old_results.py
======================
Mục đích:
    Chấm lại TOÀN BỘ caption ĐÃ SINH TỪ TRƯỚC (model GPT-2 cũ, khi chưa đổi
    sang Transformer decoder) bằng compute_metrics() ĐÃ SỬA LỖI (thêm
    PTBTokenizer) -- để CÔ LẬP RÕ RÀNG 2 hiệu ứng đang trộn lẫn với nhau khi
    so sánh kết quả cũ/mới:

        (a) Hiệu ứng SỬA LỖI TÍNH TOÁN (PTBTokenizer) -- không liên quan gì
            đến chất lượng model, chỉ là sửa CÁCH ĐO. Script này đo ĐÚNG
            hiệu ứng (a), vì dùng lại 100% caption GPT-2 cũ (không sinh lại).
        (b) Hiệu ứng ĐỔI KIẾN TRÚC DECODER (GPT-2 -> Transformer tự huấn
            luyện) -- CHỈ biết được SAU KHI train xong 4 checkpoint mới và
            chạy evaluate.py/evaluate_flickr30k.py trên checkpoint đó.

    Sau khi có cả 2 bảng số (bảng này cho (a), evaluate.py sau khi train
    xong cho (a)+(b) gộp lại), bạn trừ ra được đóng góp riêng của (b) --
    đúng tinh thần "kiểm soát biến số" mà báo cáo NCKH cần.

Yêu cầu:
    - Các file results/{strategy}_generated_captions.json (COCO, GPT-2 cũ)
    - Các file results/flickr30k_{strategy}_generated.json (Flickr30k DETR v1)
    - Các file results/flickr30k_{strategy}_yoloworld_generated.json (Flickr30k YOLO-World v2)
    - Các file evaluation_results*.json TƯƠNG ỨNG (số cũ để đối chiếu)
    vẫn còn nguyên trong results/ (đúng như ảnh bạn gửi -- có đủ).

KHÔNG CẦN GPU, KHÔNG CẦN load checkpoint/model nào -- chỉ đọc lại JSON caption
đã có sẵn và tính lại metric, nên chạy trong vài phút (phần lâu nhất là SPICE
với Flickr30k full ~31,783 ảnh, giống thời gian SPICE đã từng tốn lúc chạy
evaluate_flickr30k.py gốc).

Cách chạy (BẮT BUỘC chạy từ project root, vì import evaluate.py/
evaluate_flickr30k.py đã sửa PTBTokenizer để tái sử dụng đúng logic tính
metric, tránh viết trùng code):
    python rescore_old_results.py
"""

import json
import os

# Tái sử dụng ĐÚNG hàm compute_metrics() (đã có PTBTokenizer) và các hàm load
# ground-truth từ 2 file evaluate.py / evaluate_flickr30k.py ĐÃ SỬA -- tránh
# viết trùng logic, đảm bảo dùng chính xác cùng 1 công thức tính điểm.
from evaluate import (
    compute_metrics as compute_metrics_coco,
    load_full_ground_truth,
    RESULTS_DIR as COCO_RESULTS_DIR,
)
from evaluate_flickr30k import (
    compute_metrics as compute_metrics_f30k,
    load_flickr30k_captions,
    CAPTIONS_FILE,
    RESULTS_DIR as F30K_RESULTS_DIR,
)

ALL_STRATEGIES = ["baseline", "concat", "one_directional", "bidirectional"]


def _print_comparison(title: str, old_results: dict, new_results: dict):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not new_results:
        print("  (Không có strategy nào chấm được -- kiểm tra lại đường dẫn file.)")
        return

    metric_names = list(next(iter(new_results.values())).keys())
    header = f"{'Strategy':<18}" + "".join(f"{m:>20}" for m in metric_names)
    print(header)
    for strategy, new_m in new_results.items():
        old_m = old_results.get(strategy, {})
        row_parts = []
        for m in metric_names:
            old_v = old_m.get(m)
            new_v = new_m[m]
            cell = f"{old_v:.4f}->{new_v:.4f}" if old_v is not None else f"?->{new_v:.4f}"
            row_parts.append(cell)
        row = f"{strategy:<18}" + "".join(f"{p:>20}" for p in row_parts)
        print(row)
    print("\n  (Định dạng mỗi ô: điểm CŨ (chưa sửa PTBTokenizer) -> điểm MỚI (đã sửa),")
    print("   CÙNG 1 bộ caption GPT-2 -- chênh lệch 100% do lỗi tính toán, KHÔNG do model.)")


# ============================================================
# COCO val2017
# ============================================================
def rescore_coco():
    print("\n### COCO val2017 (caption GPT-2 cũ) ###")

    old_results_path = os.path.join(COCO_RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(old_results_path):
        print(f"  Không tìm thấy {old_results_path}, bỏ qua COCO.")
        return
    with open(old_results_path, "r", encoding="utf-8") as f:
        old_results = json.load(f)

    full_gts = load_full_ground_truth()

    new_results = {}
    for strategy in ALL_STRATEGIES:
        gen_path = os.path.join(COCO_RESULTS_DIR, f"{strategy}_generated_captions.json")
        if not os.path.exists(gen_path):
            print(f"  Bỏ qua '{strategy}': không tìm thấy {gen_path}")
            continue
        with open(gen_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        common_ids = [cid for cid in generated.keys() if cid in full_gts]
        gts = {cid: full_gts[cid] for cid in common_ids}
        res = {cid: [generated[cid]] for cid in common_ids}

        print(f"\n  [{strategy}] Đang chấm lại {len(common_ids)} caption (đã có sẵn, KHÔNG sinh lại) ...")
        new_results[strategy] = compute_metrics_coco(gts, res)

    _print_comparison(
        "COCO val2017 — SO SÁNH TRƯỚC/SAU FIX PTBTokenizer (CÙNG caption GPT-2 cũ)",
        old_results, new_results,
    )

    out_path = os.path.join(COCO_RESULTS_DIR, "evaluation_results_rescored_old_gpt2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Đã lưu: {out_path}")


# ============================================================
# Flickr30k (dùng chung cho cả DETR v1 và YOLO-World v2)
# ============================================================
def rescore_flickr30k(label: str, old_results_filename: str, gen_filename_pattern):
    print(f"\n### Flickr30k — {label} (caption GPT-2 cũ) ###")

    old_results_path = os.path.join(F30K_RESULTS_DIR, old_results_filename)
    if not os.path.exists(old_results_path):
        print(f"  Không tìm thấy {old_results_path}, bỏ qua.")
        return
    with open(old_results_path, "r", encoding="utf-8") as f:
        old_results = json.load(f)

    captions_gts = load_flickr30k_captions(CAPTIONS_FILE)

    new_results = {}
    for strategy in ALL_STRATEGIES:
        gen_path = os.path.join(F30K_RESULTS_DIR, gen_filename_pattern(strategy))
        if not os.path.exists(gen_path):
            print(f"  Bỏ qua '{strategy}': không tìm thấy {gen_path}")
            continue
        with open(gen_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        common_ids = [img_id for img_id in generated if img_id in captions_gts]
        gts = {img_id: captions_gts[img_id] for img_id in common_ids}
        res = {img_id: [generated[img_id]] for img_id in common_ids}

        print(f"\n  [{strategy}] Đang chấm lại {len(common_ids)} caption (đã có sẵn, KHÔNG sinh lại) ...")
        new_results[strategy] = compute_metrics_f30k(gts, res)

    _print_comparison(
        f"Flickr30k {label} — SO SÁNH TRƯỚC/SAU FIX PTBTokenizer (CÙNG caption GPT-2 cũ)",
        old_results, new_results,
    )

    tag_suffix = "_detr" if label.startswith("DETR") else "_yoloworld"
    out_path = os.path.join(F30K_RESULTS_DIR, f"flickr30k_evaluation_results{tag_suffix}_rescored_old_gpt2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Đã lưu: {out_path}")


def main():
    rescore_coco()

    # Flickr30k DETR (v1) -- file KHÔNG có hậu tố tag (script gốc lúc đó
    # chưa có tham số --tag), tên file: flickr30k_{strategy}_generated.json
    rescore_flickr30k(
        label="DETR (v1)",
        old_results_filename="flickr30k_evaluation_results.json",
        gen_filename_pattern=lambda s: f"flickr30k_{s}_generated.json",
    )

    # Flickr30k YOLO-World (v2) -- file CÓ hậu tố "_yoloworld_"
    rescore_flickr30k(
        label="YOLO-World (v2)",
        old_results_filename="flickr30k_evaluation_results_yoloworld.json",
        gen_filename_pattern=lambda s: f"flickr30k_{s}_yoloworld_generated.json",
    )

    print("\n" + "=" * 100)
    print("HOÀN TẤT — 3 file *_rescored_old_gpt2.json đã lưu trong results/,")
    print("dùng để đối chiếu SAU KHI train xong Transformer decoder mới:")
    print("  Tổng chênh lệch (evaluate.py mới nhất - evaluation_results.json cũ)")
    print("  = [chênh lệch do fix PTBTokenizer (đo được NGAY BÂY GIỜ, ở đây)]")
    print("  + [chênh lệch do đổi kiến trúc decoder (đo được SAU KHI train xong)]")
    print("=" * 100)


if __name__ == "__main__":
    main()