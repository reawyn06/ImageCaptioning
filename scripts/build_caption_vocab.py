"""
build_caption_vocab.py
======================
Mục đích:
    Xây vocab CHO CAPTION DECODER MỚI (Transformer tự huấn luyện, KHÔNG dùng
    BPE tokenizer của GPT-2 nữa). Đây là bước bắt buộc phải làm trước khi
    dùng transformer_caption_decoder.py, vì decoder mới cần 1 bảng embedding
    học từ đầu (không có pretrained weight nào để "thừa hưởng" như GPT-2).

Phương pháp luận (theo đúng chuẩn các paper captioning kinh điển):
    - Tokenize: lowercase + loại bỏ ký tự không phải chữ/số/apostrophe, tách
      theo khoảng trắng -- CÙNG QUY ƯỚC với Karpathy split / Vinyals et al.
      "Show and Tell" (2015): giữ nguyên câu caption gốc, không stem/lemma.
    - Lọc theo tần suất: CHỈ giữ từ xuất hiện >= MIN_FREQ=5 lần trong toàn bộ
      caption tập TRAIN (KHÔNG dùng val -- tránh rò rỉ thông tin/leakage).
      Đây là ngưỡng chuẩn dùng trong hầu hết paper captioning kinh điển
      (Vinyals 2015, Xu 2015, Anderson 2018 đều dùng min_freq tương tự,
      cho ra vocab khoảng 9,000-10,000 từ trên full COCO -- với 48,365 ảnh
      train của bạn (ít hơn full COCO ~110K), vocab dự kiến sẽ nhỏ hơn).
    - 4 token đặc biệt: <pad>, <bos>, <eos>, <unk> -- đặt cố định ở đầu vocab
      (index 0-3) để dễ tra cứu (PAD_ID=0, BOS_ID=1, EOS_ID=2, UNK_ID=3).

Lưu ý quan trọng:
    Vocab CHỈ xây từ train2017 (đúng thực hành ML: không được "nhìn thấy"
    từ vựng của tập val trước khi đánh giá) -- từ nào trong val không có
    trong vocab sẽ tự động map thành <unk> khi encode_captions().

Cách chạy:
    python build_caption_vocab.py
"""

import json
import os
import re
from collections import Counter

import torch

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
CAPTIONS_TRAIN_PATH = os.path.join(PROJECT_ROOT, "../datasets", "coco", "annotations", "captions_train2017.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "../features", "caption_vocab.pt")

MIN_FREQ = 5  # ngưỡng chuẩn của literature (Vinyals 2015, Anderson 2018, ...)

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ============================================================
# Hàm tokenize -- PHẢI dùng NHẤT QUÁN với transformer_caption_decoder.py
# (hàm _tokenize_text ở đó phải sinh ra kết quả GIỐNG HỆT hàm này, nếu
# không sẽ có mismatch từ vựng lúc train vs lúc build vocab)
# ============================================================
def tokenize(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return [w for w in text.split() if w]


def main():
    print(f"Đang đọc {CAPTIONS_TRAIN_PATH} ...")
    with open(CAPTIONS_TRAIN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Tổng số caption train2017: {len(data['annotations'])}")

    counter = Counter()
    max_len_seen = 0
    for ann in data["annotations"]:
        words = tokenize(ann["caption"])
        counter.update(words)
        max_len_seen = max(max_len_seen, len(words))

    print(f"Tổng số từ duy nhất (trước lọc tần suất): {len(counter)}")
    print(f"Độ dài caption dài nhất (số từ, sau tokenize): {max_len_seen}")

    vocab_words = sorted([w for w, c in counter.items() if c >= MIN_FREQ])
    idx2word = SPECIAL_TOKENS + vocab_words
    word2idx = {w: i for i, w in enumerate(idx2word)}

    n_covered = sum(c for w, c in counter.items() if c >= MIN_FREQ)
    n_total = sum(counter.values())
    coverage = n_covered / n_total * 100

    result = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "min_freq": MIN_FREQ,
        "special_tokens": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID},
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(result, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("HOÀN TẤT XÂY VOCAB CHO TRANSFORMER DECODER")
    print("=" * 60)
    print(f"Vocab cuối cùng: {len(idx2word)} token (gồm 4 token đặc biệt + "
          f"{len(vocab_words)} từ có tần suất >= {MIN_FREQ})")
    print(f"Độ phủ (coverage) trên tổng số lượt xuất hiện từ: {coverage:.2f}% "
          f"(phần còn lại sẽ map thành <unk> khi encode)")
    print(f"Đã lưu tại: {OUTPUT_PATH}")
    print("\nVí dụ 20 từ đầu tiên trong vocab (sau token đặc biệt):")
    print(f"  {vocab_words[:20]}")


if __name__ == "__main__":
    main()