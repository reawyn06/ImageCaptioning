"""
build_glove_vocab.py
======================
Mục đích:
    Đọc toàn bộ scene graph đã xây (features/semantic/{train2017,val2017}/*.json),
    thu thập vocab thực tế (object names + predicates) đang được dùng,
    rồi tra cứu embedding GloVe 840B.300d cho từng từ/cụm từ trong vocab đó.

    Kết quả lưu thành 1 file nhỏ (glove_vocab.pt) chứa:
        - object_vocab: list object name (đã sort, để cố định thứ tự index)
        - predicate_vocab: list predicate (đã sort)
        - object_embeddings: Tensor (num_objects, 300)
        - predicate_embeddings: Tensor (num_predicates, 300)
        - oov_objects / oov_predicates: list các từ KHÔNG tìm được embedding
          (kể cả sau khi fallback average) -> các từ này sẽ được khởi tạo random
          khi train (cần biết trước để xử lý)

    Lý do làm bước này riêng (không tra GloVe trực tiếp lúc train):
        - File glove.840B.300d.txt nặng ~5.65GB, đọc toàn bộ vào RAM mỗi lần
          train là lãng phí và chậm (nhất là máy 16GB RAM).
        - Vocab thực tế của bài toán chỉ có vài trăm đến vài nghìn từ
          (object/predicate trong VG), nên chỉ cần trích đúng phần cần dùng,
          lưu lại 1 lần, dùng mãi.

Cách xử lý cụm từ nhiều token (theo quyết định đã chốt):
    1. Thử tra NGUYÊN cụm từ trước (vd "traffic light" -> tìm đúng key "traffic light"
       nếu GloVe có sẵn token dạng cụm — thực tế GloVe 840B đôi khi có một số cụm
       được nối bằng dấu cách như từ đơn do lỗi tokenize gốc, nhưng phổ biến hơn là
       KHÔNG có, nên bước này thường sẽ rơi xuống bước 2).
    2. Nếu không có, fallback: tách từng từ trong cụm, tra embedding từng từ,
       lấy trung bình (mean) các vector tìm được.
    3. Nếu không từ nào trong cụm tìm được embedding -> đánh dấu OOV (out-of-vocab),
       không có vector, sẽ phải random init khi train.

Lưu ý quan trọng về hiệu năng đọc file:
    - KHÔNG load toàn bộ 5.65GB vào RAM rồi mới lọc.
    - Đọc file GloVe theo từng dòng (streaming), chỉ giữ lại dòng có từ đầu tiên
      khớp với 1 trong các từ đơn lẻ cần dùng (đã tách từ vocab cụm từ).
      Cách này giảm RAM cần dùng xuống rất nhiều so với load hết 2.2M từ.
"""

import json
import os
import re
import torch

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"

SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "features", "semantic")
TRAIN_DIR = os.path.join(SEMANTIC_DIR, "train2017")
VAL_DIR = os.path.join(SEMANTIC_DIR, "val2017")

GLOVE_PATH = os.path.join(PROJECT_ROOT, "datasets", "glove", "glove.840B.300d.txt")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")

EMBED_DIM = 300


# ============================================================
# BƯỚC 1 — Thu thập vocab thực tế từ scene graph đã xây
# ============================================================
def collect_vocab():
    """
    Quét toàn bộ file scene graph (train + val), thu thập:
        - object_vocab: set tất cả object name xuất hiện
        - predicate_vocab: set tất cả predicate xuất hiện
    """
    print("Đang quét scene graph để thu thập vocab thực tế ...")

    object_vocab = set()
    predicate_vocab = set()

    for split_dir in [TRAIN_DIR, VAL_DIR]:
        files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
        for fname in files:
            with open(os.path.join(split_dir, fname), "r", encoding="utf-8") as f:
                record = json.load(f)
            object_vocab.update(record.get("objects", []))
            for s, p, o in record.get("triples", []):
                predicate_vocab.add(p)
                # objects trong triples đã được include trong "objects" field rồi
                # (build_scene_graphs.py đã đảm bảo điều này), không cần add lại

    object_vocab = sorted(object_vocab)
    predicate_vocab = sorted(predicate_vocab)

    print(f"  -> Vocab object: {len(object_vocab)} từ/cụm từ")
    print(f"  -> Vocab predicate: {len(predicate_vocab)} từ/cụm từ")
    return object_vocab, predicate_vocab


# ============================================================
# BƯỚC 2 — Xác định danh sách "từ đơn cần tra" từ vocab cụm từ
# ============================================================
def get_lookup_keys(vocab):
    """
    Với mỗi entry trong vocab (có thể là cụm nhiều từ), trả về set tất cả
    các "key" cần tra trong GloVe:
        - chính cụm từ đó (để thử tra nguyên cụm trước)
        - từng từ đơn trong cụm (để fallback average)
        - nếu 1 từ có dạng sở hữu cách ("bear's"), thêm cả phần gốc ("bear")
          -> vì GloVe (Common Crawl tokenizer) thường tách rời 's khỏi từ gốc,
          nên token "bear's" nguyên vẹn thường KHÔNG có trong GloVe, nhưng "bear" thì có.
    """
    lookup_keys = set()
    for entry in vocab:
        lookup_keys.add(entry)  # thử nguyên cụm
        for word in entry.split(" "):
            lookup_keys.add(word)  # từng từ đơn, để fallback
            if word.endswith("'s"):
                lookup_keys.add(word[:-2])  # "bear's" -> thêm "bear"
    return lookup_keys


# ============================================================
# BƯỚC 3 — Đọc GloVe theo dạng streaming, chỉ giữ key cần dùng
# ============================================================
def load_glove_subset(needed_keys: set):
    """
    Đọc glove.840B.300d.txt theo từng dòng, chỉ parse và giữ lại các dòng
    mà token đầu tiên khớp với 1 trong needed_keys.
    Trả về dict: token -> Tensor(300,)
    """
    print(f"Đang đọc GloVe (streaming) để tìm {len(needed_keys)} từ cần dùng ...")
    print("  (file ~5.65GB, quá trình này có thể mất vài phút, hãy kiên nhẫn chờ)")

    found = {}
    n_lines = 0

    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if n_lines % 2_000_000 == 0:
                print(f"    ... đã đọc {n_lines:,} dòng, tìm được {len(found)}/{len(needed_keys)} từ")

            # GloVe 840B đôi khi có dòng lỗi format (thiếu giá trị, token chứa lỗi
            # encoding) -> bọc try/except để bỏ qua an toàn, không crash cả script
            parts = line.rstrip("\n").split(" ")
            token = parts[0]

            if token not in needed_keys:
                continue

            try:
                vector = [float(x) for x in parts[1:]]
                if len(vector) != EMBED_DIM:
                    # Dòng lỗi (token gốc có khoảng trắng bị tách sai) -> bỏ qua
                    continue
                found[token] = torch.tensor(vector, dtype=torch.float32)
            except ValueError:
                continue

            # Tối ưu: nếu đã tìm đủ hết needed_keys thì dừng đọc sớm
            if len(found) == len(needed_keys):
                print(f"    -> Đã tìm đủ toàn bộ {len(needed_keys)} từ, dừng đọc sớm tại dòng {n_lines:,}")
                break

    print(f"  -> Hoàn tất: tìm được {len(found)}/{len(needed_keys)} từ trong GloVe.")
    return found


# ============================================================
# BƯỚC 4 — Build embedding cho từng entry trong vocab (xử lý cụm từ)
# ============================================================
def build_embeddings(vocab, glove_dict):
    """
    Với mỗi entry trong vocab, áp dụng đúng thứ tự ưu tiên đã chốt:
        1. Tra nguyên cụm từ trước.
        2. Nếu không có, tách từ; với mỗi từ, nếu có dạng sở hữu cách ("bear's")
           và không tìm thấy nguyên token, fallback tra phần gốc ("bear").
           Sau đó average các từ TÌM ĐƯỢC trong cụm.
        3. Nếu không từ nào tìm được -> OOV.

    Trả về:
        embeddings: Tensor (len(vocab), 300)  -- entry OOV sẽ là vector 0 tạm
        oov_list: list các entry bị OOV (để xử lý random init riêng lúc train)
    """
    embeddings = torch.zeros(len(vocab), EMBED_DIM)
    oov_list = []

    for idx, entry in enumerate(vocab):
        if entry in glove_dict:
            # Trường hợp 1: nguyên cụm từ có trong GloVe
            embeddings[idx] = glove_dict[entry]
            continue

        # Trường hợp 2: fallback average từng từ trong cụm
        words = entry.split(" ")
        found_vectors = []
        for w in words:
            if w in glove_dict:
                found_vectors.append(glove_dict[w])
            elif w.endswith("'s") and w[:-2] in glove_dict:
                # "bear's" không có nhưng "bear" có -> dùng vector của "bear"
                found_vectors.append(glove_dict[w[:-2]])

        if found_vectors:
            embeddings[idx] = torch.stack(found_vectors).mean(dim=0)
        else:
            # Trường hợp 3: OOV hoàn toàn
            oov_list.append(entry)
            # Giữ vector 0 tạm; lúc train cần random init riêng cho các entry này
            # (không dùng vector 0 thật khi train, sẽ xử lý ở bước embedding layer)

    return embeddings, oov_list


# ============================================================
# MAIN
# ============================================================
def main():
    object_vocab, predicate_vocab = collect_vocab()

    needed_keys = get_lookup_keys(object_vocab) | get_lookup_keys(predicate_vocab)
    glove_dict = load_glove_subset(needed_keys)

    print("\nĐang build embedding cho object vocab ...")
    object_embeddings, oov_objects = build_embeddings(object_vocab, glove_dict)

    print("Đang build embedding cho predicate vocab ...")
    predicate_embeddings, oov_predicates = build_embeddings(predicate_vocab, glove_dict)

    result = {
        "object_vocab": object_vocab,
        "predicate_vocab": predicate_vocab,
        "object_embeddings": object_embeddings,
        "predicate_embeddings": predicate_embeddings,
        "oov_objects": oov_objects,
        "oov_predicates": oov_predicates,
        "embed_dim": EMBED_DIM,
    }

    torch.save(result, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("HOÀN TẤT TRÍCH XUẤT GLOVE VOCAB")
    print("=" * 60)
    print(f"Object vocab: {len(object_vocab)} (OOV: {len(oov_objects)}, {len(oov_objects)/len(object_vocab)*100:.1f}%)")
    print(f"Predicate vocab: {len(predicate_vocab)} (OOV: {len(oov_predicates)}, {len(oov_predicates)/len(predicate_vocab)*100:.1f}%)")
    print(f"Đã lưu tại: {OUTPUT_PATH}")
    if oov_objects:
        print(f"\nMột số object OOV (ví dụ): {oov_objects[:10]}")
    if oov_predicates:
        print(f"Một số predicate OOV (ví dụ): {oov_predicates[:10]}")


if __name__ == "__main__":
    main()