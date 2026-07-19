"""
clean_scene_graphs.py
======================
Mục đích:
    Lọc lại scene graph đã xây ở build_scene_graphs.py (features/semantic/*.json),
    áp dụng bộ lọc nhiễu NÂNG CẤP (mạnh hơn bộ lọc cơ bản trước đó), để loại các
    trường hợp nhiễu rõ ràng phát hiện được từ kết quả GloVe OOV:
        - Token chứa số (vd '"10"', '"2002"')
        - Token có ký tự đặc biệt/dấu ngoặc kép thừa
        - Từ quá dài bất thường (dấu hiệu lỗi dính chữ, vd "behindabove")
        - Từ không tồn tại trong tiếng Anh (lỗi chính tả, vd "abovve", "bechs")

    KHÔNG bó hẹp theo vocab chuẩn VG-150 (150 object / 50 predicate) — vẫn giữ
    open-vocabulary, chỉ loại các trường hợp nhiễu/lỗi rõ ràng.

Cách kiểm tra "từ có hợp lệ trong tiếng Anh" (ĐÃ ĐỔI từ nltk.corpus.words sang
pyspellchecker theo quyết định cập nhật — nltk.corpus.words là corpus từ cổ/văn học,
thiếu nhiều từ ghép hiện đại (burger, donut, cellphone...) và không xử lý đúng các
biến thể bất quy tắc (held, children); pyspellchecker là dictionary chính tả thực
dụng hiện đại, tự xử lý đúng số nhiều/thì động từ mà KHÔNG cần suffix rules thủ công):
    Một từ được coi là HỢP LỆ nếu thỏa ít nhất 1 trong các điều kiện:
        1. Có trong dictionary của pyspellchecker (tiếng Anh hiện đại, ~thực dụng)
        2. Có trong MODERN_WHITELIST (một số từ rất hiện đại pyspellchecker vẫn thiếu,
           vd "selfie", "smartphone", "checkmark")

Output:
    Ghi đè lại các file trong features/semantic/{train2017,val2017}/{coco_id}.json
    với "objects" và "triples" đã lọc theo bộ lọc mới.
    (Bản cũ KHÔNG backup riêng vì có thể chạy lại build_scene_graphs.py để tái tạo
    nếu cần — nhưng script này sẽ in rõ thống kê trước/sau để bạn đối chiếu.)
"""

import json
import os
import re

from spellchecker import SpellChecker

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "features", "semantic")
TRAIN_DIR = os.path.join(SEMANTIC_DIR, "train2017")
VAL_DIR = os.path.join(SEMANTIC_DIR, "val2017")

MAX_WORD_LEN = 15  # từ dài hơn mức này coi là khả năng bị lỗi dính chữ

# ============================================================
# Whitelist bổ sung — pyspellchecker vẫn thiếu một số từ rất hiện đại/đặc thù.
# Bổ sung dần nếu phát hiện thêm từ đúng bị lọc nhầm sau khi chạy.
# ============================================================
MODERN_WHITELIST = {
    "selfie", "selfies", "smartphone", "smartphones", "checkmark", "checkmarks",
    "iphone", "ipad", "wifi", "bluetooth", "skateboarder", "skateboarders",
    "jetski", "jetskiing", "webcam", "webcams", "hoverboard",
    "emoji", "emojis", "hashtag", "gif", "gifs",
    # Bổ sung từ phát hiện bị loại nhầm ở lần chạy thực tế trên máy Rea:
    "earbuds", "earbud", "eyewear", "hoodie", "hoodies", "kneepads", "kneepad",
    "crouton", "croutons", "eclair", "eclairs", "moulding", "mouldings",
    "skatepark", "skateparks",
}

SPELL = SpellChecker()


# ============================================================
# BƯỚC 1 — Kiểm tra 1 từ đơn có hợp lệ không
# ============================================================
def is_known_word(word: str) -> bool:
    """Kiểm tra từ đơn (không phải cụm) có hợp lệ theo pyspellchecker hoặc
    whitelist bổ sung không. pyspellchecker tự xử lý đúng số nhiều/thì động từ
    (kể cả bất quy tắc như 'held', 'children'), không cần suffix rules thủ công."""
    word = word.lower()
    return word in SPELL or word in MODERN_WHITELIST


def has_invalid_characters(text: str) -> bool:
    """True nếu text chứa số hoặc ký tự đặc biệt ngoài chữ/khoảng trắng/gạch ngang/apostrophe.
    Dấu ' (apostrophe) được cho phép để giữ các cụm sở hữu cách hợp lệ
    (vd "bus's windows", "company's logo")."""
    return bool(re.search(r"[^a-zA-Z\s\-']", text))


def has_abnormal_word_length(text: str) -> bool:
    """True nếu có từ nào trong cụm dài hơn MAX_WORD_LEN (dấu hiệu dính chữ)."""
    return any(len(w) > MAX_WORD_LEN for w in text.split())


# ============================================================
# BƯỚC 2 — Kiểm tra 1 entry (object name hoặc predicate, có thể là cụm từ)
# ============================================================
def is_valid_entry(text: str) -> bool:
    """
    Một entry (object name / predicate) hợp lệ nếu:
        - Không chứa số/ký tự đặc biệt
        - Không có từ nào dài bất thường
        - TẤT CẢ từ trong cụm đều là từ tiếng Anh hợp lệ (hoặc biến thể/whitelist)
    """
    text = text.strip().lower()
    if not text:
        return False
    if has_invalid_characters(text):
        return False
    if has_abnormal_word_length(text):
        return False

    words_in_entry = text.split()
    if not all(is_known_word(w) for w in words_in_entry):
        return False

    return True


# ============================================================
# BƯỚC 3 — Lọc lại từng file scene graph
# ============================================================
def clean_split(split_dir: str, split_name: str):
    files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
    print(f"\nĐang lọc lại {split_name}: {len(files)} file ...")

    n_files_emptied = 0  # file không còn object/triple nào sau lọc
    total_objects_before = 0
    total_objects_after = 0
    total_triples_before = 0
    total_triples_after = 0
    removed_object_examples = set()
    removed_predicate_examples = set()

    for fname in files:
        fpath = os.path.join(split_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            record = json.load(f)

        old_objects = record.get("objects", [])
        old_triples = record.get("triples", [])
        total_objects_before += len(old_objects)
        total_triples_before += len(old_triples)

        # Lọc lại objects
        new_objects = []
        for obj in old_objects:
            if is_valid_entry(obj):
                new_objects.append(obj)
            elif len(removed_object_examples) < 30:
                removed_object_examples.add(obj)
        new_objects_set = set(new_objects)

        # Lọc lại triples: cả subject, predicate, object phải hợp lệ
        new_triples = []
        for s, p, o in old_triples:
            valid_s = is_valid_entry(s)
            valid_p = is_valid_entry(p)
            valid_o = is_valid_entry(o)

            if valid_s and valid_p and valid_o:
                new_triples.append([s, p, o])
                new_objects_set.add(s)
                new_objects_set.add(o)
            else:
                if not valid_p and len(removed_predicate_examples) < 30:
                    removed_predicate_examples.add(p)
                if not valid_s and len(removed_object_examples) < 30:
                    removed_object_examples.add(s)
                if not valid_o and len(removed_object_examples) < 30:
                    removed_object_examples.add(o)

        new_objects = sorted(new_objects_set)
        total_objects_after += len(new_objects)
        total_triples_after += len(new_triples)

        if not new_objects and not new_triples:
            n_files_emptied += 1
            os.remove(fpath)  # ảnh không còn semantic info hợp lệ -> bỏ khỏi dataset
            continue

        record["objects"] = new_objects
        record["triples"] = new_triples
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)

    print(f"  Objects: {total_objects_before} -> {total_objects_after} "
          f"(loại {total_objects_before - total_objects_after}, "
          f"{(total_objects_before - total_objects_after)/max(total_objects_before,1)*100:.1f}%)")
    print(f"  Triples: {total_triples_before} -> {total_triples_after} "
          f"(loại {total_triples_before - total_triples_after}, "
          f"{(total_triples_before - total_triples_after)/max(total_triples_before,1)*100:.1f}%)")
    print(f"  File bị xóa (không còn semantic info hợp lệ): {n_files_emptied}")

    return removed_object_examples, removed_predicate_examples, n_files_emptied


def main():
    print("=" * 60)
    print("LỌC NHIỄU NÂNG CẤP CHO SCENE GRAPH")
    print("=" * 60)

    obj_ex_train, pred_ex_train, emptied_train = clean_split(TRAIN_DIR, "train2017")
    obj_ex_val, pred_ex_val, emptied_val = clean_split(VAL_DIR, "val2017")

    print("\n" + "=" * 60)
    print("HOÀN TẤT")
    print("=" * 60)
    print(f"Tổng file bị xóa (cả train+val): {emptied_train + emptied_val}")
    print("\nVí dụ object bị loại (train):", sorted(obj_ex_train)[:20])
    print("Ví dụ predicate bị loại (train):", sorted(pred_ex_train)[:20])
    print("\n⚠️  Lưu ý: hãy xem qua các ví dụ bị loại — nếu thấy từ ĐÚNG bị loại nhầm,")
    print("   báo lại để mình bổ sung vào MODERN_WHITELIST rồi chạy lại.")


if __name__ == "__main__":
    main()