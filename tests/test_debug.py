import json

with open(r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning\features\yolo_world_vocab_FULL.json", encoding="utf-8") as f:
    full_vocab = json.load(f)

vocab_dict = {item["label"]: item["count"] for item in full_vocab}
rank_lookup = {item["label"]: i + 1 for i, item in enumerate(full_vocab)}

for label in ["lion", "fox", "bear", "deer"]:
    if label in vocab_dict:
        print(f"'{label}': tần suất={vocab_dict[label]}, hạng={rank_lookup[label]}")
    else:
        print(f"'{label}': KHÔNG có trong vocab")