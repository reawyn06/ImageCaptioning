import torch
data = torch.load("features/glove_vocab.pt", weights_only=False)
preds = set(data["predicate_vocab"])
candidates = ["bigger than", "smaller than", "in front of", "behind", "next to", "larger than", "taller than", "in front", "front of"]
for c in candidates:
    status = "CO" if c in preds else "KHONG CO"
    print(f"{c!r}: {status}")

print()
print("Tong so predicate trong vocab:", len(preds))
print("Predicate chua front/behind/bigger/smaller/larger (neu co):")
for p in sorted(preds):
    if "front" in p or "behind" in p or "bigger" in p or "smaller" in p or "larger" in p:
        print(" ", repr(p))
