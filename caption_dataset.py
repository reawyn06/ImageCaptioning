"""
caption_dataset.py
======================
Mục đích:
    PyTorch Dataset + collate function, ghép 3 loại dữ liệu đã chuẩn bị sẵn
    cho 1 ảnh:
        - Visual feature (features/visual/{split}/{coco_id}.pt)
        - Semantic feature thô (features/semantic/{split}/{coco_id}.json) --
          CHƯA qua R-GCN ở đây, Dataset chỉ trả raw (objects, triples); R-GCN
          sẽ chạy trong training loop (vì R-GCN là nn.Module có tham số học
          được, cần nằm trong forward pass chính, không nên tiền xử lý 1 lần
          rồi cache -- nếu cache, sẽ "đóng băng" semantic feature, không cho
          R-GCN học cùng toàn bộ pipeline).
        - Caption (captions_{split}2017.json) -- RANDOM chọn 1 trong 5 caption
          mỗi lần được lấy ra (__getitem__), theo quyết định đã chốt (data
          augmentation nhẹ, mỗi epoch model thấy caption khác nhau cho cùng ảnh).

Dataset CHỈ trả về danh sách ảnh đã có ĐỦ CẢ 3 loại dữ liệu trên (visual +
semantic + caption) -- vì semantic chỉ có ~50,500/123,287 ảnh (Phương án A).

collate_fn xử lý:
    - Stack visual feature (luôn cố định 196, không cần pad)
    - Giữ list object_names/triples thô (R-GCN.forward_batch() sẽ tự pad sau,
      không pad ở đây)
    - Tokenize + pad caption (qua decoder.encode_captions(), gọi từ training
      loop, KHÔNG gọi trong collate_fn -- vì collate_fn chạy trên worker
      process riêng (num_workers>0), không nên giữ tokenizer ở đó để tránh
      lỗi pickling/đồng bộ; tokenize caption ngay trong training loop ở main
      process đơn giản và an toàn hơn).
"""

import os
import json
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    Args:
        project_root: đường dẫn gốc project (vd C:\\...\\ImageCaptioning)
        split: "train2017" hoặc "val2017"
    """

    def __init__(self, project_root: str, split: str):
        self.project_root = project_root
        self.split = split

        self.visual_dir = os.path.join(project_root, "features", "visual", split)
        self.semantic_dir = os.path.join(project_root, "features", "semantic", split)
        captions_path = os.path.join(project_root, "datasets", "coco", "annotations", f"captions_{split}.json")

        # ----- Load + group caption theo image_id -----
        print(f"[{split}] Đang đọc {captions_path} ...")
        with open(captions_path, "r", encoding="utf-8") as f:
            captions_data = json.load(f)

        self.captions_by_id: Dict[int, List[str]] = {}
        for ann in captions_data["annotations"]:
            img_id = ann["image_id"]
            self.captions_by_id.setdefault(img_id, []).append(ann["caption"])

        # ----- Xác định danh sách ảnh có ĐỦ CẢ 3 loại dữ liệu -----
        visual_ids = {f.replace(".pt", "") for f in os.listdir(self.visual_dir) if f.endswith(".pt")}
        semantic_ids = {f.replace(".json", "") for f in os.listdir(self.semantic_dir) if f.endswith(".json")}
        caption_ids = {str(k) for k in self.captions_by_id.keys()}

        valid_ids = visual_ids & semantic_ids & caption_ids
        self.image_ids = sorted(valid_ids, key=lambda x: int(x))

        print(f"[{split}] Visual: {len(visual_ids)}, Semantic: {len(semantic_ids)}, "
              f"Caption: {len(caption_ids)} -> Dùng được: {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        coco_id = self.image_ids[idx]

        # ----- Visual feature -----
        visual_path = os.path.join(self.visual_dir, f"{coco_id}.pt")
        visual_feat = torch.load(visual_path, weights_only=False)
        if isinstance(visual_feat, dict):
            visual_feat = visual_feat.get("features", visual_feat.get("embeddings"))

        # ----- Semantic feature (thô, chưa qua R-GCN) -----
        semantic_path = os.path.join(self.semantic_dir, f"{coco_id}.json")
        with open(semantic_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        objects = record["objects"]
        triples = [tuple(t) for t in record["triples"]]

        # ----- Caption: RANDOM 1 trong 5 mỗi lần lấy (data augmentation nhẹ) -----
        caption = random.choice(self.captions_by_id[int(coco_id)])

        return {
            "coco_id": coco_id,
            "visual_feature": visual_feat,   # (196, 768)
            "objects": objects,               # list[str]
            "triples": triples,               # list[tuple(s,p,o)]
            "caption": caption,                # str
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Ghép 1 batch các sample từ CaptionDataset.__getitem__ thành tensor/list
    phù hợp để đưa vào RGCNEncoder.forward_batch() + CaptionDecoder.

    Visual: stack trực tiếp (luôn cố định 196 patch).
    Semantic: giữ list thô (objects, triples) -- pad/mask xử lý trong
              RGCNEncoder.forward_batch(), KHÔNG xử lý ở đây.
    Caption: giữ list string thô -- tokenize xử lý trong training loop
             (gọi decoder.encode_captions()), KHÔNG xử lý ở đây.
    """
    coco_ids = [item["coco_id"] for item in batch]
    visual_features = torch.stack([item["visual_feature"] for item in batch])  # (B, 196, 768)
    batch_objects = [item["objects"] for item in batch]
    batch_triples = [item["triples"] for item in batch]
    captions = [item["caption"] for item in batch]

    return {
        "coco_ids": coco_ids,
        "visual_features": visual_features,
        "batch_objects": batch_objects,
        "batch_triples": batch_triples,
        "captions": captions,
    }