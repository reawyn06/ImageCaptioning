"""
yolo_world_detector.py
======================
Class YOLOWorldDetector — thay thế hoàn toàn ObjectDetector (DETR) trong
sgg_lite.py. Giữ đúng interface detect() -> List[DetectedObject] để không
cần sửa các phần khác của pipeline (filter_to_known_vocab, infer_triples,
build_scene_graph_for_image đều dùng chung interface này qua duck-typing).

Quyết định đã chốt với người dùng:
    - Class list: dùng ĐÚNG whitelist đã build ở build_yolo_vocab.py
      (features/yolo_world_vocab.txt — 1218 category, lọc theo tần suất
      thật từ 48,362 scene graph train + manual include cho động vật hiếm
      như "deer" vốn bị cắt oan nếu chỉ lọc theo tần suất thuần túy).
    - Confidence threshold: 0.4 (tránh vừa thiếu object thật vừa không
      ngập lụt node rác cho R-GCN).
    - Model: CỐ ĐỊNH yolov8s-worldv2 (không multi-fallback l/m/s) — đã
      xác nhận đủ chạy ổn với RTX 5060 Ti 8GB VRAM.
"""

import os
from typing import List, Optional
import torch
from PIL import Image

# Import dataclass + hàm chuẩn hóa từ sgg_lite (cùng thư mục).
# Lưu ý: sgg_lite.py KHÔNG import ngược lại file này, để tránh circular import.
from sgg_lite import DetectedObject, _normalize_text

# ============================================================
# CONFIG
# ============================================================
VOCAB_PATH = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning\features\yolo_world_vocab.txt"
MODEL_NAME = "yolov8s-worldv2.pt"
CONFIDENCE_THRESHOLD = 0.4
MAX_OBJECTS_PER_IMAGE = 12  # khớp giá trị đã dùng trong sgg_lite.py


def _load_vocab(vocab_path: str) -> List[str]:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Không tìm thấy whitelist vocab tại {vocab_path}. "
            f"Cần chạy build_yolo_vocab.py trước để sinh file này."
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError(f"File vocab {vocab_path} rỗng — kiểm tra lại build_yolo_vocab.py.")
    return classes


class YOLOWorldDetector:
    """
    Object detector dùng YOLO-World (open-vocabulary), thay thế DETR hoàn toàn.
    Interface giống hệt ObjectDetector cũ: detect(image) -> List[DetectedObject].
    """

    def __init__(self,
                 device: torch.device,
                 vocab_path: str = VOCAB_PATH,
                 model_name: str = MODEL_NAME,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        from ultralytics import YOLO

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_list = _load_vocab(vocab_path)

        print(f"[YOLOWorld] Đang load {model_name} ...")
        self.model = YOLO(model_name)
        self.model.set_classes(self.class_list)
        print(f"[YOLOWorld] Sẵn sàng — {len(self.class_list)} category "
              f"(vd: {self.class_list[:8]}...)")

    @torch.no_grad()
    def detect(self, image: Image.Image,
               score_threshold: Optional[float] = None) -> List[DetectedObject]:
        """
        Detect objects trong ảnh. Trả về List[DetectedObject], sort theo score
        giảm dần, giới hạn MAX_OBJECTS_PER_IMAGE — đúng hành vi ObjectDetector
        (DETR) cũ để phần code còn lại của pipeline không cần sửa gì thêm.
        """
        threshold = score_threshold if score_threshold is not None else self.confidence_threshold
        image_rgb = image.convert("RGB")

        results = self.model.predict(
            image_rgb,
            conf=threshold,
            verbose=False,
            device=str(self.device),
        )

        detections: List[DetectedObject] = []
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id >= len(self.class_list):
                    continue  # phòng hờ index lệch (không nên xảy ra, nhưng an toàn hơn crash

                score = float(boxes.conf[i].item())
                label = _normalize_text(self.class_list[class_id])
                xmin, ymin, xmax, ymax = [float(v) for v in boxes.xyxy[i].tolist()]

                detections.append(DetectedObject(
                    label=label, score=score, box=(xmin, ymin, xmax, ymax)
                ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:MAX_OBJECTS_PER_IMAGE]

    def debug_raw_predict(self, image, conf=0.05):
        """In TOÀN BỘ kết quả detect ở ngưỡng rất thấp, để xem 'deer' có
        từng được model tính ra hay không, và ở confidence bao nhiêu."""
        image_rgb = image.convert("RGB")
        results = self.model.predict(
            image_rgb, conf=conf, verbose=False, device=str(self.device)
        )
        boxes = results[0].boxes
        print(f"\n[DEBUG] Tổng số box ở conf>={conf}: {len(boxes)}")
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            label = self.class_list[cid] if cid < len(self.class_list) else "?"
            print(f"  {label:<20} score={score:.4f}")