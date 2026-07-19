"""
visual_extractor.py
======================
Mục đích:
    Trích visual feature (patch embeddings) từ 1 ảnh THÔ (bất kỳ, không thuộc
    COCO) bằng ViT-B/16, ra đúng shape (1, 196, 768) -- khớp định dạng mà
    pipeline đã train kỳ vọng (xem features/visual/{split}/{coco_id}.pt đã
    chuẩn bị sẵn trước đó cho COCO).

Vì sao ra đúng 196 patch:
    ViT-B/16 input chuẩn 224x224, patch size 16x16 -> (224/16)^2 = 14*14 = 196
    patch, mỗi patch project thành vector 768-dim (đúng hidden_dim của
    "ViT-B/16" theo định nghĩa gốc, google/vit-base-patch16-224).

Lưu ý quan trọng (cần xác nhận khớp với cách build_visual_features.py gốc
của dự án đã trích xuất, vì module đó KHÔNG có trong các file đã xem):
    - Giả định mặc định: dùng last_hidden_state, BỎ token [CLS] (vị trí 0),
      giữ đúng 196 token patch còn lại -- đây là cách phổ biến nhất khi cần
      patch-wise feature (không cần pooled feature đại diện toàn ảnh).
    - Nếu cách trích xuất gốc của dự án (build_visual_features.py) dùng quy
      ước khác (vd có giữ CLS token, hoặc dùng model khác như CLIP ViT), cần
      đối chiếu lại checkpoint đã train -- vì Fusion Module/Decoder đã học
      theo đúng phân phối của features/visual gốc, lệch quy ước trích xuất
      có thể làm giảm chất lượng caption (không gây lỗi crash, chỉ giảm
      chất lượng output, nên khó nhận ra ngay nếu không kiểm tra kỹ).
"""

import torch
from PIL import Image

VIT_MODEL_NAME = "google/vit-base-patch16-224"
EXPECTED_NUM_PATCHES = 196
EXPECTED_HIDDEN_DIM = 768


class VisualFeatureExtractor:
    def __init__(self, device: torch.device, model_name: str = VIT_MODEL_NAME):
        from transformers import ViTImageProcessor, ViTModel

        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Args:
            image: PIL Image (bất kỳ kích thước/mode, sẽ tự convert RGB + resize)
        Returns:
            visual_features: (1, 196, 768) FloatTensor, trên đúng device đã khởi tạo
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (1, 197, 768) -- gồm CLS + 196 patch

        # Bỏ token CLS (vị trí 0), giữ đúng 196 patch token
        patch_features = last_hidden_state[:, 1:, :]  # (1, 196, 768)

        assert patch_features.shape[1] == EXPECTED_NUM_PATCHES, (
            f"Số patch ra {patch_features.shape[1]}, mong đợi {EXPECTED_NUM_PATCHES}. "
            f"Kiểm tra lại cấu hình ViTImageProcessor (image_size/patch_size)."
        )
        assert patch_features.shape[2] == EXPECTED_HIDDEN_DIM, (
            f"Hidden dim ra {patch_features.shape[2]}, mong đợi {EXPECTED_HIDDEN_DIM}."
        )

        return patch_features