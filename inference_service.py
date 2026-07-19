"""
inference_service.py
======================
Mục đích:
    Service trung tâm cho web app: load 1 LẦN DUY NHẤT (lúc khởi động FastAPI)
    toàn bộ 4 checkpoint (baseline, concat, one_directional, bidirectional) +
    ViT extractor + YOLO-World detector + GIT (Semantic Override engine) +
    GloveVocab, rồi expose 1 hàm duy nhất:
        caption_all_strategies(image: PIL.Image) -> dict[strategy, result]

    KHÔNG load lại model mỗi request (rất chậm, lãng phí VRAM) -- đây là lý
    do tách riêng thành service module, khởi tạo 1 lần ở module-level/app
    startup, dùng lại cho mọi request sau đó.

Quan trọng -- KHÔNG sửa code training/eval gốc:
    Toàn bộ model (RGCNEncoder, Fusion Module, CaptionDecoder) và checkpoint
    được TÁI SỬ DỤNG NGUYÊN VẸN từ train.py/evaluate.py, không sửa logic bên
    trong. Phần MỚI duy nhất ở layer này là: (1) trích visual feature từ ảnh
    thô thay vì đọc .pt có sẵn, (2) sinh scene graph on-the-fly thay vì đọc
    .json có sẵn từ Visual Genome, (3) Semantic Override -- kết hợp GIT để
    sửa lỗi "class competition" của YOLO-World khi phân loại loài động vật
    (xem semantic_override.py để biết chi tiết cơ chế và lý do).
"""

import os
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image

# Đảm bảo import được các module gốc của project (train.py, rgcn_encoder.py, ...)
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rgcn_encoder import GloveVocab          # noqa: E402
from train import ImageCaptioningModel, GLOVE_VOCAB_PATH  # noqa: E402

from visual_extractor import VisualFeatureExtractor
from sgg_lite import build_scene_graph_for_image
from yolo_world_detector import YOLOWorldDetector
from semantic_override import SemanticOverrideEngine, extract_candidate_context

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ALL_STRATEGIES = ["baseline", "concat", "one_directional", "bidirectional"]
MAX_GEN_LENGTH = 30


class InferenceService:
    """
    Khởi tạo 1 lần khi app start. Giữ tất cả model trong RAM/VRAM xuyên suốt
    đời sống của process FastAPI.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[InferenceService] Khởi tạo trên device: {self.device}")

        print("[InferenceService] Đang load GloveVocab ...")
        self.glove_vocab = GloveVocab(GLOVE_VOCAB_PATH)

        print("[InferenceService] Đang load ViT-B/16 (visual extractor) ...")
        self.visual_extractor = VisualFeatureExtractor(self.device)

        print("[InferenceService] Đang load YOLO-World (yolov8s-worldv2, vocab whitelist 1218 category) ...")
        self.detector = YOLOWorldDetector(self.device)

        print("[InferenceService] Đang load GIT (Semantic Override engine) ...")
        self.semantic_override = SemanticOverrideEngine(self.device)

        print("[InferenceService] Đang load 4 checkpoint (baseline/concat/one_directional/bidirectional) ...")
        self.models: Dict[str, ImageCaptioningModel] = {}
        for strategy in ALL_STRATEGIES:
            self.models[strategy] = self._load_model(strategy)

        # Biến tạm lưu debug info của Semantic Override cho request hiện tại
        # (được gán trong _detections_postprocess, đọc lại ngay sau đó trong
        # cùng 1 lời gọi caption_all_strategies -- KHÔNG shared giữa các
        # request khác nhau vì Python xử lý tuần tự từng request, không có
        # race condition trong ngữ cảnh FastAPI sync endpoint đơn giản này).
        self._last_override_debug = None
        self._last_forced_triples = []   # MỚI: Khởi tạo danh sách lưu tạm các quan hệ ép buộc[cite: 1]

        print("[InferenceService] Sẵn sàng nhận request.")

    def _load_model(self, strategy: str) -> ImageCaptioningModel:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{strategy}_best.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint cho strategy '{strategy}' tại {checkpoint_path}. "
                f"Đảm bảo đã train xong và checkpoint nằm đúng thư mục checkpoints/."
            )

        model = ImageCaptioningModel(strategy, self.glove_vocab).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"  -> [{strategy}] checkpoint epoch={checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.4f}")
        return model

    @torch.no_grad()
    def _generate_one(self, strategy: str, visual_features: torch.Tensor,
                      object_names: List[str], triples: List[Tuple[str, str, str]]) -> str:
        model = self.models[strategy]

        semantic_features, semantic_mask = model.rgcn.forward_batch([object_names], [triples])
        fused_features, fused_mask = model.fusion(visual_features, semantic_features, semantic_mask)
        captions = model.decoder.generate(
            fused_features, fused_mask,
            max_length=MAX_GEN_LENGTH,
            method="beam",  # đổi từ mặc định "greedy" sang "beam"
            num_beams=4,
            length_penalty=1.0,
        )
        return captions[0]

    def _detections_postprocess(self, detections, image):
        """
        Hook được gọi ngay sau YOLOWorldDetector.detect() và TRƯỚC infer_triples()
        (xem tham số detections_postprocess trong build_scene_graph_for_image,
        sgg_lite.py). Đây là nơi duy nhất InferenceService "biết" về sự tồn
        tại của Semantic Override -- sgg_lite.py và yolo_world_detector.py
        hoàn toàn không import gì từ semantic_override.py, giữ đúng nguyên
        tắc tách biệt module đã áp dụng xuyên suốt pipeline.
        """
        final_detections, forced_triples, override_debug = self.semantic_override.apply_override(
            detections, self.glove_vocab.object_to_idx, image=image
        )
        self._last_override_debug = override_debug
        self._last_forced_triples = forced_triples   # MỚI: lưu tạm để dùng ở hook tiếp theo[cite: 1]
        git_caption = override_debug.get("git_caption") or ""
        species_used = {override_debug.get("override_species")} if override_debug.get("override_species") else set()
        self._last_context_words = extract_candidate_context(
            git_caption, self.glove_vocab.object_to_idx, exclude=species_used, max_candidates=3
        )
        return final_detections

    def _triples_postprocess(self, triples, detections):
        """
        Ghép forced_triples (quan hệ ĐÃ BIẾT CHẮC CHẮN từ Semantic Override) vào
        kết quả suy luận hình học -- ưu tiên forced_triples: nếu 1 cặp (part,
        species) đã có forced "part of", loại bỏ mọi quan hệ khác (vd "next to")
        mau infer_triples() lỡ suy ra cho ĐÚNG CẶP đó, tránh vừa có "part of" vừa
        có "next to" cho cùng 1 cặp object -- gây mâu thuẫn ngữ nghĩa trong graph.
        """
        forced = self._last_forced_triples
        if not forced:
            return triples

        forced_pairs = {frozenset((s, o)) for s, _, o in forced}
        filtered = [t for t in triples if frozenset((t[0], t[2])) not in forced_pairs]
        return filtered + forced

    @torch.no_grad()
    def caption_all_strategies(self, image: Image.Image) -> dict:
        """
        Hàm chính gọi từ FastAPI endpoint. Xử lý 1 ảnh, trả caption cho cả 4
        strategy + thông tin scene graph debug (để hiển thị lên web cho người
        dùng thấy "model nhìn thấy gì" -- hữu ích khi demo trước giảng viên).
        """
        # ----- 1. Visual feature (dùng chung cho cả 4 strategy) -----
        visual_features = self.visual_extractor.extract(image)  # (1, 196, 768)

        # ----- 2. Scene graph on-the-fly (vocab-safe) -----
        # detections_postprocess=self._detections_postprocess sẽ chèn Semantic
        # Override (GIT) ngay sau bước YOLO-World detect, trước khi suy triples.
        # triples_postprocess loại bỏ các quan hệ hình học mâu thuẫn với tri thức cứng.
        sg_result = build_scene_graph_for_image(
            image, self.detector,
            self.glove_vocab.object_to_idx, self.glove_vocab.predicate_to_idx,
            detections_postprocess=self._detections_postprocess,
            triples_postprocess=self._triples_postprocess,   # MỚI: Đăng ký hook lọc triples[cite: 1]
        )
        object_names = sg_result["object_names"] + [
            w for w in self._last_context_words if w not in sg_result["object_names"]
        ]
        triples = sg_result["triples"]

        # ----- 3. Sinh caption cho cả 4 strategy -----
        captions = {}
        for strategy in ALL_STRATEGIES:
            try:
                captions[strategy] = self._generate_one(strategy, visual_features, object_names, triples)
            except Exception as e:
                captions[strategy] = f"[Lỗi khi sinh caption: {e}]"

        return {
            "captions": captions,
            "scene_graph_debug": {
                "detected_objects": [
                    {"label": d.label, "score": round(d.score, 3)} for d in sg_result["raw_detections"]
                ],
                "triples_used": [list(t) for t in triples],
                "n_objects_dropped_oov": sg_result["n_objects_dropped_oov"],
                "semantic_override": self._last_override_debug,
            },
        }