"""
build_flickr30k_features.py
============================
Mục đích:
    Trích xuất toàn bộ feature cần thiết cho pipeline Image Captioning
    trên dataset Flickr30k, bao gồm:
        1. Visual feature: ViT-B/16 patch embeddings (196, 768).
        2. Scene graph: sinh on-the-fly.

CẬP NHẬT QUAN TRỌNG (thay pipeline SGG cũ bằng pipeline mới):
    - TRƯỚC: DETR + heuristic hình học thuần túy (sgg_lite.py bản gốc)
    - SAU:   YOLO-World (vocab whitelist 1218 category) + Semantic Override
             (GIT giải quyết class-competition) + containment/background
             filter đã nâng cấp -- đây chính là pipeline đã dùng cho web demo.

    Output ghi ra THƯ MỤC MỚI (features/flickr30k/semantic_yoloworld/),
    KHÔNG ghi đè lên features/flickr30k/semantic/ (kết quả DETR cũ) --
    giữ lại cả 2 để có thể đối chiếu/tái tạo lại nếu cần, và để
    evaluate_flickr30k.py có thể chạy song song trên cả 2 phiên bản
    bằng cách đổi --semantic-dir.

    LƯU Ý: --mode visual KHÔNG CẦN chạy lại -- ViT-B/16 không thay đổi
    gì trong toàn bộ quá trình sửa lỗi, chỉ có phần semantic (detector +
    scene graph logic) thay đổi. Giữ nguyên features/flickr30k/visual/
    đã có sẵn để tiết kiệm thời gian.

Cách chạy:
    # Test tốc độ trước trên tập nhỏ (BẮT BUỘC làm trước khi chạy full,
    # vì GIT chạy thêm cho MỌI ảnh sẽ làm tổng thời gian tăng đáng kể
    # so với lần chạy DETR thuần túy trước đây)
    python build_flickr30k_features.py --mode semantic --max-images 300

    # Chạy full sau khi đã ước tính được ETA từ bước test
    python build_flickr30k_features.py --mode semantic

    # --resume: bỏ qua ảnh đã xử lý (mặc định True, an toàn vì ghi thư mục mới)
"""

import os
import sys
import csv
import json
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\ADMIN\Documents\NCKH\ImageCaptioning"
FLICKR30K_DIR = os.path.join(PROJECT_ROOT, "datasets", "flickr30k")
IMAGES_DIR = os.path.join(FLICKR30K_DIR, "Images")
CAPTIONS_FILE = os.path.join(FLICKR30K_DIR, "captions.txt")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "features", "flickr30k")
VISUAL_DIR = os.path.join(OUTPUT_ROOT, "visual")

# THAY ĐỔI: ghi ra thư mục MỚI, giữ nguyên semantic/ (DETR) làm bản lưu trữ
SEMANTIC_DIR_OLD_DETR = os.path.join(OUTPUT_ROOT, "semantic")             # bản gốc, KHÔNG đụng vào
SEMANTIC_DIR_NEW = os.path.join(OUTPUT_ROOT, "semantic_yoloworld")        # bản mới, pipeline YOLO-World+GIT

GLOVE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "glove_vocab.pt")

VISUAL_BATCH_SIZE = 1
LOG_INTERVAL = 200   # giảm từ 500 -> 200 vì pipeline mới chậm hơn, muốn theo dõi ETA sát hơn


# ============================================================
# BƯỚC 1 — Đọc danh sách ảnh unique từ captions.txt (KHÔNG ĐỔI)
# ============================================================
def load_image_ids(captions_file: str) -> list:
    image_ids = []
    seen = set()
    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            img_id = row[0].strip()
            if img_id and img_id not in seen:
                seen.add(img_id)
                image_ids.append(img_id)
    return image_ids


# ============================================================
# BƯỚC 2 — Visual feature extraction (KHÔNG ĐỔI, giữ nguyên để tham khảo
# nếu cần chạy lại từ đầu trên máy khác -- nhưng bạn KHÔNG CẦN gọi lại
# hàm này, vì features/flickr30k/visual/ đã có sẵn và không đổi)
# ============================================================
def extract_visual_features(image_ids: list, images_dir: str,
                             output_dir: str, device: torch.device,
                             resume: bool = True, batch_size: int = 1):
    from transformers import ViTImageProcessor, ViTModel

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Visual] Đang load ViT-B/16 ...")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model.eval()
    print(f"[Visual] Loaded. Device: {device}")

    if resume:
        todo = [img_id for img_id in image_ids
                if not os.path.exists(os.path.join(output_dir, img_id.replace(".jpg", ".pt")))]
        print(f"[Visual] Resume mode: {len(image_ids) - len(todo)} đã có sẵn, "
              f"còn {len(todo)} ảnh cần xử lý.")
    else:
        todo = image_ids
        print(f"[Visual] Full mode: xử lý tất cả {len(todo)} ảnh.")

    if not todo:
        print("[Visual] Không có ảnh nào cần xử lý. Bỏ qua.")
        return

    total = len(todo)
    done = 0
    errors = []
    t0 = time.time()

    for img_id in todo:
        img_path = os.path.join(images_dir, img_id)
        out_path = os.path.join(output_dir, img_id.replace(".jpg", ".pt"))
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            patch_features = outputs.last_hidden_state[0, 1:, :].cpu()
            torch.save(patch_features, out_path)
        except Exception as e:
            errors.append((img_id, str(e)))

        done += 1
        if done % LOG_INTERVAL == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  [Visual] {done}/{total} ảnh | "
                  f"{speed:.1f} ảnh/s | ETA: {eta/60:.1f} phút | lỗi: {len(errors)}")

    print(f"\n[Visual] Hoàn tất: {done} ảnh | {len(errors)} lỗi")
    if errors:
        print(f"  Ví dụ lỗi (5 đầu): {errors[:5]}")


# ============================================================
# BƯỚC 3 — Scene graph generation (ĐÃ THAY THẾ HOÀN TOÀN LOGIC CŨ)
# ============================================================
def build_scene_graphs_yolo_world(image_ids: list, images_dir: str,
                                   output_dir: str, device: torch.device,
                                   glove_vocab_path: str, resume: bool = True):
    """
    Sinh scene graph bằng pipeline MỚI: YOLO-World (vocab whitelist 1218
    category, threshold 0.4, model cố định yolov8s-worldv2) + Semantic
    Override (GIT, xử lý cả 2 tình huống: part_union và relabel) +
    containment/background filter đã nâng cấp -- TÁI SỬ DỤNG NGUYÊN VẸN
    build_scene_graph_for_image() từ sgg_lite.py, đúng logic đã dùng cho
    web demo, đảm bảo tính nhất quán giữa demo và eval.
    """
    sys.path.insert(0, PROJECT_ROOT)
    from sgg_lite import build_scene_graph_for_image
    from yolo_world_detector import YOLOWorldDetector
    from semantic_override import SemanticOverrideEngine
    from rgcn_encoder import GloveVocab

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Semantic] Đang load GloveVocab ...")
    glove_vocab = GloveVocab(glove_vocab_path)
    object_to_idx = glove_vocab.object_to_idx
    predicate_to_idx = glove_vocab.predicate_to_idx
    print(f"[Semantic] Vocab: {len(object_to_idx)} objects, "
          f"{len(predicate_to_idx)} predicates")

    print(f"[Semantic] Đang load YOLO-World (yolov8s-worldv2, vocab whitelist) ...")
    detector = YOLOWorldDetector(device)

    print(f"[Semantic] Đang load GIT (Semantic Override engine) ...")
    semantic_override = SemanticOverrideEngine(device)
    print(f"[Semantic] Cả 2 model đã sẵn sàng. Device: {device}")

    # Biến tạm dùng chung giữa 2 hook, giống hệt pattern trong inference_service.py
    state = {"forced_triples": []}

    def _detections_postprocess(detections, image):
        final_detections, forced_triples, _debug = semantic_override.apply_override(
            detections, object_to_idx, image=image
        )
        state["forced_triples"] = forced_triples
        return final_detections

    def _triples_postprocess(triples, detections):
        forced = state["forced_triples"]
        if not forced:
            return triples
        forced_pairs = {frozenset((s, o)) for s, _, o in forced}
        filtered = [t for t in triples if frozenset((t[0], t[2])) not in forced_pairs]
        return filtered + forced

    if resume:
        todo = [img_id for img_id in image_ids
                if not os.path.exists(
                    os.path.join(output_dir, img_id.replace(".jpg", ".json")))]
        print(f"[Semantic] Resume mode: {len(image_ids) - len(todo)} đã có sẵn, "
              f"còn {len(todo)} ảnh cần xử lý.")
    else:
        todo = image_ids
        print(f"[Semantic] Full mode: xử lý tất cả {len(todo)} ảnh.")

    if not todo:
        print("[Semantic] Không có ảnh nào cần xử lý. Bỏ qua.")
        return

    total = len(todo)
    done = 0
    n_empty = 0
    n_override_applied = 0   # THỐNG KÊ MỚI: theo dõi tần suất Semantic Override được kích hoạt
    errors = []
    t0 = time.time()

    for img_id in todo:
        img_path = os.path.join(images_dir, img_id)
        out_path = os.path.join(output_dir, img_id.replace(".jpg", ".json"))

        try:
            image = Image.open(img_path).convert("RGB")

            sg_result = build_scene_graph_for_image(
                image, detector, object_to_idx, predicate_to_idx,
                detections_postprocess=_detections_postprocess,
                triples_postprocess=_triples_postprocess,
            )

            safe_objects = sg_result["object_names"]
            safe_triples = sg_result["triples"]

            record = {
                "image_id": img_id,
                "objects": safe_objects,
                "triples": [list(t) for t in safe_triples],
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)

            if not safe_objects:
                n_empty += 1
            if state["forced_triples"]:
                n_override_applied += 1

        except Exception as e:
            errors.append((img_id, str(e)))
            with open(out_path, "w") as f:
                json.dump({"image_id": img_id, "objects": [], "triples": []}, f)

        done += 1
        if done % LOG_INTERVAL == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  [Semantic] {done}/{total} ảnh | "
                  f"{speed:.2f} ảnh/s | ETA: {eta/60:.1f} phút | "
                  f"rỗng: {n_empty} | override: {n_override_applied} | lỗi: {len(errors)}")

    print(f"\n[Semantic] Hoàn tất: {done} ảnh")
    print(f"  Rỗng (không có object hợp lệ): {n_empty}")
    print(f"  Semantic Override được kích hoạt: {n_override_applied} "
          f"({n_override_applied/max(done,1)*100:.1f}%)")
    print(f"  Lỗi: {len(errors)}")
    if errors:
        print(f"  Ví dụ lỗi (5 đầu): {errors[:5]}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Build Flickr30k features (pipeline YOLO-World + Semantic Override)")
    parser.add_argument("--mode", choices=["visual", "semantic", "all"],
                        default="semantic",   # đổi mặc định -- visual không cần chạy lại
                        help="Chế độ chạy. Mặc định 'semantic' vì visual đã có sẵn, không đổi.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Xử lý lại toàn bộ (ghi đè) -- dùng nếu muốn build lại từ đầu")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Giới hạn số ảnh -- BẮT BUỘC dùng để test tốc độ trước khi "
                             "chạy full 31,783 ảnh (vd --max-images 300)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Resume: {args.resume} | Max images: {args.max_images}")

    for path, name in [(IMAGES_DIR, "Images dir"), (CAPTIONS_FILE, "captions.txt"),
                       (GLOVE_VOCAB_PATH, "glove_vocab.pt")]:
        if not os.path.exists(path):
            print(f"LỖI: Không tìm thấy {name} tại {path}")
            sys.exit(1)

    print(f"\nĐang đọc danh sách ảnh từ {CAPTIONS_FILE} ...")
    image_ids = load_image_ids(CAPTIONS_FILE)
    print(f"Tổng số ảnh unique: {len(image_ids)}")

    if args.max_images:
        image_ids = image_ids[:args.max_images]
        print(f"[TEST MODE] Giới hạn còn {len(image_ids)} ảnh để test tốc độ.")

    if args.mode in ("visual", "all"):
        extract_visual_features(
            image_ids, IMAGES_DIR, VISUAL_DIR, device,
            resume=args.resume, batch_size=VISUAL_BATCH_SIZE
        )

    if args.mode in ("semantic", "all"):
        build_scene_graphs_yolo_world(
            image_ids, IMAGES_DIR, SEMANTIC_DIR_NEW, device,
            GLOVE_VOCAB_PATH, resume=args.resume
        )

    print("\n=== HOÀN TẤT ===")
    n_vis = len(list(Path(VISUAL_DIR).glob("*.pt"))) if os.path.exists(VISUAL_DIR) else 0
    n_sem_new = len(list(Path(SEMANTIC_DIR_NEW).glob("*.json"))) if os.path.exists(SEMANTIC_DIR_NEW) else 0
    n_sem_old = len(list(Path(SEMANTIC_DIR_OLD_DETR).glob("*.json"))) if os.path.exists(SEMANTIC_DIR_OLD_DETR) else 0
    print(f"Visual features: {n_vis} file (không đổi)")
    print(f"Semantic (YOLO-World+GIT, MỚI): {n_sem_new} file -> {SEMANTIC_DIR_NEW}")
    print(f"Semantic (DETR, cũ, giữ nguyên): {n_sem_old} file -> {SEMANTIC_DIR_OLD_DETR}")
    print(f"\nBước tiếp theo:")
    print(f"  python evaluate_flickr30k.py --semantic-dir {SEMANTIC_DIR_NEW} --tag yoloworld")


if __name__ == "__main__":
    main()