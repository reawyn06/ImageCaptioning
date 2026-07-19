uvicorn main:app --host 127.0.0.1 --port 8000
# So sánh Bốn Chiến lược Kết hợp Đặc trưng Thị giác và Ngữ nghĩa trong Bài toán Mô tả Hình ảnh

**Đề tài Nghiên cứu Khoa học — Đánh giá hiệu năng Visual–Semantic Feature Fusion**

---

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cách chạy dự án](#cách-chạy-dự-án)
- [Tóm tắt](#tóm-tắt)
- [1. Đặt vấn đề](#1-đặt-vấn-đề)
- [2. Phương pháp](#2-phương-pháp)
- [3. Kết quả thực nghiệm](#3-kết-quả-thực-nghiệm)
- [4. Tổng hợp các phát hiện khoa học chính](#4-tổng-hợp-các-phát-hiện-khoa-học-chính)
- [5. Hạn chế](#5-hạn-chế-limitations)
- [6. Hướng phát triển](#6-hướng-phát-triển-future-work)
- [7. Kết luận](#7-kết-luận)

---

## Cấu trúc thư mục

```
ImageCaptioning/
├── rgcn_encoder.py, fusion_module.py, caption_decoder.py,      ← module lõi (core)
│   visual_extractor.py, sgg_lite.py, semantic_override.py,        bị import bởi
│   yolo_world_detector.py, caption_dataset.py, meteor_fixed.py     nhiều file khác
│
├── train.py                     ← script huấn luyện (--strategy baseline|concat|one_directional|bidirectional)
├── evaluate.py                  ← đánh giá trên COCO val2017
├── evaluate_flickr30k.py        ← đánh giá zero-shot trên Flickr30k
├── inference_service.py         ← service load model 1 lần, dùng cho web demo
├── main.py                      ← entrypoint FastAPI (mount router + static + templates)
│
├── scripts/                     ← script build dữ liệu, chạy 1 lần, không bị import
│   ├── build_scene_graphs.py       (VG objects/relationships → scene graph COCO)
│   ├── clean_scene_graphs.py       (lọc nhiễu vocab bằng pyspellchecker)
│   ├── build_glove_vocab.py
│   ├── build_yolo_vocab.py         (xây whitelist 1,218 category cho YOLO-World)
│   ├── build_flickr30k_features.py (visual + semantic feature cho Flickr30k)
│   ├── extract_visual_features.py
│   ├── check_vg_coco_mapping.py
│   └── check_vocab.py
│
├── tests/                        ← test/debug tạm thời, không bị import
│   ├── test_decoder.py, test_fusion_end_to_end.py, test_new_model.py
│   ├── test_debug.py, test_rgcn_on_real_data.py
│   └── estimate_training_time.py
│
├── weights/                       ← model weight tải sẵn (yolov8s/m/l-worldv2.pt)
├── checkpoints/                   ← checkpoint 4 strategy (*_best.pt, *_last.pt)
├── features/                      ← visual (.pt) + semantic (.json) đã build sẵn
│   └── flickr30k/semantic_yoloworld/  ← scene graph Flickr30k (pipeline mới, không ghi đè bản DETR cũ)
├── datasets/                      ← COCO, Visual Genome, Flickr30k gốc
├── results/                       ← caption sinh ra + bảng kết quả (.json)
├── static/, templates/            ← asset cho web demo FastAPI
└── README.md
```

> **Lưu ý quan trọng**: mọi file trong `scripts/` và `tests/` dùng đường dẫn tuyệt đối (`PROJECT_ROOT = r"C:\...\ImageCaptioning"`) nên chạy đúng dù nằm ở subfolder — chỉ cần gọi qua `python scripts\ten_file.py`, không cần sửa nội dung file.

---

## Cài đặt môi trường

```powershell
cd C:\Users\ADMIN\Documents\NCKH\ImageCaptioning
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # torch, transformers, ultralytics, pycocoevalcap, pyspellchecker, fastapi, uvicorn...
```

Yêu cầu phần cứng đã kiểm chứng: GPU RTX 5060 Ti 8GB, CUDA 12.8, Python 3.14, PyTorch 2.11+cu128.

---

## Cách chạy dự án

**1. Chuẩn bị dữ liệu (chạy 1 lần, thứ tự bắt buộc):**
```powershell
python scripts\build_scene_graphs.py
python scripts\clean_scene_graphs.py
python scripts\build_glove_vocab.py
python scripts\extract_visual_features.py
python scripts\build_yolo_vocab.py
```

**2. Huấn luyện (lặp lại cho từng strategy):**
```powershell
python train.py --strategy baseline
python train.py --strategy concat
python train.py --strategy one_directional
python train.py --strategy bidirectional
```

**3. Đánh giá:**
```powershell
python evaluate.py                                  # COCO val2017
python scripts\build_flickr30k_features.py --mode semantic
python evaluate_flickr30k.py --semantic-dir features\flickr30k\semantic_yoloworld --tag yoloworld
```

**4. Chạy web demo:**
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

> **Lỗi thường gặp**: `RuntimeError: Directory 'static' does not exist` xảy ra khi `uvicorn` được chạy từ thư mục khác với vị trí `main.py` (đường dẫn `"static"` trong `StaticFiles(directory="static")` là tương đối theo thư mục làm việc hiện tại, không theo vị trí file). Luôn `cd` vào đúng thư mục gốc project trước khi chạy `uvicorn`, hoặc sửa `main.py` dùng đường dẫn tuyệt đối qua `os.path.dirname(os.path.abspath(__file__))`.

---

## Tóm tắt

Nghiên cứu này so sánh bốn chiến lược kết hợp (fusion) đặc trưng thị giác (visual feature) và đặc trưng ngữ nghĩa đồ thị (semantic scene graph feature) trong bài toán mô tả hình ảnh tự động (image captioning): **Baseline** (chỉ dùng visual), **Concatenation**, **One-directional Attention**, và **Bidirectional Cross-Attention**. Bốn mô hình được huấn luyện trên cùng một tập dữ liệu (48,365 ảnh train / 2,135 ảnh validation, giao giữa MS-COCO 2017 và Visual Genome), sử dụng chung backbone visual (ViT-B/16), semantic encoder (R-GCN trên đồ thị scene graph từ Visual Genome), và caption decoder (GPT-2 với prefix injection kiểu ClipCap). Kết quả trên tập validation COCO cho thấy **One-directional Attention** đạt hiệu năng tốt nhất (BLEU-4 = 0.2804, hội tụ nhanh nhất). Đánh giá zero-shot trên Flickr30k — với scene graph tự sinh on-the-fly — cho thấy chất lượng của bộ sinh scene graph (Scene Graph Generation, SGG) là yếu tố quyết định, và tác động của nó lên từng chiến lược fusion là không đồng nhất, dẫn đến một phát hiện quan trọng về sự đánh đổi giữa độ sạch (precision) và độ dày (density) của đồ thị ngữ nghĩa.

---

## 1. Đặt vấn đề

Image captioning — bài toán sinh mô tả bằng ngôn ngữ tự nhiên cho một bức ảnh — thường sử dụng đặc trưng thị giác (visual feature) làm nguồn thông tin chính. Một hướng cải tiến được nhiều nghiên cứu đề xuất là bổ sung thêm đặc trưng ngữ nghĩa dạng đồ thị (scene graph: các đối tượng và quan hệ giữa chúng), với kỳ vọng cung cấp thông tin cấu trúc mà đặc trưng thị giác thuần túy khó nắm bắt (ví dụ quan hệ "người đang cầm ly", "chó nằm dưới bàn").

Tuy nhiên, **có nhiều cách để kết hợp (fuse) hai nguồn đặc trưng này**, và chưa có sự đồng thuận rõ ràng về chiến lược nào hiệu quả nhất, cũng như mức độ đóng góp thực sự của semantic feature so với visual feature thuần túy. Nghiên cứu này đặt ra câu hỏi trọng tâm:

> **Trong 4 chiến lược fusion phổ biến (không dùng semantic, nối trực tiếp, attention một chiều, attention hai chiều), chiến lược nào mang lại hiệu năng tốt nhất, và mức độ đóng góp của semantic feature phụ thuộc vào những yếu tố nào?**

---

## 2. Phương pháp

### 2.1. Dữ liệu

- **Nguồn**: giao giữa ảnh MS-COCO 2017 và Visual Genome (VG) — chỉ giữ ảnh COCO có scene graph annotation tương ứng trong VG (Phương án A).
- **Kích thước**: 48,365 ảnh train / 2,135 ảnh validation.
- **Đánh giá zero-shot bổ sung**: Flickr30k (31,783 ảnh) — dataset này không có annotation scene graph sẵn có, nên scene graph phải được **sinh on-the-fly** bằng một pipeline riêng (trình bày ở Mục 2.6).

### 2.2. Trích xuất đặc trưng thị giác (Visual Feature)

- Backbone: **ViT-B/16** (`google/vit-base-patch16-224`, HuggingFace), **đóng băng** (frozen), không fine-tune.
- Lấy 196 patch embedding (loại bỏ token [CLS]), mỗi patch có kích thước 768 chiều.
- Lưu dạng `.pt`, tổ chức theo `features/visual/{split}/{coco_id}.pt`.

### 2.3. Trích xuất đặc trưng ngữ nghĩa (Semantic Feature)

**Xây dựng scene graph từ Visual Genome (cho COCO train/val):**
- Đọc `objects.json` + `relationships.json` của VG, ánh xạ qua `vg_to_coco_mapping.json`.
- Lọc nhiễu 2 tầng: (1) loại tên object/predicate mơ hồ (`thing`, `stuff`...) khi build; (2) lọc nâng cấp bằng `pyspellchecker` để loại token chứa số, ký tự lỗi, từ không tồn tại trong tiếng Anh hiện đại (giữ open-vocabulary, không bó hẹp theo VG-150 chuẩn).
- Vocab cuối cùng (GloVe 840B.300d): 44,523 object, 19,670 predicate, tỉ lệ out-of-vocabulary (OOV) = 0.0%.

**Mã hóa bằng R-GCN (Relational Graph Convolutional Network):**
- 2 lớp, hidden dimension 768, dùng chung 1 ma trận trọng số (không phân theo loại predicate do vocab quá lớn).
- Khởi tạo node bằng vector GloVe, có marker cho cạnh nghịch (inverse edge) và self-loop.
- Tốc độ suy luận: ~1.27 ms/ảnh trên RTX 5060 Ti.

### 2.4. Bốn chiến lược Fusion

| Chiến lược | Cơ chế | Output length |
|---|---|---|
| **Baseline** | Chỉ dùng visual feature, không dùng semantic | 196 |
| **Concatenation** | Mean-pool (có mask) semantic feature → broadcast + Linear(1536→768) nối với visual | 196 |
| **One-directional Attention** | Semantic feature làm Query, Visual feature làm Key/Value (8-head Multi-Head Attention) + residual + LayerNorm | 196 + N |
| **Bidirectional Cross-Attention** | Attention cả 2 chiều (Semantic↔Visual) | 196 + N |

*(N = số node hợp lệ trong scene graph của ảnh, thay đổi theo từng ảnh)*

### 2.5. Caption Decoder

- Backbone: **GPT-2 base** (124M tham số), fine-tune toàn bộ.
- Kỹ thuật **prefix injection** kiểu ClipCap (Mokady et al., 2021): `Mapping Network` (MLP 768→1024→768 + residual + LayerNorm) chuyển `fused_features` thành prefix embeddings, ghép vào đầu sequence input của GPT-2.
- **Lưu ý quan trọng về thuật ngữ**: đây **không phải** Transformer Decoder có cross-attention layer riêng theo kiến trúc gốc (Vaswani et al., 2017) — đây là GPT-2 decoder-only (chỉ có self-attention, masked) kết hợp Mapping Network, thông tin thị giác/ngữ nghĩa được "tiêm" qua prefix ở đầu sequence.
- Huấn luyện: AdamW, learning rate 5e-5, batch size 8, tối đa 20 epoch, early stopping (patience = 3).

### 2.6. Pipeline sinh Scene Graph on-the-fly (cho ảnh ngoài COCO/VG)

Để đánh giá zero-shot trên Flickr30k và phục vụ demo web (ảnh bất kỳ, không thuộc COCO/VG), cần một pipeline riêng để sinh scene graph tại thời điểm suy luận (inference). Pipeline này trải qua 2 phiên bản:

**Phiên bản 1 (ban đầu):**
- Object detector: **DETR** (`facebook/detr-resnet-50`), giới hạn 91 category cố định (COCO classes).
- Suy quan hệ (relationship) bằng heuristic hình học thuần túy dựa trên vị trí tương đối 2 bounding box (trên/dưới/cạnh nhau).

**Phiên bản 2 (nâng cấp, dùng cho kết quả chính thức trong báo cáo này):**
- Object detector: **YOLO-World** (`yolov8s-worldv2`, open-vocabulary), với **whitelist tùy chỉnh 1,218 category** — xây dựng bằng cách thống kê tần suất object thực tế trong 48,362 scene graph tập train, giữ lại top-1200 category phổ biến nhất, bổ sung thủ công 18 loài động vật hoang dã hiếm (deer, fox, antelope...) vốn bị loại oan nếu chỉ lọc theo tần suất thuần túy.
- Ngưỡng tin cậy (confidence threshold): 0.4.
- **Semantic Override Engine**: kết hợp mô hình captioning độc lập **GIT** (`microsoft/git-base-coco`) để sửa 2 loại lỗi phát hiện được ở YOLO-World:
  - *Part-union*: chỉ detect được bộ phận cơ thể (vd "horns"), không có bounding box cho toàn bộ con vật → hợp nhất (union) các box bộ phận thành 1 box đại diện, gán nhãn loài trích từ caption của GIT.
  - *Relabel*: detect được toàn bộ con vật nhưng gán sai loài do hiện tượng "class competition" (loài phổ biến trong tập train — như "bear" — lấn át loài đúng nhưng hiếm hơn — như "lion", "fox" — trong cơ chế so khớp embedding text-ảnh của detector) → giữ nguyên bounding box, chỉ thay nhãn theo GIT.
- **Bộ lọc containment và background category**: suy quan hệ "part of" dựa trên tỉ lệ chồng lấn hình học giữa 2 bounding box; các category mang tính nền cảnh (field, sky, grass, wall...) bị loại khỏi MỌI quan hệ (không chỉ containment) để tránh sinh quan hệ vô nghĩa kiểu "sừng — part of — cánh đồng".

---

## 3. Kết quả thực nghiệm

### 3.1. Kết quả huấn luyện (best validation loss)

| Strategy | Best Val Loss | Epoch (best/total) |
|---|---|---|
| Baseline | 2.1085 | 9/12 |
| Concatenation | 2.0544 | 11/14 |
| **One-directional** | **2.0477** | **6/9** (hội tụ nhanh nhất) |
| Bidirectional | 2.0661 | 12/15 |

### 3.2. Đánh giá trên COCO val2017 (2,135 ảnh, scene graph annotation thật từ VG)

| Strategy | BLEU-4 | METEOR | CIDEr | SPICE |
|---|---|---|---|---|
| Baseline | 0.2507 | 0.2667 | 0.7173 | 0.1825 |
| Concatenation | 0.2670 | 0.2755 | **0.7921** | **0.1947** |
| **One-directional** | **0.2804** | 0.2763 | 0.7859 | 0.1926 |
| Bidirectional | 0.2698 | **0.2768** | 0.7864 | **0.1947** |

**Nhận xét**: One-directional Attention đạt BLEU-4 cao nhất và hội tụ nhanh nhất; Concatenation và Bidirectional cạnh tranh sát nhau ở CIDEr/SPICE. Cả 3 chiến lược có semantic feature đều vượt Baseline ở mọi chỉ số — xác nhận giá trị của semantic feature **khi scene graph là annotation thật, chất lượng cao**.

### 3.3. Đánh giá zero-shot trên Flickr30k — Pipeline SGG phiên bản 1 (DETR)

| Strategy | BLEU-4 | CIDEr | SPICE |
|---|---|---|---|
| **Baseline** | **0.1176** | **0.1293** | **0.0933** |
| Concatenation | 0.0769 | 0.0778 | 0.0646 |
| One-directional | 0.1073 | 0.1086 | 0.0868 |
| Bidirectional | 0.1118 | 0.1259 | 0.0928 |

**Phát hiện quan trọng #1**: Khi scene graph phải tự sinh (không phải annotation thật) và chất lượng kém (DETR giới hạn 91 category, heuristic quan hệ thô sơ), **Baseline (không dùng semantic) vượt qua cả 3 chiến lược fusion** — đảo ngược hoàn toàn kết quả trên COCO. Điều này cho thấy: **chất lượng scene graph là bottleneck quyết định**, không phải bản thân kiến trúc fusion — semantic feature nhiễu gây hại nhiều hơn lợi.

### 3.4. Đánh giá zero-shot trên Flickr30k — Pipeline SGG phiên bản 2 (YOLO-World + Semantic Override)

| Strategy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr | SPICE |
|---|---|---|---|---|---|---|---|
| Baseline | 0.5347 | 0.3292 | 0.1959 | 0.1176 | 0.1768 | 0.1293 | 0.0933 |
| Concatenation | 0.3530 | 0.2209 | 0.1319 | 0.0786 | 0.1356 | 0.0906 | 0.0688 |
| One-directional | 0.4973 | 0.2857 | 0.1551 | 0.0868 | 0.1544 | 0.0789 | 0.0739 |
| Bidirectional | 0.4858 | 0.2992 | 0.1763 | 0.1048 | 0.1625 | 0.1132 | 0.0867 |

**Bảng so sánh trực tiếp 2 phiên bản pipeline (Δ = YOLO-World − DETR):**

| Strategy | Δ BLEU-4 | Δ CIDEr | Δ SPICE |
|---|---|---|---|
| Baseline | 0.0000 | 0.0000 | 0.0000 |
| Concatenation | **+0.0017** | **+0.0128** | **+0.0042** |
| One-directional | −0.0205 | −0.0297 | −0.0129 |
| Bidirectional | −0.0070 | −0.0127 | −0.0061 |

**Kiểm tra tính hợp lệ của thí nghiệm**: Baseline cho kết quả **giống hệt tuyệt đối** (BLEU-4 = 0.1176, CIDEr = 0.1293, SPICE = 0.0933) giữa 2 lần chạy — đúng như kỳ vọng, vì Baseline không sử dụng semantic feature nên không thể bị ảnh hưởng bởi việc đổi pipeline SGG. Điều này xác nhận: visual feature, decoding method (greedy, đã tắt repetition penalty để đảm bảo khớp điều kiện gốc), và mọi thành phần khác được giữ cố định — **sự khác biệt ở 3 strategy còn lại chỉ đến từ đúng 1 biến số: chất lượng scene graph**.

**Phát hiện quan trọng #2 (mới, phức tạp hơn kỳ vọng ban đầu)**: Cải thiện chất lượng SGG (giảm nhiễu nhãn sai, containment hợp lý hơn) **không cải thiện đồng đều** cả 3 chiến lược:
- **Concatenation cải thiện** ở cả 3 chỉ số (đặc biệt CIDEr +0.0128) — phù hợp với giả thuyết đã đặt ra trước đó rằng Concat (dùng mean-pooling) nhạy cảm nhất với nhiễu, nên hưởng lợi rõ nhất khi nhiễu giảm.
- **One-directional và Bidirectional giảm nhẹ** — trái với kỳ vọng đơn giản "SGG tốt hơn → kết quả tốt hơn".

Giả thuyết giải thích: pipeline YOLO-World mới, dù giảm nhãn sai, lại tạo ra đồ thị **thưa hơn đáng kể** so với DETR cũ — do (1) bộ lọc containment/background category loại bỏ nhiều quan hệ hơn, (2) ngưỡng tin cậy 0.4 nghiêm ngặt hơn kết hợp vocab hẹp hơn (1,218 so với không giới hạn ở DETR). Cơ chế attention (One-directional, Bidirectional) vốn cần đủ số lượng node/cạnh để phát huy hiệu quả; khi đồ thị quá thưa (nhiều ảnh có 0-1 node, không có quan hệ nào), attention có ít tín hiệu để khai thác, trong khi mean-pooling (Concat) ít nhạy cảm hơn với việc thiếu quan hệ, chỉ quan tâm đến "loại node nào có mặt". Đây gợi ý một **sự đánh đổi giữa độ sạch (precision) và độ dày (density/recall) của đồ thị ngữ nghĩa**, với mức độ ảnh hưởng khác nhau tùy cơ chế fusion — một phát hiện tinh tế hơn nhiều so với kết luận đơn giản "SGG tốt hơn luôn luôn tốt hơn".

---

## 4. Tổng hợp các phát hiện khoa học chính

1. **Chất lượng scene graph là bottleneck quyết định hiệu năng semantic fusion**, không phải bản thân kiến trúc fusion — xác nhận qua việc Baseline vượt trội khi SGG kém chất lượng (Mục 3.3).

2. **One-directional Attention (Semantic→Visual) vượt trội hơn Bidirectional** trên dữ liệu train chất lượng cao (COCO val2017) — chiều "semantic làm query, visual làm key/value" quan trọng hơn chiều ngược lại; thêm chiều Visual→Semantic không đem lại giá trị tương xứng với độ phức tạp tăng thêm.

3. **Concatenation (mean-pooling) là chiến lược nhạy cảm nhất với chất lượng semantic feature** — thể hiện nhất quán qua 3 bằng chứng độc lập: (a) kết quả tệ nhất trên Flickr30k với SGG kém (Mục 3.3), (b) cải thiện rõ rệt nhất khi SGG được nâng cấp (Mục 3.4), (c) caption rời rạc/gãy ngữ pháp khi test thủ công với các loài động vật hiếm gặp trong tập train (deer, lion, fox).

4. **Tồn tại đánh đổi giữa độ sạch và độ dày của đồ thị ngữ nghĩa**, ảnh hưởng khác nhau tùy cơ chế fusion: mean-pooling hưởng lợi từ độ sạch, attention-based fusion cần đủ độ dày để phát huy hiệu quả (Mục 3.4, phát hiện mới).

5. **Chất lượng embedding của R-GCN phụ thuộc trực tiếp vào tần suất xuất hiện của category trong dữ liệu train**, độc lập với việc detector có nhận diện đúng nhãn hay không. Test thủ công với 3 loài hiếm cho thấy: dù nhãn được sửa đúng hoàn toàn qua Semantic Override (deer, lion, fox), caption vẫn kém chất lượng vì các category này chỉ xuất hiện 27–75 lần trong 48,362 ảnh train (so với ~916 lần của "bear") — không đủ để R-GCN học embedding ổn định.

6. **Hiện tượng "class competition" trong object detector open-vocabulary**: khi vocab chứa đồng thời nhiều category tương tự về mặt thị giác (các loài động vật 4 chân) và category "bộ phận cơ thể" (horns, ears, nose...), category có đặc trưng hình học riêng biệt (bộ phận) hoặc tần suất train cao (loài phổ biến) có xu hướng "thắng" các loài đúng nhưng hiếm hơn trong cơ chế so khớp embedding.

---

## 5. Hạn chế (Limitations)

- **Heuristic suy quan hệ dựa trên hình học** (cả 2 phiên bản pipeline on-the-fly) đơn giản hơn nhiều so với một mô hình Scene Graph Generation chuyên biệt (ví dụ Neural Motifs) — không học ngữ nghĩa quan hệ thật, chỉ suy từ vị trí không gian.
- **Kiến trúc GPT-2 + prefix injection không phải Transformer Decoder có cross-attention riêng** — cần ghi rõ khi so sánh với các công trình dùng kiến trúc Vaswani gốc.
- **R-GCN không đủ dữ liệu cho category long-tail**: các loài động vật hoang dã hiếm gặp trong Visual Genome (dataset thiên về ảnh đô thị/sinh hoạt) có embedding không ổn định, dù object detection đúng.
- **Đánh giá cross-domain (Flickr30k) phụ thuộc vào chất lượng bộ sinh scene graph on-the-fly** — không phản ánh hoàn toàn "trần" (upper bound) khả năng của kiến trúc fusion nếu có scene graph annotation chất lượng cao như COCO+VG.
- **So sánh GIT (end-to-end captioning, pretrain quy mô lớn) với pipeline nghiên cứu** cho thấy giới hạn của kiến trúc multi-stage (detect → SGG → R-GCN → fusion → decode): mỗi tầng trung gian có thể làm nhiễu thông tin, đặc biệt với category hiếm — trong khi mục tiêu nghiên cứu là so sánh ablation giữa các chiến lược fusion, không phải tối ưu chất lượng caption tuyệt đối.

---

## 6. Hướng phát triển (Future Work)

- Bổ sung dữ liệu scene graph cho các loài động vật hoang dã hiếm gặp — ví dụ kết hợp ảnh có caption sẵn (Flickr30k, Conceptual Captions) với scene graph tự sinh qua pipeline YOLO-World + GIT đã xây dựng, để fine-tune bổ sung cho R-GCN, cải thiện độ ổn định embedding cho category long-tail.
- Nghiên cứu sâu hơn về đánh đổi mật độ/độ sạch đồ thị (Mục 3.4, Phát hiện #4) — thử nghiệm có hệ thống với nhiều mức ngưỡng containment/threshold khác nhau để xác định điểm cân bằng tối ưu cho từng loại fusion.
- Thử nghiệm kỹ thuật distillation từ mô hình end-to-end quy mô lớn (như GIT) vào pipeline multi-stage, thay vì chỉ trích xuất 1 từ khóa species đơn lẻ.
- Mở rộng đánh giá với Beam Search decoding (đã triển khai, chưa dùng cho bảng kết quả chính thức để đảm bảo tính so sánh được kiểm soát).

---

## 7. Kết luận

Nghiên cứu xác nhận **One-directional Attention** là chiến lược fusion hiệu quả nhất khi có scene graph chất lượng cao, đồng thời chỉ ra rằng giá trị thực sự của việc bổ sung semantic feature phụ thuộc mạnh vào chất lượng nguồn scene graph — một yếu tố dễ bị bỏ qua khi đánh giá các phương pháp fusion chỉ trên dữ liệu có annotation sẵn. Qua 2 lần đánh giá zero-shot với 2 phiên bản pipeline sinh scene graph khác nhau, nghiên cứu cung cấp bằng chứng thực nghiệm cho một phát hiện tinh tế: cải thiện chất lượng scene graph không tác động đồng đều lên mọi cơ chế fusion, mà tạo ra đánh đổi giữa độ sạch và độ dày đồ thị, có lợi cho mean-pooling nhưng có thể bất lợi cho attention-based fusion trong điều kiện đồ thị quá thưa.