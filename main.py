"""
main.py
======================
FastAPI app cho web demo: upload 1 ảnh bất kỳ, sinh caption qua cả 4 strategy
fusion (baseline, concat, one_directional, bidirectional), hiển thị song song
để so sánh trực tiếp.

Cách chạy (trên máy Rea, trong .venv đã activate, tại PROJECT_ROOT):
    pip install fastapi uvicorn python-multipart --break-system-packages
    uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload

Sau đó mở trình duyệt: http://127.0.0.1:8000

Lưu ý:
    - Load model 1 LẦN lúc app khởi động (qua lifespan), KHÔNG load lại mỗi
      request -- nếu không sẽ rất chậm (mỗi lần load 4 checkpoint + ViT + DETR).
    - Lần đầu chạy sẽ tự tải pretrained weights (ViT-B/16 ~330MB, DETR ~160MB)
      từ HuggingFace Hub -- cần internet, chỉ tải 1 lần (cache lại).
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference_service import InferenceService

# ============================================================
# Lifespan: load model 1 lần lúc startup, dùng lại cho mọi request
# ============================================================
_service: InferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    print("=" * 60)
    print("Đang khởi tạo InferenceService (load 4 checkpoint + ViT + DETR) ...")
    print("Việc này có thể mất 1-2 phút lần đầu (tải pretrained weights).")
    print("=" * 60)
    _service = InferenceService()
    yield
    print("Đang tắt app, giải phóng model ...")
    _service = None


app = FastAPI(title="Image Captioning Demo - Semantic Fusion Comparison", lifespan=lifespan)


# ============================================================
# Endpoint chính: upload ảnh -> caption từ cả 4 strategy
# ============================================================
@app.post("/api/caption")
async def caption_image(file: UploadFile = File(...)):
    if _service is None:
        raise HTTPException(status_code=503, detail="Model đang khởi tạo, vui lòng thử lại sau vài giây.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là ảnh (jpg, png, ...).")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được file ảnh -- file có thể bị hỏng hoặc sai định dạng.")

    try:
        result = _service.caption_all_strategies(image)
    except Exception as e:
        # Lỗi không mong đợi ở tầng pipeline (vd OOM, lỗi model) -- trả lỗi rõ
        # ràng cho frontend hiển thị, không để app crash hoàn toàn.
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {e}")

    return JSONResponse(content=result)


@app.get("/api/health")
async def health_check():
    return {"status": "ok" if _service is not None else "loading"}


# ============================================================
# Frontend: serve trang HTML tĩnh đơn giản tại "/"
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")