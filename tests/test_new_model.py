import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 1. Thiết bị chạy
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Đang chạy mô hình GIT trên thiết bị: {device.upper()}")

# 2. Tải mô hình GIT-base (Hoàn toàn tương thích với transformers 5.12.1 của bạn)
model_id = "microsoft/git-base-coco"
print("⏳ Đang tải mô hình từ Hugging Face...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Danh sách ảnh của bạn
test_images = ["deer.jpg", "OIP.webp", "alainaudet-fox-715588_1920.jpg", "ambquinn-lion-7540888.jpg"]

print("\n🚀 KẾT QUẢ KIỂM TRA NHẬN DIỆN VẠN VẬT:")
print("=" * 60)

for img_path in test_images:
    try:
        image = Image.open(img_path).convert("RGB")

        # Tiền xử lý ảnh
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Sinh caption
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"🖼️ Ảnh: {img_path}")
        print(f"👉 Kết quả mô hình đoán: {generated_text}\n")

    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh {img_path}: {e}\n")

print("=" * 60)