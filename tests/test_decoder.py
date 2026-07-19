from caption_decoder import CaptionDecoder

decoder = CaptionDecoder()  # sẽ tự tải GPT-2 base lần đầu chạy

# Test encode + loss với dữ liệu giả lập
import torch
fused_features = torch.randn(2, 50, 768)
fused_mask = torch.ones(2, 50, dtype=torch.long)
captions = ["a man riding a bike", "a dog running in the park"]
caption_ids, caption_mask = decoder.encode_captions(captions)

loss = decoder.compute_loss(fused_features, fused_mask, caption_ids, caption_mask)
print("Loss:", loss.item())