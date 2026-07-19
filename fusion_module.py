"""
fusion_module.py
======================
Mục đích:
    Module Fusion — kết hợp visual feature (từ ViT-B/16) và semantic feature
    (từ R-GCN encoder) theo 4 chiến lược khác nhau, dùng cho 4 thực nghiệm
    so sánh của đồ án.

Input chung cho cả 4 strategy:
    visual_features: (B, 196, 768)        -- patch embeddings từ ViT-B/16
    visual_mask:      None hoặc (B, 196)   -- luôn đủ 196 patch, không cần mask
                                               thực tế (giữ tham số để đồng bộ interface)
    semantic_features: (B, N, 768)         -- node embeddings từ R-GCN (đã pad)
    semantic_mask:    (B, N)               -- 1 = node thật, 0 = padding

Output chung cho cả 4 strategy (đưa vào Transformer Decoder làm cross-attention source):
    fused_features: (B, L, 768)            -- L tùy strategy (xem bảng dưới)
    fused_mask:     (B, L)                  -- mask tương ứng

    | Strategy | L                  | Ghi chú                                          |
    |----------|--------------------|--------------------------------------------------|
    | 1        | 196                | chỉ visual, không cần semantic                    |
    | 2        | 196                | concat(visual,semantic)+Linear, giữ nguyên 196    |
    | 3        | 196 + N            | visual gốc + N semantic node đã enrich từ visual  |
    | 4        | 196 + N            | visual đã enrich + N semantic đã enrich           |

Lưu ý quan trọng về Strategy 2 (Concatenation):
    Semantic feature là node-wise (N node, N thay đổi theo ảnh), nhưng visual
    là patch-wise cố định (196 patch). Để concat trực tiếp visual[i] với
    semantic, cần 1 vector semantic ĐẠI DIỆN duy nhất cho cả ảnh (pooled), rồi
    broadcast vector đó vào mọi patch trước khi concat + Linear projection.
    Đây là cách duy nhất hợp lý để giữ concat đơn giản (không dùng attention)
    mà vẫn xử lý được số node thay đổi giữa các ảnh.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM = 768
NUM_HEADS = 8


# ============================================================
# STRATEGY 1 — Baseline (chỉ visual, không có semantic)
# ============================================================
class BaselineFusion(nn.Module):
    """Không fusion gì cả -- trả lại nguyên visual feature.
    Giữ để đồng bộ interface với 3 strategy còn lại (dễ swap khi train)."""

    def forward(self, visual_features, semantic_features=None, semantic_mask=None):
        batch_size, num_patches, _ = visual_features.shape
        visual_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=visual_features.device)
        return visual_features, visual_mask


# ============================================================
# STRATEGY 2 — Concatenation Fusion
# ============================================================
class ConcatFusion(nn.Module):
    """
    Pool semantic node (mean pooling có mask) thành 1 vector đại diện/ảnh,
    broadcast vào mọi visual patch, concat rồi Linear projection về 768.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, visual_features, semantic_features, semantic_mask):
        batch_size, num_patches, hidden_dim = visual_features.shape

        # Mean pooling semantic node có mask (tránh tính cả phần padding)
        mask_f = semantic_mask.unsqueeze(-1).float()                    # (B, N, 1)
        summed = (semantic_features * mask_f).sum(dim=1)                # (B, hidden_dim)
        counts = mask_f.sum(dim=1).clamp(min=1.0)                        # (B, 1)
        semantic_pooled = summed / counts                                # (B, hidden_dim)

        # Broadcast vector pooled vào mọi patch, concat, project về 768
        semantic_broadcast = semantic_pooled.unsqueeze(1).expand(-1, num_patches, -1)  # (B, 196, hidden_dim)
        concatenated = torch.cat([visual_features, semantic_broadcast], dim=-1)         # (B, 196, 2*hidden_dim)
        fused = self.proj(concatenated)                                                  # (B, 196, hidden_dim)

        fused_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=visual_features.device)
        return fused, fused_mask


# ============================================================
# Khối Multi-Head Cross-Attention dùng chung cho Strategy 3 và 4
# ============================================================
class CrossAttentionBlock(nn.Module):
    """
    1 chiều cross-attention: Query từ nguồn A, Key/Value từ nguồn B.
    Dùng nn.MultiheadAttention (batch_first=True) của PyTorch, kèm residual
    connection + LayerNorm (kiến trúc chuẩn Transformer).

    key_padding_mask: (B, S_kv) -- True tại vị trí CẦN BỊ CHE (padding),
    đúng theo convention của nn.MultiheadAttention (ngược với mask "1=thật"
    mình dùng ở các nơi khác trong code -- cần đảo dấu khi gọi).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_src, kv_src, kv_padding_mask_true_is_real):
        """
        query_src: (B, Lq, hidden_dim)
        kv_src:    (B, Lkv, hidden_dim)
        kv_padding_mask_true_is_real: (B, Lkv) bool, True = vị trí thật (không phải padding)
        """
        # nn.MultiheadAttention cần key_padding_mask với True = cần CHE (padding)
        # -> đảo dấu so với convention "True = thật" đang dùng trong toàn bộ pipeline.
        key_padding_mask = ~kv_padding_mask_true_is_real

        attn_output, _ = self.attn(
            query=query_src, key=kv_src, value=kv_src,
            key_padding_mask=key_padding_mask,
        )
        # Residual + LayerNorm
        return self.norm(query_src + attn_output)


# ============================================================
# STRATEGY 3 — One-directional Attention (Semantic -> Q, Visual -> K/V)
# ============================================================
class OneDirectionalAttentionFusion(nn.Module):
    """
    Semantic node làm Query, Visual patch làm Key/Value -- mỗi semantic node
    "hỏi" toàn bộ 196 visual patch để lấy thông tin hình ảnh liên quan đến nó.

    Output: ghép (concat theo chiều sequence) N semantic node ĐÃ ENRICH với
    196 visual patch GỐC (không enrich) -- đúng theo quyết định đã chốt.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads)

    def forward(self, visual_features, semantic_features, semantic_mask):
        batch_size, num_patches, hidden_dim = visual_features.shape

        # Visual luôn đủ 196 patch thật, không cần mask thực tế -> tạo mask toàn True
        visual_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=visual_features.device)

        # Semantic node (Query) attend vào Visual patch (Key/Value)
        semantic_enriched = self.cross_attn(
            query_src=semantic_features, kv_src=visual_features, kv_padding_mask_true_is_real=visual_mask,
        )  # (B, N, hidden_dim)

        # Ghép visual gốc + semantic đã enrich theo chiều sequence
        fused = torch.cat([visual_features, semantic_enriched], dim=1)        # (B, 196+N, hidden_dim)
        fused_mask = torch.cat([visual_mask, semantic_mask], dim=1)            # (B, 196+N)

        return fused, fused_mask


# ============================================================
# STRATEGY 4 — Bidirectional Cross-Attention
# ============================================================
class BidirectionalAttentionFusion(nn.Module):
    """
    Cả 2 chiều cross-attention:
        - Visual (Query) attend vào Semantic (Key/Value)  -> visual_enriched
        - Semantic (Query) attend vào Visual (Key/Value)   -> semantic_enriched
    Output: ghép visual_enriched + semantic_enriched theo chiều sequence.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.visual_to_semantic = CrossAttentionBlock(hidden_dim, num_heads)
        self.semantic_to_visual = CrossAttentionBlock(hidden_dim, num_heads)

    def forward(self, visual_features, semantic_features, semantic_mask):
        batch_size, num_patches, hidden_dim = visual_features.shape
        visual_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=visual_features.device)

        # Visual (Query) attend vào Semantic (Key/Value) -- cần mask semantic
        # vì semantic có padding (N thay đổi theo ảnh)
        visual_enriched = self.visual_to_semantic(
            query_src=visual_features, kv_src=semantic_features, kv_padding_mask_true_is_real=semantic_mask,
        )  # (B, 196, hidden_dim)

        # Semantic (Query) attend vào Visual (Key/Value) -- visual không có padding
        semantic_enriched = self.semantic_to_visual(
            query_src=semantic_features, kv_src=visual_features, kv_padding_mask_true_is_real=visual_mask,
        )  # (B, N, hidden_dim)

        fused = torch.cat([visual_enriched, semantic_enriched], dim=1)         # (B, 196+N, hidden_dim)
        fused_mask = torch.cat([visual_mask, semantic_mask], dim=1)             # (B, 196+N)

        return fused, fused_mask


# ============================================================
# Factory function — tạo đúng fusion module theo tên strategy
# ============================================================
def build_fusion_module(strategy: str, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS) -> nn.Module:
    """
    strategy: 1 trong ["baseline", "concat", "one_directional", "bidirectional"]
    """
    if strategy == "baseline":
        return BaselineFusion()
    elif strategy == "concat":
        return ConcatFusion(hidden_dim)
    elif strategy == "one_directional":
        return OneDirectionalAttentionFusion(hidden_dim, num_heads)
    elif strategy == "bidirectional":
        return BidirectionalAttentionFusion(hidden_dim, num_heads)
    else:
        raise ValueError(f"Strategy không hợp lệ: {strategy}. "
                          f"Chọn 1 trong: baseline, concat, one_directional, bidirectional")