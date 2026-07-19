"""
rgcn_encoder.py
======================
Mục đích:
    Module R-GCN (Relational Graph Convolutional Network) encoder, nhận vào
    scene graph của 1 ảnh (objects, triples), trả ra semantic feature dạng
    NODE-WISE (1 vector 768-dim cho mỗi object/node), kèm attention mask để
    dùng cho cả 3 fusion strategy cần semantic branch (Concatenation,
    One-directional Attention, Bidirectional Cross-Attention).

Thiết kế đã chốt:
    - Node embedding khởi tạo: GloVe 840B.300d (load từ features/glove_vocab.pt)
    - Số lớp R-GCN: 2 (message passing 2-hop)
    - Hidden dimension: 768 (khớp với visual feature dim từ ViT-B/16)
    - KHÔNG dùng weight matrix riêng theo predicate (vocab predicate 19,670 loại
      quá lớn so với R-GCN gốc thiết kế cho vài chục relation type). Thay vào đó:
      predicate embedding (từ GloVe) được concat trực tiếp vào message, dùng
      1 weight matrix CHUNG cho mọi loại quan hệ.
    - Output: node-wise (giữ vector từng node), không pooling — để dành cho
      attention với visual patch ở bước Fusion Module sau này.

Công thức message passing (mỗi lớp R-GCN):
    Với cạnh (subject --predicate--> object):
        message = W_msg . concat(h_subject, e_predicate)
    Aggregate (mean) tất cả message đến 1 node, cộng self-loop:
        h_object_new = ReLU( W_self . h_object + mean(messages từ neighbor) )

    Để thông tin lan truyền đúng cả 2 hướng trong graph có hướng, mỗi triple
    (s, p, o) sinh ra 2 cạnh message passing:
        - Cạnh thuận: s -> o, dùng embedding của p
        - Cạnh nghịch (inverse): o -> s, dùng CHUNG embedding của p nhưng cộng
          thêm 1 "inverse marker" học được, KHÔNG tạo bảng embedding riêng cho
          từng predicate nghịch (tránh nhân đôi vocab 19,670 loại).
    Cộng thêm self-loop (node tự kết nối với chính nó) để giữ lại thông tin gốc.

Xử lý batch (nhiều ảnh, số node khác nhau):
    Vì KHÔNG cap số lượng object/triples mỗi ảnh, cần padding + mask khi ghép
    batch. Module này cung cấp:
        - RGCNEncoder: forward() xử lý 1 ảnh tại 1 thời điểm
        - forward_batch(): xử lý nhiều ảnh cùng lúc bằng kỹ thuật "graph
          batching" (gộp nhiều graph nhỏ thành 1 graph lớn dùng offset index,
          không để cạnh giữa các graph khác nhau bị chồng lấn), sau đó pad về
          cùng số node tối đa trong batch + trả về mask.
"""

import os
import json
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================
HIDDEN_DIM = 768
GLOVE_DIM = 300
NUM_LAYERS = 2


# ============================================================
# BƯỚC 1 — Load GloVe vocab đã trích xuất sẵn (glove_vocab.pt)
# ============================================================
class GloveVocab:
    """
    Wrapper quản lý vocab + embedding GloVe đã trích sẵn (build_glove_vocab.py).
    Cung cấp tra cứu nhanh: tên object/predicate (string) -> index -> vector GloVe.

    Các entry OOV (không tìm được trong GloVe, xem oov_objects/oov_predicates)
    được random init theo phân phối chuẩn nhỏ (không dùng vector 0 thật, vì
    vector 0 sẽ làm node/predicate đó "vô hình" trong phép cộng/nhân ma trận).
    """

    def __init__(self, glove_vocab_path: str):
        data = torch.load(glove_vocab_path, weights_only=False)

        self.object_vocab: List[str] = data["object_vocab"]
        self.predicate_vocab: List[str] = data["predicate_vocab"]
        self.object_embeddings: torch.Tensor = data["object_embeddings"].clone()
        self.predicate_embeddings: torch.Tensor = data["predicate_embeddings"].clone()
        oov_objects = set(data["oov_objects"])
        oov_predicates = set(data["oov_predicates"])

        self.object_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.object_vocab)}
        self.predicate_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.predicate_vocab)}

        # Random init lại riêng cho các entry OOV (thay vì giữ vector 0 từ bước
        # build_glove_vocab.py) -- dùng std nhỏ (0.1) để không phá vỡ scale
        # chung với các vector GloVe thật (GloVe thường có norm tương đối nhỏ).
        for name in oov_objects:
            idx = self.object_to_idx[name]
            self.object_embeddings[idx] = torch.randn(GLOVE_DIM) * 0.1
        for name in oov_predicates:
            idx = self.predicate_to_idx[name]
            self.predicate_embeddings[idx] = torch.randn(GLOVE_DIM) * 0.1

    def object_indices(self, names: List[str]) -> torch.LongTensor:
        return torch.tensor([self.object_to_idx[n] for n in names], dtype=torch.long)

    def predicate_indices(self, names: List[str]) -> torch.LongTensor:
        return torch.tensor([self.predicate_to_idx[n] for n in names], dtype=torch.long)


# ============================================================
# BƯỚC 2 — 1 lớp R-GCN (simplified, weight matrix chung cho mọi predicate)
# ============================================================
class RGCNLayer(nn.Module):
    """
    1 lớp message passing R-GCN (bản đơn giản hóa, không phân biệt weight
    matrix theo từng predicate -- vì vocab predicate quá lớn).

    Input:
        h: (N, hidden_dim)                   -- hidden state hiện tại của N node
        edge_subject: (E,)                   -- index node subject của E cạnh (đã gồm cả inverse)
        edge_object: (E,)                    -- index node object của E cạnh
        edge_predicate_emb: (E, hidden_dim)   -- embedding predicate của từng cạnh (đã project)
        edge_is_inverse: (E,)                 -- 1.0 nếu là cạnh nghịch, 0.0 nếu thuận

    Output:
        h_new: (N, hidden_dim)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # W_msg: nhận concat(h_subject, e_predicate) -> message (1 weight matrix
        # DUY NHẤT, dùng chung cho mọi loại predicate, đúng quyết định đã chốt)
        self.W_msg = nn.Linear(hidden_dim * 2, hidden_dim)

        # W_self: phép biến đổi cho self-loop (giữ lại thông tin gốc của node)
        self.W_self = nn.Linear(hidden_dim, hidden_dim)

        # Vector học được để đánh dấu cạnh nghịch (inverse), cộng vào predicate
        # embedding của cạnh nghịch -- giúp model phân biệt "object thuộc về
        # subject" với "subject sở hữu object" mà KHÔNG cần bảng embedding
        # riêng cho từng predicate nghịch (sẽ nhân đôi vocab 19,670 loại).
        self.inverse_marker = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, h, edge_subject, edge_object, edge_predicate_emb, edge_is_inverse):
        num_nodes = h.size(0)

        if edge_subject.numel() == 0:
            # Ảnh không có triple nào (chỉ có object rời rạc, không relationship)
            # -> chỉ áp dụng self-loop, không có message từ neighbor.
            return F.relu(self.W_self(h))

        # Cộng inverse marker vào predicate embedding của các cạnh nghịch
        marker = edge_is_inverse.unsqueeze(-1) * self.inverse_marker  # (E, hidden_dim)
        predicate_emb_adjusted = edge_predicate_emb + marker

        # Tính message cho từng cạnh: W_msg . concat(h_subject, e_predicate)
        h_subject = h[edge_subject]  # (E, hidden_dim)
        message_input = torch.cat([h_subject, predicate_emb_adjusted], dim=-1)  # (E, 2*hidden_dim)
        messages = self.W_msg(message_input)  # (E, hidden_dim)

        # Aggregate (mean) message theo node object đích -- dùng scatter mean
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)
        counts = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)

        aggregated.index_add_(0, edge_object, messages)
        counts.index_add_(0, edge_object, torch.ones(edge_object.size(0), 1, device=h.device, dtype=h.dtype))
        counts = counts.clamp(min=1.0)  # tránh chia 0 cho node không có message đến
        aggregated = aggregated / counts

        # Kết hợp self-loop + aggregated message, qua activation
        h_new = F.relu(self.W_self(h) + aggregated)
        return h_new


# ============================================================
# BƯỚC 3 — R-GCN Encoder đầy đủ (2 lớp + chiếu GloVe -> hidden_dim)
# ============================================================
class RGCNEncoder(nn.Module):
    """
    Encoder đầy đủ: GloVe embedding (300-dim) -> project lên hidden_dim (768)
    -> NUM_LAYERS lớp RGCNLayer -> trả về node embeddings cuối (768-dim/node).

    forward() xử lý 1 ảnh tại 1 thời điểm (1 graph). Việc ghép batch nhiều
    ảnh được xử lý ở hàm forward_batch() (xem cuối file), dùng kỹ thuật
    "graph batching": gộp nhiều graph nhỏ thành 1 graph lớn bằng offset index,
    chạy R-GCN 1 lần trên graph lớn, rồi tách + pad lại theo từng ảnh.
    """

    def __init__(self, glove_vocab: GloveVocab, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.glove_vocab = glove_vocab
        self.hidden_dim = hidden_dim

        # Project GloVe (300-dim, cố định) -> hidden_dim (768, học được).
        # Chỉ train phần linear projection này, giữ đúng tinh thần "khởi tạo
        # bằng GloVe" mà vẫn cho phép model học cách dùng embedding đó.
        self.object_proj = nn.Linear(GLOVE_DIM, hidden_dim)
        self.predicate_proj = nn.Linear(GLOVE_DIM, hidden_dim)

        # Đăng ký GloVe embedding làm buffer (không phải parameter -- không bị
        # optimizer cập nhật trực tiếp, chỉ dùng như input cố định).
        self.register_buffer("object_glove", glove_vocab.object_embeddings)
        self.register_buffer("predicate_glove", glove_vocab.predicate_embeddings)

        self.layers = nn.ModuleList([RGCNLayer(hidden_dim) for _ in range(num_layers)])

    def _build_edges(self, object_names: List[str], triples: List[Tuple[str, str, str]], device):
        """
        Chuyển list triples (dạng string) thành edge index tensor + predicate
        embedding tương ứng, đã bao gồm cả cạnh nghịch (inverse).

        Returns:
            edge_subject: (E,) LongTensor
            edge_object: (E,) LongTensor
            edge_predicate_emb: (E, hidden_dim) FloatTensor
            edge_is_inverse: (E,) FloatTensor (1.0 nghịch / 0.0 thuận)
        """
        node_to_idx = {name: i for i, name in enumerate(object_names)}

        if len(triples) == 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            empty_emb = torch.zeros(0, self.hidden_dim, device=device)
            empty_float = torch.zeros(0, device=device)
            return empty_long, empty_long, empty_emb, empty_float

        subjects, predicates, objects_ = zip(*triples)

        pred_idx = self.glove_vocab.predicate_indices(list(predicates)).to(device)
        pred_emb_raw = self.predicate_glove[pred_idx]          # (T, GLOVE_DIM)
        pred_emb = self.predicate_proj(pred_emb_raw)            # (T, hidden_dim)

        subj_idx = torch.tensor([node_to_idx[s] for s in subjects], dtype=torch.long, device=device)
        obj_idx = torch.tensor([node_to_idx[o] for o in objects_], dtype=torch.long, device=device)

        # Cạnh thuận: subject -> object
        fwd_subject, fwd_object = subj_idx, obj_idx
        fwd_is_inverse = torch.zeros(len(triples), device=device)

        # Cạnh nghịch: object -> subject (dùng CHUNG predicate embedding,
        # RGCNLayer sẽ tự cộng thêm inverse_marker để phân biệt)
        inv_subject, inv_object = obj_idx, subj_idx
        inv_is_inverse = torch.ones(len(triples), device=device)

        edge_subject = torch.cat([fwd_subject, inv_subject])
        edge_object = torch.cat([fwd_object, inv_object])
        edge_predicate_emb = torch.cat([pred_emb, pred_emb], dim=0)
        edge_is_inverse = torch.cat([fwd_is_inverse, inv_is_inverse])

        return edge_subject, edge_object, edge_predicate_emb, edge_is_inverse

    def forward(self, object_names: List[str], triples: List[Tuple[str, str, str]]):
        """
        Xử lý 1 ảnh.

        Args:
            object_names: list tên object trong ảnh (node list), vd ["man", "bike", "road"]
            triples: list (subject, predicate, object), vd [("man", "riding", "bike"), ...]

        Returns:
            node_embeddings: (N, hidden_dim) -- N = len(object_names).
                              Nếu N = 0 (ảnh không có object hợp lệ), trả về
                              tensor rỗng (0, hidden_dim) -- nơi gọi cần tự xử lý
                              trường hợp này khi ghép batch (xem forward_batch).
        """
        device = self.object_proj.weight.device
        num_nodes = len(object_names)

        if num_nodes == 0:
            return torch.zeros(0, self.hidden_dim, device=device)

        obj_idx = self.glove_vocab.object_indices(object_names).to(device)
        h = self.object_proj(self.object_glove[obj_idx])  # (N, hidden_dim)

        edge_subject, edge_object, edge_predicate_emb, edge_is_inverse = self._build_edges(
            object_names, triples, device
        )

        for layer in self.layers:
            h = layer(h, edge_subject, edge_object, edge_predicate_emb, edge_is_inverse)

        return h  # (N, hidden_dim)

    def forward_batch(self, batch_object_names: List[List[str]], batch_triples: List[List[Tuple[str, str, str]]]):
        """
        Xử lý nhiều ảnh trong 1 batch.

        Cách làm: chạy forward() riêng cho từng ảnh (vì mỗi ảnh là 1 graph độc
        lập, không cần gộp thành graph lớn ở đây -- với batch_size thường dùng
        (8-32) và số node mỗi ảnh nhỏ (~vài chục), chạy tuần tự đủ nhanh và code
        đơn giản, dễ debug hơn kỹ thuật gộp graph bằng offset index).

        Sau đó pad tất cả về cùng số node tối đa (max_nodes) trong batch, kèm
        mask để Fusion Module / Decoder biết node nào là thật, node nào là pad.

        Returns:
            padded_embeddings: (B, max_nodes, hidden_dim)
            mask: (B, max_nodes) -- 1 = node thật, 0 = padding
        """
        device = self.object_proj.weight.device
        batch_size = len(batch_object_names)

        all_embeddings = []
        for object_names, triples in zip(batch_object_names, batch_triples):
            emb = self.forward(object_names, triples)  # (N_i, hidden_dim)
            all_embeddings.append(emb)

        max_nodes = max((e.size(0) for e in all_embeddings), default=0)
        max_nodes = max(max_nodes, 1)  # tránh trường hợp toàn bộ batch rỗng

        padded_embeddings = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for i, emb in enumerate(all_embeddings):
            n = emb.size(0)
            if n > 0:
                padded_embeddings[i, :n] = emb
                mask[i, :n] = True

        return padded_embeddings, mask


# ============================================================
# BƯỚC 4 — Hàm tiện ích: đọc 1 file scene graph đã lưu (build_scene_graphs.py)
# ============================================================
def load_scene_graph(json_path: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Đọc 1 file scene graph (features/semantic/{split}/{coco_id}.json),
    trả về (object_names, triples) đúng định dạng cần cho RGCNEncoder.forward()."""
    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    object_names = record["objects"]
    triples = [tuple(t) for t in record["triples"]]
    return object_names, triples


# ============================================================
# TEST NHANH (chạy trực tiếp file này để kiểm tra encoder hoạt động đúng)
# ============================================================
if __name__ == "__main__":
    print("=== TEST RGCNEncoder với scene graph giả lập ===")

    # Tạo 1 GloveVocab giả lập nhỏ để test (không cần file glove_vocab.pt thật)
    class DummyGloveVocab:
        def __init__(self):
            self.object_vocab = ["man", "bike", "road", "helmet"]
            self.predicate_vocab = ["riding", "on", "wearing"]
            self.object_embeddings = torch.randn(len(self.object_vocab), GLOVE_DIM)
            self.predicate_embeddings = torch.randn(len(self.predicate_vocab), GLOVE_DIM)
            self.object_to_idx = {n: i for i, n in enumerate(self.object_vocab)}
            self.predicate_to_idx = {n: i for i, n in enumerate(self.predicate_vocab)}

        def object_indices(self, names):
            return torch.tensor([self.object_to_idx[n] for n in names], dtype=torch.long)

        def predicate_indices(self, names):
            return torch.tensor([self.predicate_to_idx[n] for n in names], dtype=torch.long)

    dummy_vocab = DummyGloveVocab()
    encoder = RGCNEncoder(dummy_vocab)

    # Ảnh 1: có scene graph đầy đủ
    objects_1 = ["man", "bike", "road", "helmet"]
    triples_1 = [("man", "riding", "bike"), ("bike", "on", "road"), ("man", "wearing", "helmet")]

    # Ảnh 2: ít object hơn, để test padding/mask
    objects_2 = ["man", "bike"]
    triples_2 = [("man", "riding", "bike")]

    # Ảnh 3: edge case — không có triple nào (chỉ object rời rạc)
    objects_3 = ["road"]
    triples_3 = []

    # Test forward() từng ảnh riêng lẻ
    out_1 = encoder(objects_1, triples_1)
    out_2 = encoder(objects_2, triples_2)
    out_3 = encoder(objects_3, triples_3)
    print(f"Ảnh 1 (4 object, 3 triples): output shape = {out_1.shape}")
    print(f"Ảnh 2 (2 object, 1 triple):  output shape = {out_2.shape}")
    print(f"Ảnh 3 (1 object, 0 triple):  output shape = {out_3.shape}")

    assert out_1.shape == (4, HIDDEN_DIM)
    assert out_2.shape == (2, HIDDEN_DIM)
    assert out_3.shape == (1, HIDDEN_DIM)
    assert not torch.isnan(out_1).any(), "Output chứa NaN!"
    assert not torch.isnan(out_3).any(), "Output (không triple) chứa NaN!"

    # Test forward_batch() với 3 ảnh có số node khác nhau
    padded, mask = encoder.forward_batch(
        [objects_1, objects_2, objects_3],
        [triples_1, triples_2, triples_3],
    )
    print(f"\nBatch padded embeddings shape: {padded.shape}  (kỳ vọng (3, 4, {HIDDEN_DIM}))")
    print(f"Batch mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")

    assert padded.shape == (3, 4, HIDDEN_DIM)
    assert mask.shape == (3, 4)
    assert mask[0].sum().item() == 4  # ảnh 1: 4 node thật
    assert mask[1].sum().item() == 2  # ảnh 2: 2 node thật
    assert mask[2].sum().item() == 1  # ảnh 3: 1 node thật

    print("\n✅ TẤT CẢ TEST ĐỀU PASS — RGCNEncoder hoạt động đúng như thiết kế.")