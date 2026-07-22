"""
transformer_caption_decoder.py
======================
Mục đích:
    THAY THẾ caption_decoder.py (GPT-2 + prefix injection) bằng 1 Transformer
    Decoder ĐÚNG NGUYÊN BẢN theo kiến trúc gốc (Vaswani et al., "Attention Is
    All You Need", NeurIPS 2017), HUẤN LUYỆN TỪ ĐẦU (from scratch) -- không
    dùng bất kỳ pretrained weight nào.

LÝ DO THAY ĐỔI (theo yêu cầu giáo viên hướng dẫn):
    1. GPT-2 fine-tune mang theo kiến thức ngôn ngữ khổng lồ từ pretraining
       trên corpus text khổng lồ ngoài phạm vi bài toán -- khi so sánh 4
       chiến lược fusion, sự khác biệt giữa chúng dễ bị "che lấp" bởi khả
       năng ngôn ngữ có sẵn của GPT-2, làm giảm tính khách quan của ablation
       study (ảnh hưởng trực tiếp đến độ tin cậy của Mục 3, 4 trong báo cáo).
    2. GPT-2 là kiến trúc decoder-only (chỉ có MASKED SELF-ATTENTION), không
       có cross-attention layer riêng -- "prefix injection" (ghép fused_features
       vào đầu sequence, để GPT-2 "nhìn" qua self-attention) là 1 kỹ thuật
       thực dụng (ClipCap, Mokady 2021) nhưng KHÔNG PHẢI cách kiến trúc
       Transformer decoder NGUYÊN BẢN xử lý thông tin điều kiện (conditioning
       information) -- đây chính là điều giáo viên đã lưu ý.
    3. Decoder mới có CROSS-ATTENTION THẬT: mỗi bước sinh từ, Query đến từ
       caption đang sinh, Key/Value đến từ fused_features -- đây MỚI là cách
       các paper captioning classic (Bottom-Up Top-Down, SGAE) thực sự dùng
       (dù bản thân họ dùng LSTM + attention, không phải Transformer, nhưng
       cơ chế "decoder attend vào visual/semantic feature" là tương đương).

KIẾN TRÚC:
    Token Embedding (học từ đầu, vocab riêng -- xem build_caption_vocab.py)
        + Positional Encoding (sinusoidal cố định, ĐÚNG công thức Vaswani 2017)
        -> nn.TransformerDecoder (PyTorch built-in, đã kiểm chứng, KHÔNG tự
           cài đặt lại attention từ đầu -- giảm rủi ro bug, đúng chuẩn Vaswani):
               Mỗi lớp: Masked Self-Attention (chỉ nhìn token trước đó)
                        -> Cross-Attention (Query=caption, Key/Value=fused_features)
                        -> Feed-Forward Network
                        (mỗi khối đều có residual connection + LayerNorm)
        -> Linear output projection -> vocab_size

    Input/Output GIỮ NGUYÊN INTERFACE với caption_decoder.py (GPT-2) cũ:
        forward(fused_features, fused_mask, caption_ids, caption_mask) -> logits
        compute_loss(...) -> loss
        generate(...) -> list[str]
        encode_captions(...) -> (caption_ids, caption_mask)
    -> train.py / evaluate.py / evaluate_flickr30k.py / inference_service.py
       CHỈ CẦN đổi 1-2 dòng import + khởi tạo (xem hướng dẫn migrate ở cuối
       file), KHÔNG cần sửa logic training/eval nào khác.

Weight tying (Press & Wolf, "Using the Output Embedding to Improve Language
Models", EACL 2017): output projection layer DÙNG CHUNG ma trận trọng số với
token embedding -- giảm số tham số, đã được chứng minh cải thiện perplexity
cho mô hình ngôn ngữ cỡ nhỏ/vừa (đúng tình huống ở đây -- vocab chỉ vài nghìn
từ, hidden_dim 768, KHÔNG có pretraining để "bù" cho việc thiếu tham số).
"""

import math
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM = 768
NUM_LAYERS = 6          # số lớp decoder -- Vaswani gốc dùng 6 cho base model
NUM_HEADS = 8           # khớp với NUM_HEADS đã dùng trong fusion_module.py
DIM_FEEDFORWARD = 2048  # khớp tỉ lệ 4x hidden_dim của Vaswani gốc (768*~2.67, làm tròn theo base Transformer)
DROPOUT = 0.1
MAX_POSITIONS = 64      # độ dài caption tối đa (kể cả BOS/EOS) -- dư so với MAX_CAPTION_LENGTH=30 đang dùng


# ============================================================
# BƯỚC 1 — Positional Encoding (sinusoidal, ĐÚNG công thức Vaswani et al. 2017)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Dùng bảng CỐ ĐỊNH (không học được) -- đúng lựa chọn gốc của Vaswani 2017
    cho "base model" (bài báo cũng thử learned positional embedding, kết quả
    gần như tương đương, nên chọn sinusoidal vì không tốn thêm tham số và tự
    ngoại suy được với chuỗi dài hơn lúc train nếu cần).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, max_len: int = MAX_POSITIONS):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, hidden_dim) -- buffer, không phải parameter

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = token_embeddings.size(1)
        return token_embeddings + self.pe[:, :seq_len, :]


# ============================================================
# BƯỚC 2 — Tokenizer đơn giản (PHẢI khớp CHÍNH XÁC với build_caption_vocab.py)
# ============================================================
def _tokenize_text(text: str) -> List[str]:
    """
    QUAN TRỌNG: hàm này PHẢI cho ra kết quả GIỐNG HỆT hàm tokenize() trong
    build_caption_vocab.py -- nếu lệch nhau (vd 1 bên giữ dấu chấm, 1 bên bỏ),
    sẽ có mismatch giữa vocab đã xây và cách encode lúc train/eval, khiến
    tỉ lệ <unk> tăng bất thường mà không ai nhận ra ngay.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return [w for w in text.split() if w]


# ============================================================
# BƯỚC 3 — Caption Decoder (Transformer chuẩn, huấn luyện từ đầu)
# ============================================================
class CaptionDecoderTransformer(nn.Module):
    """
    Interface đồng nhất với CaptionDecoder (GPT-2) cũ trong caption_decoder.py
    -- xem hướng dẫn migrate ở cuối file để biết cách thay thế trong
    train.py / evaluate.py / evaluate_flickr30k.py / inference_service.py.
    """

    def __init__(
        self,
        vocab_path: str,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        max_positions: int = MAX_POSITIONS,
        tie_weights: bool = True,
    ):
        super().__init__()

        vocab_data = torch.load(vocab_path, weights_only=False)
        self.word2idx: dict = vocab_data["word2idx"]
        self.idx2word: List[str] = vocab_data["idx2word"]
        self.vocab_size = len(self.idx2word)

        special = vocab_data["special_tokens"]
        self.pad_id = special["pad"]
        self.bos_id = special["bos"]
        self.eos_id = special["eos"]
        self.unk_id = special["unk"]

        self.hidden_dim = hidden_dim
        self.max_positions = max_positions

        # ----- Token embedding (học từ đầu -- KHÔNG pretrained) -----
        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.pad_id)

        # FIX (phát hiện qua smoke test -- loss ban đầu ~206 thay vì ~ln(vocab_size)~9.2):
        # nn.Embedding mặc định init N(0, 1) -- QUÁ LỚN khi weight bị TIE với
        # output projection (tie_weights=True bên dưới). Vì output_proj DÙNG
        # TRỰC TIẾP ma trận này (không qua embed_scale, chỉ áp dụng ở input),
        # std=1 khiến logits có phương sai ~hidden_dim (~768) -> với vocab
        # 10,297 lớp, giá trị logit lớn nhất do thuần túy ngẫu nhiên có thể
        # lên tới hàng trăm, làm cross-entropy loss ban đầu vọt lên bất
        # thường (206-211 thay vì ~9.2 kỳ vọng cho phân phối gần đều).
        #
        # Khởi tạo lại với std = 1/sqrt(hidden_dim) -- ĐÚNG CẶP với
        # embed_scale = sqrt(hidden_dim) dùng ở _embed_tokens(): sau khi
        # nhân embed_scale, giá trị embedding đưa vào decoder có std ~1
        # (khớp scale positional encoding), trong khi RAW weight (dùng trực
        # tiếp cho output_proj do tie) vẫn giữ std nhỏ, giữ logit ban đầu ở
        # mức hợp lý (~ln(vocab_size), đúng kỳ vọng cho model chưa train).
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=hidden_dim ** -0.5)
        with torch.no_grad():
            # nn.Embedding mặc định tự zero-init hàng padding_idx, nhưng lệnh
            # normal_() ở trên ghi đè LÊN TOÀN BỘ weight (kể cả hàng pad) --
            # cần zero lại hàng pad để giữ đúng hành vi mặc định của PyTorch.
            self.token_embedding.weight[self.pad_id].zero_()

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_positions)
        self.embed_dropout = nn.Dropout(dropout)
        # Scale embedding theo sqrt(d_model) -- đúng thực hành Vaswani 2017
        # (Mục 3.4 bài báo gốc): giữ scale hợp lý giữa embedding và positional
        # encoding trước khi cộng với nhau.
        self.embed_scale = math.sqrt(hidden_dim)

        # ----- Transformer Decoder (dùng nn.TransformerDecoder built-in của
        # PyTorch -- đã được kiểm chứng rộng rãi, đúng kiến trúc Vaswani 2017:
        # mỗi layer = Masked Self-Attention -> Cross-Attention -> FFN, đều có
        # residual + LayerNorm. KHÔNG tự cài đặt lại từ đầu để giảm rủi ro bug
        # tinh vi trong phần attention -- phần "đóng góp khoa học" của đồ án
        # nằm ở SO SÁNH 4 FUSION STRATEGY, không phải ở việc tự cài Transformer.) -----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, hidden_dim) -- khớp convention dùng xuyên suốt pipeline
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ----- Output projection -----
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)
        nn.init.zeros_(self.output_proj.bias)  # bias KHÔNG bị tie -- zero-init cho sạch (thực hành chuẩn khi tie weight)
        if tie_weights:
            # Weight tying (Press & Wolf, 2017) -- dùng CHUNG ma trận trọng số
            # giữa token embedding và output projection.
            self.output_proj.weight = self.token_embedding.weight

    @property
    def device(self):
        return self.token_embedding.weight.device

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.token_embedding(token_ids) * self.embed_scale)

    # ============================================================
    # forward() — TRAIN MODE (teacher forcing)
    # ============================================================
    def forward(
        self,
        fused_features: torch.Tensor,   # (B, L, hidden_dim) -- từ Fusion Module
        fused_mask: torch.Tensor,        # (B, L) -- 1 = thật, 0 = padding
        caption_ids: torch.Tensor,       # (B, T) -- input ids (đã có BOS ở đầu)
        caption_mask: torch.Tensor,      # (B, T) -- 1 = token thật, 0 = padding
    ) -> torch.Tensor:
        """
        Trả về logits (B, T, vocab_size).

        Điểm khác biệt CỐT LÕI so với GPT-2 + prefix injection: fused_features
        không còn bị ghép vào ĐẦU sequence caption rồi để self-attention "tự
        suy luận" -- mà được truyền TRỰC TIẾP làm `memory` cho cross-attention,
        đúng cơ chế kiến trúc Transformer decoder nguyên bản.
        """
        tgt_embeddings = self.embed_dropout(self._embed_tokens(caption_ids))  # (B, T, hidden_dim)

        seq_len = caption_ids.size(1)
        # Causal mask (tam giác trên = True -- vị trí BỊ CHE) -- đảm bảo token
        # vị trí t chỉ nhìn được token 0..t-1, đúng ràng buộc autoregressive.
        #
        # FIX (phát hiện qua smoke test -- UserWarning "mismatched key_padding_mask
        # and attn_mask"): nn.Transformer.generate_square_subsequent_mask() trả
        # về mask dạng FLOAT (0.0 / -inf), trong khi tgt_key_padding_mask /
        # memory_key_padding_mask bên dưới là BOOL -- PyTorch cảnh báo 2 loại
        # mask khác dtype sẽ ngừng được hỗ trợ ở bản sau. Tự dựng causal mask
        # dạng BOOL (torch.triu) để đồng nhất dtype với padding mask, tránh
        # cảnh báo và tránh lỗi tiềm ẩn khi PyTorch nâng cấp.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device), diagonal=1
        )  # True ở tam giác trên (vị trí tương lai) -- đúng convention "True = CẦN CHE"

        # nn.TransformerDecoder quy ước *_key_padding_mask: True = vị trí CẦN CHE
        # (padding) -- giống hệt convention đã dùng trong fusion_module.py
        # (CrossAttentionBlock), nên đảo dấu tương tự ở đây.
        tgt_key_padding_mask = ~caption_mask.bool()
        memory_key_padding_mask = ~fused_mask.bool()

        decoder_output = self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=fused_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T, hidden_dim)

        logits = self.output_proj(decoder_output)  # (B, T, vocab_size)
        return logits

    def compute_loss(
        self,
        fused_features: torch.Tensor,
        fused_mask: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        GIỮ NGUYÊN quy ước next-token-prediction giống bản GPT-2 cũ: input =
        caption_ids[:, :-1], target = caption_ids[:, 1:], loss chỉ tính trên
        vị trí có target_mask = 1 (loại bỏ padding).
        """
        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]
        target_mask = caption_mask[:, 1:]

        logits = self.forward(fused_features, fused_mask, input_ids, input_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        )
        loss = loss.reshape(target_ids.shape)
        loss = (loss * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        return loss

    # ============================================================
    # encode_captions() — tokenize text -> ids (dùng vocab riêng, KHÔNG BPE)
    # ============================================================
    def encode_captions(self, caption_texts: List[str], max_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize + thêm BOS đầu, EOS cuối, pad về cùng độ dài trong batch.
        Từ không có trong vocab (OOV so với train2017) -> map thành <unk>.
        """
        batch_ids = []
        for text in caption_texts:
            words = _tokenize_text(text)[: max_length - 2]  # chừa chỗ cho BOS/EOS
            ids = [self.bos_id] + [self.word2idx.get(w, self.unk_id) for w in words] + [self.eos_id]
            batch_ids.append(ids)

        seq_len = min(max(len(ids) for ids in batch_ids), max_length)
        caption_ids = torch.full((len(batch_ids), seq_len), self.pad_id, dtype=torch.long)
        caption_mask = torch.zeros((len(batch_ids), seq_len), dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            ids = ids[:seq_len]
            caption_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            caption_mask[i, : len(ids)] = 1

        return caption_ids, caption_mask

    # ============================================================
    # generate() — INFERENCE MODE (greedy hoặc beam search)
    # ============================================================
    @torch.no_grad()
    def generate(
        self,
        fused_features: torch.Tensor,
        fused_mask: torch.Tensor,
        max_length: int = 30,
        method: str = "greedy",
        num_beams: int = 4,
        length_penalty: float = 1.0,
        **_ignored_kwargs,  # nuốt các tham số cũ (repetition_penalty, no_repeat_ngram_size)
                             # của bản GPT-2 để evaluate.py/evaluate_flickr30k.py/
                             # inference_service.py KHÔNG cần sửa lời gọi generate()
    ) -> List[str]:
        """
        LƯU Ý: decoder tự huấn luyện (không có pretrained language prior) ít
        gặp hiện tượng "vòng lặp hấp dẫn" (attractor loop) như GPT-2 khi gặp
        ảnh ngoài phân phối, vì bản thân model không có xu hướng ngôn ngữ học
        được từ dữ liệu ngoài phạm vi caption -- nên KHÔNG cần cơ chế
        repetition_penalty / no_repeat_ngram_size phức tạp như bản cũ. Tham
        số này được "nuốt" (ignore) qua **_ignored_kwargs để giữ tương thích
        ngược với code gọi generate() đã có sẵn.
        """
        if method == "beam":
            return self._generate_beam(fused_features, fused_mask, max_length, num_beams, length_penalty)

        self.eval()
        batch_size = fused_features.size(0)
        device = fused_features.device

        generated = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Tính lại toàn bộ sequence mỗi bước (không dùng KV-cache) -- đơn
            # giản, đủ nhanh với caption ngắn (<=30 token), ưu tiên tính đúng
            # đắn/dễ debug hơn tối ưu tốc độ (đúng tinh thần code hiện tại
            # của dự án -- xem comment tương tự trong RGCNEncoder.forward_batch()).
            caption_mask_full = torch.ones_like(generated, dtype=torch.long)
            logits = self.forward(fused_features, fused_mask, generated, caption_mask_full)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, self.pad_id), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == self.eos_id)
            if finished.all():
                break

        return [self._ids_to_text(generated[i, 1:].tolist()) for i in range(batch_size)]

    @torch.no_grad()
    def _generate_beam(
        self,
        fused_features: torch.Tensor,
        fused_mask: torch.Tensor,
        max_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """Beam Search, hỗ trợ batch_size=1 (đúng use-case web demo -- giống
        thiết kế _generate_beam() trong caption_decoder.py bản GPT-2 cũ)."""
        device = fused_features.device
        assert fused_features.size(0) == 1, "Beam search hiện chỉ hỗ trợ batch_size=1."

        beams = [(torch.tensor([[self.bos_id]], device=device), 0.0, False)]

        for _ in range(max_length):
            candidates = []
            for token_ids, log_prob, finished in beams:
                if finished:
                    candidates.append((token_ids, log_prob, True))
                    continue

                mask_full = torch.ones_like(token_ids, dtype=torch.long)
                logits = self.forward(fused_features, fused_mask, token_ids, mask_full)
                next_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(num_beams, dim=-1)

                for k in range(num_beams):
                    new_token = topk_ids[:, k : k + 1]
                    new_log_prob = log_prob + topk_log_probs[0, k].item()
                    new_ids = torch.cat([token_ids, new_token], dim=1)
                    is_finished = new_token.item() == self.eos_id
                    candidates.append((new_ids, new_log_prob, is_finished))

            def _score(cand):
                ids, lp, _ = cand
                return lp / (ids.size(1) ** length_penalty)

            candidates.sort(key=_score, reverse=True)
            beams = candidates[:num_beams]

            if all(f for _, _, f in beams):
                break

        best_ids, _, _ = max(beams, key=lambda c: c[1] / (c[0].size(1) ** length_penalty))
        return [self._ids_to_text(best_ids[0, 1:].tolist())]

    def _ids_to_text(self, ids: List[int]) -> str:
        if self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        words = [self.idx2word[t] for t in ids if t != self.pad_id]
        return " ".join(words)


# ============================================================
# HƯỚNG DẪN MIGRATE — các file cần sửa (chỉ vài dòng, KHÔNG đổi logic khác)
# ============================================================
"""
1) train.py — trong class ImageCaptioningModel.__init__():

   TRƯỚC:
       from caption_decoder import CaptionDecoder
       ...
       self.decoder = CaptionDecoder()

   SAU:
       from transformer_caption_decoder import CaptionDecoderTransformer
       ...
       CAPTION_VOCAB_PATH = os.path.join(PROJECT_ROOT, "features", "caption_vocab.pt")
       ...
       self.decoder = CaptionDecoderTransformer(CAPTION_VOCAB_PATH)

   (Thêm dòng CAPTION_VOCAB_PATH vào phần CONFIG đầu file train.py, cạnh
   GLOVE_VOCAB_PATH đã có sẵn.)

2) evaluate.py, evaluate_flickr30k.py, inference_service.py:
   KHÔNG cần sửa gì thêm -- các file này chỉ gọi model.decoder.generate(...)
   thông qua ImageCaptioningModel đã import từ train.py, nên tự động dùng
   decoder mới. Tham số repetition_penalty/no_repeat_ngram_size cũ (nếu còn
   sót trong lời gọi generate() ở evaluate.py bản đã sửa PTBTokenizer) sẽ bị
   "nuốt" an toàn qua **_ignored_kwargs, không gây lỗi.

3) BẮT BUỘC chạy trước khi train:
       python build_caption_vocab.py

4) QUAN TRỌNG NHẤT: vì kiến trúc decoder đổi hoàn toàn (không còn tương
   thích với checkpoint GPT-2 cũ), phải TRAIN LẠI CẢ 4 STRATEGY TỪ ĐẦU:
       python train.py --strategy baseline
       python train.py --strategy concat
       python train.py --strategy one_directional
       python train.py --strategy bidirectional
   Khuyến nghị: chạy estimate_training_time.py (đã có sẵn trong tests/)
   trước với decoder mới để ước lượng lại thời gian train (nhiều khả năng
   NHANH HƠN bản GPT-2 cũ vì decoder mới ít tham số hơn nhiều: ~15-25M so
   với 124M của GPT-2 fine-tune toàn bộ).
"""