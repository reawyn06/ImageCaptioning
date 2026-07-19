"""
caption_decoder.py
======================
Mục đích:
    Module sinh caption (Caption Decoder), dùng GPT-2 base pretrained làm
    backbone, kết hợp kỹ thuật PREFIX INJECTION (kiểu ClipCap, Mokady et al.
    2021) để đưa fused_features (output của Fusion Module) vào GPT-2 mà
    KHÔNG cần thêm cross-attention layer (vì GPT-2 gốc là decoder-only,
    không có sẵn cross-attention).

Thiết kế đã chốt:
    - Backbone: GPT-2 base, nguyên 12 lớp, 768 hidden dim, fine-tune toàn bộ
      (không freeze).
    - Tokenizer: GPT-2 BPE (~50,257 vocab). Thêm pad_token = eos_token (GPT-2
      gốc không có pad token sẵn).
    - Cách inject feature: Mapping Network (Linear/MLP) nhận TOÀN BỘ
      fused_features (B, L, 768), L thay đổi theo ảnh/strategy (196 cho
      Baseline/Concat, 196+N cho One-directional/Bidirectional), sinh ra
      prefix embeddings CÙNG ĐỘ DÀI L (không cố định k token như ClipCap gốc
      -- quyết định này giữ được thông tin chi tiết per-patch/per-object,
      đồng thời phản ánh đúng lượng thông tin khác nhau giữa 4 strategy).
    - Train: Teacher forcing -- ghép [prefix_embeddings, caption_token_embeddings]
      thành 1 sequence, cho GPT-2 dự đoán next-token, chỉ tính loss trên phần
      caption (không tính loss trên phần prefix, vì prefix không phải token
      thật, không có "next token" đúng nghĩa để dự đoán).

Cấu trúc input/output:
    forward() [TRAIN MODE - teacher forcing]:
        fused_features: (B, L, 768)        -- từ Fusion Module
        fused_mask:     (B, L)             -- mask cho phần fused (attention mask)
        caption_ids:    (B, T)             -- token id của caption (đã gồm BOS,
                                               chưa gồm EOS ở vị trí cuối cùng làm
                                               input; nhãn đúng lệch 1 vị trí)
        caption_mask:   (B, T)             -- 1 = token thật, 0 = padding caption

        Returns: logits (B, T, vocab_size) -- chỉ phần ứng với caption tokens

    generate() [INFERENCE MODE - autoregressive]:
        fused_features, fused_mask -- giống trên
        Sinh caption từng token một (greedy), KHÔNG có ground-truth caption_ids.

CẬP NHẬT (fix lỗi lặp từ khi demo với ảnh ngoài dataset -- vd:
    "a black and white photo of a black and white photo of a black and white
    bird"): generate() gốc dùng argmax thuần túy, không có bất kỳ cơ chế
    chống lặp nào -- đây là nguyên nhân trực tiếp khiến greedy decoding rơi
    vào "vòng lặp hấp dẫn" (attractor loop) khi model không chắc chắn về nội
    dung ảnh (vd ảnh ngoài phân phối train). Đã thêm repetition_penalty
    (phạt token đã xuất hiện, theo công thức CTRL -- Keskar et al. 2019) và
    no_repeat_ngram_size (chặn cứng việc lặp lại y hệt 1 cụm n-gram đã sinh).

    QUAN TRỌNG: thay đổi này CHỈ nằm trong generate() (inference-only) --
    forward()/compute_loss() (dùng khi train) giữ NGUYÊN VẸN 100%, không
    ảnh hưởng đến 4 checkpoint đã train xong và bảng kết quả BLEU/CIDEr/SPICE
    đã có.

Lưu ý quan trọng về tên gọi (cần ghi rõ trong báo cáo):
    Đây KHÔNG phải Transformer Decoder tự xây theo kiến trúc gốc Vaswani 2017
    (không có cross-attention layer riêng). Đây là GPT-2 (decoder-only,
    chỉ có self-attention) + Mapping Network, theo kiến trúc ClipCap
    (Mokady et al., 2021) -- visual/semantic information được "tiêm" vào
    qua prefix embeddings ở đầu sequence, GPT-2 dùng chính self-attention
    (masked) để "nhìn" ngược lại phần prefix này khi sinh từng token caption.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


HIDDEN_DIM = 768
GPT2_MODEL_NAME = "gpt2"  # GPT-2 base (124M tham số)


# ============================================================
# BƯỚC 1 — Mapping Network (chuyển fused_features -> prefix embeddings)
# ============================================================
class MappingNetwork(nn.Module):
    """
    Nhận fused_features (B, L, 768) từ Fusion Module, chuyển thành prefix
    embeddings (B, L, 768) để ghép vào đầu sequence input của GPT-2.

    Vì fused_features đã có sẵn hidden_dim=768 (khớp với GPT-2 n_embd), về lý
    thuyết có thể dùng identity mapping. Nhưng vẫn cần 1 MLP nhỏ để:
        - Cho phép model HỌC cách "dịch" không gian đặc trưng visual/semantic
          sang không gian mà GPT-2 hiểu được (2 không gian này được train
          hoàn toàn độc lập trước đó -- ViT, GloVe/R-GCN, GPT-2 -- không có
          lý do gì để chúng tự khớp nhau nếu không qua 1 lớp học được).
        - Đồng nhất thiết kế với ClipCap gốc (luôn có mapping network, dù
          input/output dimension giống nhau).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, mlp_hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.net(fused_features) + fused_features)  # residual


# ============================================================
# HÀM TIỆN ÍCH — Repetition control cho generate() (MỚI)
# ============================================================
def _apply_repetition_penalty(
    logits: torch.Tensor,           # (B, vocab_size)
    generated_ids: torch.Tensor,    # (B, t) -- toàn bộ token đã sinh tới hiện tại (gồm cả BOS)
    penalty: float = 1.3,
) -> torch.Tensor:
    """
    Phạt các token ĐÃ XUẤT HIỆN trong sequence, giảm khả năng chúng được chọn
    lại ở bước tiếp theo -- theo đúng công thức chuẩn (CTRL, Keskar et al.
    2019; cũng là cách HuggingFace generate() implement tham số
    repetition_penalty):
        Nếu logit > 0: chia cho penalty (giảm điểm)
        Nếu logit < 0: nhân với penalty (giảm điểm càng thêm âm)
    Đây là bước XỬ LÝ TRƯỚC khi argmax, không ảnh hưởng đến forward()/loss.
    """
    for b in range(logits.size(0)):
        seen_tokens = torch.unique(generated_ids[b])
        seen_logits = logits[b, seen_tokens]
        logits[b, seen_tokens] = torch.where(
            seen_logits > 0, seen_logits / penalty, seen_logits * penalty
        )
    return logits


def _block_repeated_ngrams(
    logits: torch.Tensor,           # (B, vocab_size) -- SẼ BỊ SỬA TRỰC TIẾP (in-place)
    generated_ids: torch.Tensor,    # (B, t)
    ngram_size: int,
) -> torch.Tensor:
    """
    Chặn CỨNG việc sinh ra 1 n-gram đã từng xuất hiện y hệt trước đó trong
    cùng 1 sequence (kỹ thuật no_repeat_ngram_size chuẩn của HuggingFace
    generate()). Ví dụ với ngram_size=3: nếu cụm "black and white" đã xuất
    hiện, và 2 token gần nhất vừa sinh là "black and", token tiếp theo
    "white" sẽ bị cấm hoàn toàn (gán logit = -inf).

    Đây là cơ chế MẠNH HƠN repetition_penalty (chặn tuyệt đối thay vì chỉ
    giảm xác suất) -- dùng kết hợp cả 2 để xử lý triệt để hiện tượng lặp
    từng thấy trong caption demo (vd "a black and white photo of a black
    and white photo of a black and white bird").
    """
    batch_size = generated_ids.size(0)
    seq_len = generated_ids.size(1)

    if seq_len < ngram_size:
        return logits  # chưa đủ dài để có n-gram hoàn chỉnh nào

    for b in range(batch_size):
        seq = generated_ids[b].tolist()
        ngram_prefix = tuple(seq[-(ngram_size - 1):])  # (n-1) token gần nhất

        banned_tokens = set()
        for start in range(len(seq) - ngram_size + 1):
            if tuple(seq[start:start + ngram_size - 1]) == ngram_prefix:
                banned_tokens.add(seq[start + ngram_size - 1])

        for tok in banned_tokens:
            logits[b, tok] = -float("inf")

    return logits


# ============================================================
# BƯỚC 2 — Caption Decoder (GPT-2 + Mapping Network, kiểu ClipCap)
# ============================================================
class CaptionDecoder(nn.Module):
    """
    Module sinh caption đầy đủ: Mapping Network + GPT-2 base (fine-tune toàn bộ).

    Cách ghép sequence khi train (teacher forcing):
        input_embeds = concat([prefix_embeddings (L token), caption_embeddings (T token)], dim=1)
        attention_mask = concat([fused_mask (L), caption_mask (T)], dim=1)
        -> đưa qua GPT-2, lấy logits ở các vị trí TỪ VỊ TRÍ L TRỞ ĐI (ứng với
           caption tokens) để tính loss next-token-prediction.

    GPT-2 tự có causal mask (masked self-attention) nội bộ, nên token caption
    ở vị trí t chỉ "nhìn" được token trước nó (kể cả toàn bộ prefix, vì prefix
    luôn đứng trước caption trong sequence) -- đúng theo teacher forcing.
    """

    def __init__(self, gpt2_model_name: str = GPT2_MODEL_NAME, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        # GPT-2 gốc không có pad token -- set pad_token = eos_token theo thực
        # hành chuẩn khi dùng GPT-2 cho task cần padding (caption có độ dài khác nhau).
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        # KHÔNG freeze -- fine-tune toàn bộ GPT-2 theo quyết định đã chốt.

        self.mapping_network = MappingNetwork(hidden_dim)

        self.hidden_dim = hidden_dim
        self.vocab_size = self.gpt2.config.vocab_size

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lấy embedding của token ids qua chính embedding layer của GPT-2
        (wte = word token embeddings) -- đảm bảo dùng đúng pretrained embedding,
        và embedding này CŨNG được fine-tune cùng toàn bộ GPT-2."""
        return self.gpt2.transformer.wte(token_ids)

    def forward(
        self,
        fused_features: torch.Tensor,    # (B, L, hidden_dim)
        fused_mask: torch.Tensor,         # (B, L)
        caption_ids: torch.Tensor,        # (B, T) -- input ids (đã có BOS ở đầu)
        caption_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """
        Train mode (teacher forcing). Trả về logits (B, T, vocab_size) ứng
        với phần caption (KHÔNG bao gồm phần prefix).

        KHÔNG THAY ĐỔI so với bản gốc -- đây là logic dùng khi train, phải
        giữ nguyên để tương thích với 4 checkpoint đã lưu.
        """
        batch_size = fused_features.size(0)

        prefix_embeddings = self.mapping_network(fused_features)        # (B, L, hidden_dim)
        caption_embeddings = self._get_token_embeddings(caption_ids)     # (B, T, hidden_dim)

        input_embeds = torch.cat([prefix_embeddings, caption_embeddings], dim=1)  # (B, L+T, hidden_dim)
        attention_mask = torch.cat([fused_mask, caption_mask], dim=1)              # (B, L+T)

        outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L+T, vocab_size)

        prefix_len = fused_features.size(1)
        caption_logits = logits[:, prefix_len:, :]  # chỉ lấy phần ứng với caption

        return caption_logits

    def compute_loss(
        self,
        fused_features: torch.Tensor,
        fused_mask: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính loss next-token-prediction (cross-entropy), CHỈ trên phần caption.

        Quy ước: caption_ids đưa vào forward() là TOÀN BỘ caption (gồm BOS ở
        đầu, EOS ở cuối). Input cho GPT-2 là caption_ids[:, :-1] (bỏ token
        cuối), nhãn đúng là caption_ids[:, 1:] (bỏ token đầu) -- lệch nhau
        đúng 1 vị trí theo chuẩn next-token-prediction.

        KHÔNG THAY ĐỔI so với bản gốc.
        """
        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]
        target_mask = caption_mask[:, 1:]  # dùng để loại bỏ padding khỏi loss

        logits = self.forward(fused_features, fused_mask, input_ids, input_mask)  # (B, T-1, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        )
        loss = loss.reshape(target_ids.shape)               # (B, T-1)
        loss = (loss * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)

        return loss

    @torch.no_grad()
    def generate(
            self,
            fused_features: torch.Tensor,
            fused_mask: torch.Tensor,
            max_length: int = 30,
            method: str = "greedy",  # "greedy" hoặc "beam"
            repetition_penalty: float = 1.3,
            no_repeat_ngram_size: int = 3,
            num_beams: int = 4,  # MỚI -- chỉ dùng khi method="beam"
            length_penalty: float = 1.0,  # MỚI -- >1.0 khuyến khích câu dài hơn, <1.0 khuyến khích câu ngắn
    ) -> list:
        """
        ... (docstring cũ giữ nguyên, bổ sung) ...

        method="beam": dùng Beam Search thay vì greedy thuần túy -- giữ đồng
        thời num_beams chuỗi ứng viên tại mỗi bước, chọn chuỗi có tổng
        log-probability cao nhất (đã chuẩn hóa theo length_penalty) ở cuối cùng.
        Giúp tránh hiện tượng "kẹt vào nhánh xấu" đã quan sát được ở greedy
        (vd caption "arm." hoặc "s and a dog standing in the grass." của Concat).

        CHỈ ẢNH HƯỞNG INFERENCE -- không đụng đến forward()/compute_loss(),
        an toàn với 4 checkpoint đã train.
        """
        if method == "beam":
            return self._generate_beam(
                fused_features, fused_mask, max_length,
                num_beams=num_beams, length_penalty=length_penalty,
                repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # ---- Phần greedy giữ NGUYÊN như bản đã sửa trước đó ----
        self.eval()
        batch_size = fused_features.size(0)
        device = fused_features.device

        prefix_embeddings = self.mapping_network(fused_features)
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        generated_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            token_embeds = self._get_token_embeddings(generated_ids)
            input_embeds = torch.cat([prefix_embeddings, token_embeds], dim=1)
            token_mask = torch.ones_like(generated_ids, dtype=fused_mask.dtype)
            attn_mask = torch.cat([fused_mask, token_mask], dim=1)

            outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attn_mask)
            next_token_logits = outputs.logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                next_token_logits = _apply_repetition_penalty(next_token_logits, generated_ids,
                                                              penalty=repetition_penalty)
            if no_repeat_ngram_size > 0:
                next_token_logits = _block_repeated_ngrams(next_token_logits, generated_ids,
                                                           ngram_size=no_repeat_ngram_size)

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id), next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        captions = []
        for i in range(batch_size):
            ids = generated_ids[i, 1:].tolist()
            if eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            captions.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return captions

    @torch.no_grad()
    def _generate_beam(
            self,
            fused_features: torch.Tensor,
            fused_mask: torch.Tensor,
            max_length: int,
            num_beams: int,
            length_penalty: float,
            repetition_penalty: float,
            no_repeat_ngram_size: int,
    ) -> list:
        """
        Beam Search cho batch_size=1 (đủ dùng cho web demo -- mỗi request xử lý
        1 ảnh). Nếu cần hỗ trợ batch>1, cần mở rộng thêm logic gộp theo từng
        sample -- không cần thiết cho phạm vi hiện tại.
        """
        device = fused_features.device
        assert fused_features.size(0) == 1, "Beam search hiện chỉ hỗ trợ batch_size=1 (đúng use-case web demo)."

        prefix_embeddings = self.mapping_network(fused_features)  # (1, L, hidden_dim)
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Mỗi beam: (token_ids, cumulative_log_prob, is_finished)
        beams = [(torch.tensor([[bos_id]], device=device), 0.0, False)]

        for _ in range(max_length):
            candidates = []

            for token_ids, log_prob, finished in beams:
                if finished:
                    candidates.append((token_ids, log_prob, True))
                    continue

                token_embeds = self._get_token_embeddings(token_ids)
                input_embeds = torch.cat([prefix_embeddings, token_embeds], dim=1)
                token_mask = torch.ones_like(token_ids, dtype=fused_mask.dtype)
                attn_mask = torch.cat([fused_mask, token_mask], dim=1)

                outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attn_mask)
                next_token_logits = outputs.logits[:, -1, :].clone()

                if repetition_penalty != 1.0:
                    next_token_logits = _apply_repetition_penalty(next_token_logits, token_ids,
                                                                  penalty=repetition_penalty)
                if no_repeat_ngram_size > 0:
                    next_token_logits = _block_repeated_ngrams(next_token_logits, token_ids,
                                                               ngram_size=no_repeat_ngram_size)

                log_probs = F.log_softmax(next_token_logits, dim=-1)  # (1, vocab_size)
                topk_log_probs, topk_ids = log_probs.topk(num_beams, dim=-1)  # (1, num_beams)

                for k in range(num_beams):
                    new_token = topk_ids[:, k:k + 1]
                    new_log_prob = log_prob + topk_log_probs[0, k].item()
                    new_ids = torch.cat([token_ids, new_token], dim=1)
                    is_finished = (new_token.item() == eos_id)
                    candidates.append((new_ids, new_log_prob, is_finished))

            # Chọn num_beams ứng viên tốt nhất, CHUẨN HÓA theo length_penalty
            # (tránh thiên vị câu ngắn -- log_prob càng dài càng âm nhiều nếu
            # không chuẩn hóa, khiến beam search "thích" câu ngắn một cách giả tạo).
            def _score(cand):
                ids, log_prob, _ = cand
                length = ids.size(1)
                return log_prob / (length ** length_penalty)

            candidates.sort(key=_score, reverse=True)
            beams = candidates[:num_beams]

            if all(finished for _, _, finished in beams):
                break

        best_ids, _, _ = max(beams, key=lambda c: c[1] / (c[0].size(1) ** length_penalty))
        ids = best_ids[0, 1:].tolist()  # bỏ BOS
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]

        return [self.tokenizer.decode(ids, skip_special_tokens=True).strip()]

    def encode_captions(self, caption_texts: list, max_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hàm tiện ích: tokenize 1 batch caption text (string) thành caption_ids +
        caption_mask, đã thêm BOS ở đầu + EOS ở cuối + pad về cùng độ dài.

        GPT-2 tokenizer gốc không có bos_token riêng -- dùng eos_token làm cả
        BOS và EOS (thực hành phổ biến khi dùng GPT-2 cho task sinh có điều
        kiện, vì GPT-2 chỉ học 1 loại "boundary token" duy nhất).

        Args:
            caption_texts: list[str], vd ["a man riding a bike", ...]
        Returns:
            caption_ids: (B, T) LongTensor, đã có BOS đầu + EOS cuối + pad
            caption_mask: (B, T) LongTensor, 1 = token thật (gồm BOS/EOS), 0 = pad

        KHÔNG THAY ĐỔI so với bản gốc.
        """
        bos = self.tokenizer.eos_token  # GPT-2 dùng chung eos làm boundary token
        eos = self.tokenizer.eos_token

        texts_with_boundary = [f"{bos}{text}{eos}" for text in caption_texts]

        encoded = self.tokenizer(
            texts_with_boundary,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]