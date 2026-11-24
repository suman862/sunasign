from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FrameEncoder(nn.Module):
    """Small trainable RGB encoder. Replace/scale for research experiments."""
    def __init__(self, image_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 4, 3), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 192, 3, 2, 1), nn.GELU(),
            nn.Conv2d(192, image_dim, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(image_dim)

    def forward(self, x):
        # x: [B,T,3,H,W]
        b,t,c,h,w = x.shape
        y = self.net(x.reshape(b*t,c,h,w)).flatten(1)
        return self.norm(y).reshape(b,t,-1)


class LandmarkEncoder(nn.Module):
    def __init__(self, landmark_dim, d_model):
        super().__init__()
        self.proj = MLP(landmark_dim, d_model)

    def forward(self, x):
        return self.proj(x)


class CrossModalFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn_v2l = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_l2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_l = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )

    def forward(self, visual, landmark, key_padding_mask=None):
        v, _ = self.attn_v2l(
            visual, landmark, landmark,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        l, _ = self.attn_l2v(
            landmark, visual, visual,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.norm_v(visual + v) + self.norm_l(landmark + l)
        return fused + self.ff(fused)


class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead, layers, ff_dim, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        return self.norm(self.encoder(x, src_key_padding_mask=key_padding_mask))


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, layers, ff_dim, dropout, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, tokens, memory, memory_padding_mask=None):
        t = tokens.size(1)
        if t > self.pos.size(1):
            raise ValueError("text sequence exceeds configured max_text_len")
        x = self.emb(tokens) + self.pos[:, :t]
        causal = torch.triu(
            torch.ones(t, t, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        x = self.dec(
            x, memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.head(self.norm(x))


@dataclass
class ModelOutput:
    sign_logits: torch.Tensor
    boundary_logits: torch.Tensor
    text_logits: torch.Tensor


class SunaSign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        image_dim = cfg["image_dim"]
        self.frame = FrameEncoder(image_dim)
        self.image_proj = nn.Linear(image_dim, d)
        self.landmarks = LandmarkEncoder(cfg["landmark_dim"], d)
        self.fusion = CrossModalFusion(d, cfg["nhead"], cfg["dropout"])
        self.temporal = TemporalEncoder(
            d, cfg["nhead"], cfg["layers"], cfg["ff_dim"], cfg["dropout"]
        )
        self.sign_head = nn.Linear(d, cfg["num_sign_classes"])
        self.boundary_head = nn.Linear(d, 1)
        self.text = TextDecoder(
            cfg["vocab_size"], d, cfg["nhead"],
            max(2, cfg["layers"] // 2), cfg["ff_dim"],
            cfg["dropout"], cfg["max_text_len"]
        )

    def encode(self, frames, landmarks, padding_mask=None):
        v = self.image_proj(self.frame(frames))
        l = self.landmarks(landmarks)
        x = self.fusion(v, l, key_padding_mask=padding_mask)
        return self.temporal(x, key_padding_mask=padding_mask)

    def forward(self, frames, landmarks, text_input_ids, padding_mask=None):
        z = self.encode(frames, landmarks, padding_mask)
        return ModelOutput(
            sign_logits=self.sign_head(z),
            boundary_logits=self.boundary_head(z).squeeze(-1),
            text_logits=self.text(text_input_ids, z, memory_padding_mask=padding_mask),
        )

    @torch.no_grad()
    def generate_greedy(self, frames, landmarks, bos_id, eos_id, pad_id,
                        max_len, padding_mask=None):
        z = self.encode(frames, landmarks, padding_mask)
        b = frames.size(0)
        tokens = torch.full((b, 1), bos_id, device=frames.device, dtype=torch.long)
        finished = torch.zeros(b, dtype=torch.bool, device=frames.device)
        for _ in range(max_len - 1):
            logits = self.text(tokens, z, memory_padding_mask=padding_mask)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            finished |= nxt.eq(eos_id)
            if finished.all():
                break
        return tokens
