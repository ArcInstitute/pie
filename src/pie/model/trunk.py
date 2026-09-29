"""Perceiver building blocks: qk-normed attention, feed-forward and a pre-norm block."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pie.model.layers import RMSNorm


class MultiHeadAttention(nn.Module):
    """Bias-free multi-head attention over SDPA.

    Q and K get a per-head RMSNorm, and a learnable per-head scale (initialised to
    1/sqrt(head_dim)) replaces the fixed softmax scale.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_scale = nn.Parameter(torch.full((num_heads,), 1.0 / math.sqrt(self.head_dim)))

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        """q (B, Sq, D), kv (B, Skv, D), kv_mask (B, Skv) with True = attend."""
        b, sq, d = q.shape
        skv = kv.shape[1]
        q_heads = self.q_proj(q).view(b, sq, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = self.k_proj(kv).view(b, skv, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = self.v_proj(kv).view(b, skv, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = kv_mask[:, None, None, :] if kv_mask is not None else None
        q_heads = self.q_norm(q_heads)
        k_heads = self.k_norm(k_heads)
        q_heads = q_heads * self.attn_scale.view(1, -1, 1, 1)
        out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=1.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, sq, d)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Linear -> GELU -> Linear -> Dropout, bias-free."""

    def __init__(self, d_model: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * ff_mult
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: RMSNorm -> attention -> residual -> RMSNorm -> FFN -> residual.

    `is_cross=True` attends from `x` to an external `kv` (normed by `norm_kv`); otherwise
    `x` attends to itself.
    """

    def __init__(
        self, d_model: int, num_heads: int, ff_mult: int, dropout: float, is_cross: bool
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.norm_attn = RMSNorm(d_model)
        if is_cross:
            self.norm_kv = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_ff = RMSNorm(d_model)
        self.ff = FeedForward(d_model, ff_mult, dropout)

    def forward(
        self, x: Tensor, kv: Tensor | None = None, kv_mask: Tensor | None = None
    ) -> Tensor:
        q_normed = self.norm_attn(x)
        if self.is_cross:
            if kv is None:
                raise ValueError("kv must be provided for cross-attention blocks")
            x = x + self.attn(q_normed, self.norm_kv(kv), kv_mask)
        else:
            x = x + self.attn(q_normed, q_normed, kv_mask)
        return x + self.ff(self.norm_ff(x))
