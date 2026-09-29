"""Normalization layers. Every RMSNorm in the model calls the module-level `rms_norm`."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _RMSNormFn(torch.autograd.Function):
    """`F.rms_norm` that saves only (x, weight) and recomputes itself in backward.

    With a bf16 input and an fp32 weight (bf16 autocast), `F.rms_norm` runs a composite path that
    saves fp32 intermediates (4x the input); recomputing keeps outputs and gradients bitwise equal.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: Tensor, weight: Tensor, eps: float | None) -> Tensor:
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        return F.rms_norm(x, (x.shape[-1],), weight, eps)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, Tensor | None, None]:
        x, weight = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        with torch.enable_grad():
            xd = x.detach().requires_grad_(need_x)
            wd = weight.detach().requires_grad_(need_w)
            out = F.rms_norm(xd, (xd.shape[-1],), wd, ctx.eps)
            inputs = [t for t in (xd, wd) if t.requires_grad]
            grads = iter(torch.autograd.grad(out, inputs, grad))
        return (next(grads) if need_x else None, next(grads) if need_w else None, None)


def rms_norm(x: Tensor, weight: Tensor, eps: float | None) -> Tensor:
    """Root-mean-square normalization over the last dim, scaled by `weight`.

    `eps=None` uses `torch.finfo(x.dtype).eps`, as `nn.RMSNorm` does.
    """
    return _RMSNormFn.apply(x, weight, eps)


class RMSNorm(nn.RMSNorm):
    """`nn.RMSNorm` whose forward calls the module-level `rms_norm` (looked up at call time)."""

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)
