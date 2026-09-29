"""Per-gene evidence: modality encoders, fusion, zero-initialised injection adapters.

Every zero-initialised projection is created before the layers it follows, so the
init RNG stream matches the reference model.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from pie.data.evidence import CONTEXT_BLOCK_KEYS
from pie.model.layers import RMSNorm

N_CTRL_FEATURES = 3  # [control mean, log1p(100 * mean), mean == 0]
BLOCK_WIDTH = 4  # donor / context block columns: [value, spread, have, log1p count]


def encoder(n_in: int, width: int) -> nn.Sequential:
    """Linear -> SiLU -> Linear -> RMSNorm, one per evidence modality."""
    return nn.Sequential(
        nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), RMSNorm(width)
    )


def fuse_block(n_in: int, dim: int, dropout: float) -> nn.Sequential:
    """Fuse the concatenated modality encodings into one `dim`-wide evidence vector."""
    return nn.Sequential(
        RMSNorm(n_in),
        nn.Linear(n_in, 2 * dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(2 * dim, dim),
        RMSNorm(dim),
    )


def query_adapter(dim: int, d_model: int) -> nn.Sequential:
    """Evidence -> gene-query residual; its last Linear starts at zero."""
    out = nn.Linear(2 * dim, d_model, bias=False)
    nn.init.zeros_(out.weight)
    return nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), out)


def output_adapter(dim: int, d_model: int, dropout: float) -> nn.Sequential:
    """[evidence, decoded features] -> head-input residual; its last Linear starts at zero."""
    projection = nn.Linear(dim, d_model, bias=False)
    nn.init.zeros_(projection.weight)
    return nn.Sequential(
        RMSNorm(dim + d_model),
        nn.Linear(dim + d_model, dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        projection,
    )


def ctrl_query_features(ctrl_means: Tensor) -> Tensor:
    """(b, G) control means -> (b, G, 3) features; a row holding any NaN becomes all zeros."""
    z = ctrl_means.float()
    feats = torch.stack([z, torch.log1p(100.0 * z), (z == 0).to(z.dtype)], dim=-1)
    blank = torch.isnan(z).any(dim=-1)
    if bool(blank.any()):
        feats = torch.where(blank[:, None, None], torch.zeros_like(feats), feats)
    return feats


def drop_evidence(evidence: Mapping[str, Tensor], p: float, training: bool) -> dict[str, Tensor]:
    """Per-row dropout of whole evidence families during training.

    Draws `rand(b) >= p` once for the donor blocks, then once for the context blocks.
    Outside training, or with p <= 0, returns `evidence` unchanged.
    """
    if not training or p <= 0.0 or not evidence:
        return evidence  # type: ignore[return-value]
    first = next(iter(evidence.values()))
    b, device = first.shape[0], first.device
    keep = torch.rand(b, device=device) >= p
    keep_ctx = torch.rand(b, device=device) >= p
    return {
        key: value
        * (keep_ctx if key in CONTEXT_BLOCK_KEYS else keep).to(value.dtype)[:, None, None]
        for key, value in evidence.items()
    }
