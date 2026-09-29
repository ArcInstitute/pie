"""Prediction readout: tempered DE probability, LFC and bin-centre delta-p."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pie.data.delta_p import DeltaPGrid, dequantize
from pie.model.pie import GroupOutput, ModelConfig


@dataclass
class Readout:
    p_de: Tensor  # (b, G) float32 = nan_to_num(1 - softmax(de_logits / T_de)[..., 0])
    lfc: Tensor  # (b, G) float32
    delta_p: Tensor  # (b, G) float32 = dequantize(argmax(dp_logits / T_dp))


def temper(out: GroupOutput, cfg: ModelConfig) -> GroupOutput:
    """Divide DE logits by T_de and delta-p logits by T_dp; LFC is untouched."""
    return GroupOutput(
        rows=out.rows,
        de_logits=out.de_logits / cfg.temperature.de,
        lfc=out.lfc,
        dp_logits=out.dp_logits / cfg.temperature.delta_p,
    )


def readout(out: GroupOutput, cfg: ModelConfig, grid: DeltaPGrid) -> Readout:
    tempered = temper(out, cfg)
    probs = tempered.de_logits.softmax(dim=-1)
    p_de = torch.nan_to_num(1.0 - probs[..., 0], nan=0.0).float()
    delta_p = dequantize(tempered.dp_logits.argmax(dim=-1), grid).float()
    return Readout(p_de=p_de, lfc=tempered.lfc.float(), delta_p=delta_p)
