"""Training objective, summed with weight 1 per term.

- de_loss: 2-class cross-entropy over tested genes, class-weighted per row (cap).
- lfc_loss: Huber on log2 fold change over DE, finite, positive genes, with the
  perturbation's own gene split off and scaled by alpha.
- lfc_dir_loss: hinge on the sign of the predicted LFC, same genes with a nonzero target.
- dp_loss: two-hot cross-entropy on delta-p bins, soft class-weighted per row (cap).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from pie.data.dataset import Batch
from pie.data.delta_p import DeltaPGrid, two_hot
from pie.model.heads import temper
from pie.model.pie import GroupOutput, ModelConfig

LOSS_KEYS: tuple[str, ...] = ("loss", "de_loss", "lfc_loss", "lfc_dir_loss", "dp_loss")


def de_loss(logits: Tensor, targets: Tensor, valid: Tensor, cap: float) -> Tensor:
    """Mean CE over `valid` cells; per-row weights n_valid / (C * count_c), capped at `cap`."""
    b, g, c = logits.shape
    loss = F.cross_entropy(logits.view(-1, c), targets.view(-1), reduction="none").view(b, g)
    keep = valid.to(loss.dtype)
    class_counts = torch.zeros(b, c, device=targets.device, dtype=loss.dtype)
    class_counts.scatter_add_(1, targets, keep)
    n_kept = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
    inv_freq = (n_kept / (c * class_counts.clamp(min=1.0))).clamp(max=cap)
    loss = loss * inv_freq.gather(1, targets)
    return (loss * keep).sum() / keep.sum().clamp(min=1.0)


def lfc_targets(fold_changes: Tensor, de_mask: Tensor) -> tuple[Tensor, Tensor]:
    """log2 of linear fold changes on DE & finite & fc > 0 cells; other cells are NaN (unused)."""
    mask = de_mask & torch.isfinite(fold_changes) & (fold_changes > 0)
    lfc = torch.full_like(fold_changes, float("nan"))
    lfc[mask] = torch.log2(fold_changes[mask].clamp(min=1e-8))
    return lfc, mask


def _huber_over(pred: Tensor, target: Tensor, cells: Tensor, delta: float) -> Tensor:
    if not bool(cells.any()):
        return pred.new_zeros(())
    return F.huber_loss(pred[cells], target[cells], reduction="mean", delta=delta)


def lfc_loss(
    pred: Tensor, lfc: Tensor, mask: Tensor, target_gene: Tensor, alpha: float, delta: float
) -> Tensor:
    """Huber mean over non-target cells + alpha * Huber mean over target cells.

    `target_gene` (b,) is the row's own gene on the local axis, -1 if none. Without any
    target in the rows this is the plain Huber mean over `mask`.
    """
    has_target = target_gene >= 0
    if not bool(has_target.any()):
        return _huber_over(pred, lfc, mask, delta)
    rows = torch.nonzero(has_target, as_tuple=True)[0]
    target_cell = torch.zeros_like(mask)
    target_cell[rows, target_gene[rows]] = True
    return _huber_over(pred, lfc, mask & ~target_cell, delta) + alpha * _huber_over(
        pred, lfc, mask & target_cell, delta
    )


def lfc_direction_loss(pred: Tensor, lfc: Tensor, mask: Tensor, tau: float) -> Tensor:
    """Mean of relu(-sign(y) * yhat / tau) over mask & (y != 0); 0 (with a graph) when empty."""
    m = mask & (lfc != 0)
    p = torch.where(m, pred, 0.0)
    t = torch.where(m, lfc, 0.0)
    if p.dtype in (torch.float16, torch.bfloat16):
        p = p.float()
    signed_error = -torch.sign(t) * p / tau
    return torch.where(m, F.relu(signed_error), 0.0).sum() / m.sum().clamp_min(1)


def dp_loss(logits: Tensor, target_probs: Tensor, valid: Tensor, cap: float) -> Tensor:
    """Soft-target CE; per-row weights n_valid / (K * soft_count_k), capped, target-mixed."""
    k = logits.shape[-1]
    loss = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    keep = valid.to(loss.dtype)
    class_counts = torch.einsum("bgk,bg->bk", target_probs, keep)
    n_kept = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
    inv_freq = (n_kept / (k * class_counts.clamp(min=1e-6))).clamp(max=cap)
    loss = loss * torch.einsum("bgk,bk->bg", target_probs, inv_freq)
    return (loss * keep).sum() / keep.sum().clamp(min=1.0)


def compute_losses(
    outputs: Sequence[GroupOutput],
    batch: Batch,
    cfg: ModelConfig,
    grid: DeltaPGrid,
    temperatures: bool,
) -> dict[str, Tensor]:
    """Row-weighted (b / B) sum over groups of the four terms; `loss` is their sum.

    `temperatures=True` (validation) scores DE logits / T_de and delta-p logits / T_dp.
    """
    if len(outputs) != len(batch.groups):
        raise ValueError(f"{len(outputs)} outputs for {len(batch.groups)} gene groups")
    device = outputs[0].de_logits.device if outputs else batch.target_gene.device
    bs = int(batch.dataset_ids.shape[0])
    zero = torch.zeros((), device=device)
    de_total, lfc_total, dir_total, dp_total = zero, zero, zero, zero
    target_gene = batch.target_gene.to(device)
    for out, group in zip(outputs, batch.groups, strict=True):
        if (
            group.fold_changes is None
            or group.de_mask is None
            or group.tested is None
            or group.delta_p is None
        ):
            raise ValueError(f"gene group of dir {group.dir_index} carries no labels")
        heads = temper(out, cfg) if temperatures else out
        rows = group.rows.to(device)
        w = rows.numel() / bs
        de_mask = group.de_mask.to(device)
        lfc, lfc_mask = lfc_targets(group.fold_changes.to(device), de_mask)
        de_term = de_loss(
            heads.de_logits, de_mask.long(), group.tested.to(device), cfg.class_weight_cap
        )
        lfc_term = lfc_loss(
            heads.lfc,
            lfc,
            lfc_mask,
            target_gene[rows],
            cfg.lfc_target_gene_alpha,
            cfg.lfc_huber_delta,
        )
        dir_term = lfc_direction_loss(heads.lfc, lfc, lfc_mask, cfg.lfc_direction_temperature)
        de_total = de_total + w * de_term
        lfc_total = lfc_total + w * lfc_term
        dir_total = dir_total + w * dir_term
        probs, valid = two_hot(group.delta_p.to(device), grid)
        dp_total = dp_total + w * dp_loss(
            heads.dp_logits, probs, valid, cfg.class_weight_cap_delta_p
        )
    return {
        "loss": de_total + lfc_total + dir_total + dp_total,
        "de_loss": de_total,
        "lfc_loss": lfc_total,
        "lfc_dir_loss": dir_total,
        "dp_loss": dp_total,
    }
