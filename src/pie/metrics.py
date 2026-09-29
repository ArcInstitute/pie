"""Evaluation metrics: one pure scorer shared by validation and evaluation.

Aggregation is row -> (context, perturbation) pair -> context -> dataset -> equal-weight
mean over the datasets that have rows. The edge-case policies are part of each definition:

- P(DE) is 0 where it is NaN and wherever the LFC prediction is NaN.
- A true DE gene is DE, tested and has a true LFC that is not NaN.
- binary_auprc: average precision of P(DE) over the tested genes; a row with no tested
  gene, no positive or only positives scores 0.0 and is kept.
- sig_jaccard: P(DE) >= 0.5 and tested vs the true DE genes; the target gene leaves both
  sets when the perturbation name is a gene on the axis; an empty union scores 1.0.
- direction_match: share of true DE genes whose predicted LFC is not NaN and has the sign
  of the true LFC; a row with no true DE gene scores 0.0 and is kept.
- spearman_lfc: Spearman (average ranks) over the true DE genes; fewer than 2 genes or
  constant ranks is NaN, and NaN rows are left out of the pair and context means.
- spearman_nsig: per context, Spearman across perturbations of the mean true DE count vs
  the mean predicted DE count, over perturbations with a true DE gene; none scores 1.0;
  fewer than 2 or constant ranks is NaN.
- discrimination_score_l1 (a port of cell-eval 0.6.8): per context, pseudobulk
  max(control + mean delta, 0) over rows with finite true delta; effect = bulk - control;
  genes named like the perturbation are left out; cityblock distance; rank of the own
  truth in np.argsort order; score 1 - rank / P. An all-NaN control skips the context; a
  partial or conflicting control, or a non-finite prediction on a kept row, is an error.

Validation reads `aggregate`. Evaluation writes `per_context` (DE metrics rounded to
float32 per context, L1 in float64), its `context_mean` row and `per_pair`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from pie.data.preprocessed import target_gene_index

log = logging.getLogger(__name__)

METRIC_KEYS: tuple[str, ...] = (
    "binary_auprc",
    "sig_jaccard",
    "direction_match",
    "spearman_nsig",
    "spearman_lfc",
    "discrimination_score_l1",
)
PAIR_METRIC_KEYS: tuple[str, ...] = (
    "binary_auprc",
    "sig_jaccard",
    "direction_match",
    "spearman_lfc",
    "discrimination_score_l1",
)
DE_PROB_THRESHOLD = 0.5

_ROW_KEYS: tuple[str, ...] = ("binary_auprc", "sig_jaccard", "direction_match", "spearman_lfc")
_ARRAYS: tuple[str, ...] = (
    "p_de",
    "lfc_pred",
    "delta_p_pred",
    "de_true",
    "tested",
    "lfc_true",
    "delta_p_true",
    "ctrl_means",
)
_L1 = "discrimination_score_l1"
_NAN = math.nan


@dataclass
class ScoringInputs:
    """Rows of one dataset on one gene axis (all arrays (R, G) unless noted).

    Rows are scored in the given order: pairs are averaged in row order and a context's
    pairs in first-occurrence order.
    """

    dataset: str
    genes: list[str]
    contexts: list[str]  # (R,)
    perts: list[str]  # (R,)
    p_de: np.ndarray  # float32
    lfc_pred: np.ndarray  # float32
    delta_p_pred: np.ndarray  # float32
    de_true: np.ndarray  # bool = (fdr < thr) & tested
    tested: np.ndarray  # bool
    lfc_true: np.ndarray  # float64
    delta_p_true: np.ndarray  # float32
    ctrl_means: np.ndarray  # (R, G) float32, the row's context control mean


@dataclass
class ScoreResult:
    per_context: pd.DataFrame  # columns: dataset, context, n_rows, *METRIC_KEYS
    per_pair: pd.DataFrame  # columns: dataset, context, perturbation, n_rows, *PAIR_METRIC_KEYS
    aggregate: dict[str, float]  # METRIC_KEYS -> equal-weight mean over datasets with rows
    context_mean: dict[str, float]  # METRIC_KEYS -> mean of the per_context rows, NaN skipped


@dataclass
class _Context:
    values: dict[str, float]  # METRIC_KEYS -> unrounded context value
    pairs: list[dict[str, object]]
    n_rows: int


def _f32(value: float) -> float:
    return float(np.float32(value))


def _mean_defined(values: Sequence[float]) -> float:
    """Sequential mean of the non-NaN values; NaN when there are none."""
    defined = [v for v in values if not math.isnan(v)]
    return sum(defined) / len(defined) if defined else _NAN


def _mean_finite(values: Sequence[float]) -> float:
    """np.mean of the non-NaN values; NaN when there are none."""
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else _NAN


def _dataset_mean(values: Sequence[float]) -> float:
    """float32 mean over datasets, skipping NaN datasets."""
    defined = [v for v in values if not math.isnan(v)]
    if not defined:
        return _NAN
    if len(defined) == 1:
        return defined[0]
    return float(torch.tensor(defined, dtype=torch.float32).mean())


def _de_probability(p_de: np.ndarray, lfc_pred: np.ndarray) -> np.ndarray:
    """P(DE) at 0 where it is NaN and where the LFC prediction is NaN."""
    prob = np.nan_to_num(np.asarray(p_de, dtype=np.float32), nan=0.0)
    prob[np.isnan(lfc_pred)] = 0.0
    return prob


def _binary_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary average precision, computed op for op as torchmetrics 1.9 does (float32)."""
    preds = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
    target = torch.from_numpy(labels.astype(np.int64))
    order = torch.argsort(preds, descending=True)
    preds, target = preds[order], target[order]
    distinct = torch.where(preds[1:] - preds[:-1])[0]
    ends = torch.nn.functional.pad(distinct, [0, 1], value=target.size(0) - 1)
    tps = torch.cumsum(target * 1.0, dim=0)[ends]
    fps = 1 + ends - tps
    precision = torch.cat([(tps / (tps + fps)).flip(0), torch.ones(1)])
    recall = torch.cat([(tps / tps[-1]).flip(0), torch.zeros(1)])
    precision = torch.where(torch.isnan(precision), torch.zeros_like(precision), precision)
    recall = torch.where(torch.isnan(recall), torch.zeros_like(recall), recall)
    return float(-torch.sum((recall[1:] - recall[:-1]) * precision[:-1]))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks; NaN values share the rank after every defined value."""
    nan = np.isnan(values)
    out = np.empty(values.shape, dtype=np.float64)
    n_defined = int((~nan).sum())
    if n_defined:
        out[~nan] = rankdata(values[~nan], method="average")
    if nan.any():
        out[nan] = (n_defined + 1 + values.size) / 2.0
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation; NaN for fewer than 2 values or constant ranks."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return _NAN
    x_rank, y_rank = _ranks(x), _ranks(y)
    if np.ptp(x_rank) == 0 or np.ptp(y_rank) == 0:
        return _NAN
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _row_values(
    prob: np.ndarray,
    lfc_pred: np.ndarray,
    true_de: np.ndarray,
    tested: np.ndarray,
    lfc_true: np.ndarray,
    target: int,
) -> tuple[float, float, float, float]:
    """(binary_auprc, sig_jaccard, direction_match, spearman_lfc) of one row."""
    labels = true_de[tested]
    auprc = 0.0
    if labels.any() and not labels.all():
        auprc = _binary_average_precision(prob[tested], labels)

    pred_sig = (prob >= DE_PROB_THRESHOLD) & tested
    true_sig = true_de.copy()
    if target >= 0:
        pred_sig[target] = False
        true_sig[target] = False
    union = int((pred_sig | true_sig).sum())
    inter = int((pred_sig & true_sig).sum())
    jaccard = float(np.float32(inter) / np.float32(union)) if union > 0 else 1.0

    n_de = int(true_de.sum())
    direction = 0.0
    if n_de:
        pred_lfc = lfc_pred[true_de]
        hits = ~np.isnan(pred_lfc) & (np.sign(lfc_true[true_de]) == np.sign(pred_lfc))
        direction = float(np.float32(int(hits.sum())) / np.float32(n_de))

    spearman = _spearman(lfc_true[true_de], lfc_pred[true_de])
    return auprc, jaccard, direction, spearman


def _spearman_nsig(
    prob: np.ndarray, tested: np.ndarray, true_de: np.ndarray, perts: list[str]
) -> float:
    """Context-level Spearman of per-perturbation mean true vs predicted DE counts."""
    real = torch.from_numpy(true_de.sum(axis=1).astype(np.float64))
    pred = torch.from_numpy(((prob >= DE_PROB_THRESHOLD) & tested).sum(axis=1).astype(np.float64))
    names = np.asarray(perts, dtype=object)
    means_real: list[float] = []
    means_pred: list[float] = []
    for pert in sorted(set(perts)):
        mask = torch.from_numpy(names == pert)
        means_real.append(float(real[mask].mean()))
        means_pred.append(float(pred[mask].mean()))
    real_mean = np.asarray(means_real, dtype=np.float64)
    pred_mean = np.asarray(means_pred, dtype=np.float64)
    keep = real_mean > 0
    if not keep.any():
        return 1.0
    return _f32(_spearman(real_mean[keep], pred_mean[keep]))


def discrimination_score_l1(
    real_bulk: np.ndarray,
    pred_bulk: np.ndarray,
    ctrl: np.ndarray,
    perts: Sequence[str],
    genes: Sequence[str],
) -> np.ndarray:
    """(P,) L1 discrimination score, a port of cell-eval 0.6.8.

    Rows of `real_bulk` / `pred_bulk` follow `perts`; pass them in sorted-name order to
    match the reference tie-breaking. effect = bulk - ctrl; genes named like the
    perturbation are left out; cityblock distance from the predicted effect to every real
    effect; rank of the own truth in default np.argsort order; score = 1 - rank / P.
    """
    control = np.asarray(ctrl, dtype=np.float64)
    real_effects = np.asarray(real_bulk, dtype=np.float64) - control
    pred_effects = np.asarray(pred_bulk, dtype=np.float64) - control
    gene_names = np.asarray(list(genes), dtype=object)
    n_perts = len(perts)
    scores = np.empty(n_perts, dtype=np.float64)
    for i, pert in enumerate(perts):
        keep = np.flatnonzero(gene_names != pert)
        distances = cdist(
            real_effects[:, keep], pred_effects[i, keep].reshape(1, -1), "cityblock"
        ).ravel()
        rank = int(np.flatnonzero(np.argsort(distances) == i)[0])
        scores[i] = 1 - rank / n_perts
    return scores


def _control_mean(ctrl_means: np.ndarray, context: str) -> np.ndarray | None:
    """The context's control mean; None (context skipped) when it is all NaN."""
    controls = np.asarray(ctrl_means, dtype=np.float64)
    if np.isnan(controls).all():
        log.warning(
            "context %r has an all-NaN control mean; discrimination_score_l1 is skipped",
            context,
        )
        return None
    if not np.isfinite(controls).all():
        raise ValueError(
            f"context {context!r} has a partially missing or non-finite control mean"
        )
    reference = controls[0]
    if not np.array_equal(controls, np.broadcast_to(reference, controls.shape)):
        raise ValueError(f"context {context!r} has conflicting control means")
    return reference


def _context_l1(
    inp: ScoringInputs, rows: np.ndarray, perts: list[str], context: str
) -> dict[str, float]:
    """Per-perturbation L1 discrimination of one context ({} when the context is skipped)."""
    control = _control_mean(np.asarray(inp.ctrl_means)[rows], context)
    if control is None:
        return {}
    pred = np.asarray(np.asarray(inp.delta_p_pred)[rows], dtype=np.float64)
    true = np.asarray(np.asarray(inp.delta_p_true)[rows], dtype=np.float64)
    names = np.asarray(perts, dtype=object)
    kept: list[str] = []
    pred_bulk: list[np.ndarray] = []
    real_bulk: list[np.ndarray] = []
    for pert in sorted(set(perts)):
        mask = names == pert
        pert_true, pert_pred = true[mask], pred[mask]
        finite = np.isfinite(pert_true).all(axis=1)
        if not finite.any():
            continue
        if not np.isfinite(pert_pred[finite]).all():
            raise ValueError(
                f"context {context!r}, perturbation {pert!r} has non-finite predictions"
            )
        pred_bulk.append(np.maximum(control + pert_pred[finite].mean(axis=0), 0.0))
        real_bulk.append(np.maximum(control + pert_true[finite].mean(axis=0), 0.0))
        kept.append(pert)
    if not kept:
        return {}
    scores = discrimination_score_l1(
        np.stack(real_bulk), np.stack(pred_bulk), control, kept, inp.genes
    )
    return dict(zip(kept, scores.tolist(), strict=True))


def _score_context(
    inp: ScoringInputs, rows: np.ndarray, context: str, gene_index: dict[str, int]
) -> _Context:
    perts = [inp.perts[i] for i in rows]
    tested = np.asarray(np.asarray(inp.tested)[rows], dtype=bool)
    lfc_true = np.asarray(np.asarray(inp.lfc_true)[rows], dtype=np.float64)
    lfc_pred = np.asarray(np.asarray(inp.lfc_pred)[rows], dtype=np.float32)
    prob = _de_probability(np.asarray(inp.p_de)[rows], lfc_pred)
    de_true = np.asarray(np.asarray(inp.de_true)[rows], dtype=bool)
    true_de = de_true & tested & ~np.isnan(lfc_true)

    by_pert: dict[str, list[tuple[float, float, float, float]]] = {}
    for j, pert in enumerate(perts):
        target = target_gene_index(pert, gene_index)
        by_pert.setdefault(pert, []).append(
            _row_values(prob[j], lfc_pred[j], true_de[j], tested[j], lfc_true[j], target)
        )

    values: dict[str, float] = {}
    pair_values: dict[str, dict[str, float]] = {pert: {} for pert in by_pert}
    for k, key in enumerate(_ROW_KEYS):
        means: list[float] = []
        for pert, row_values in by_pert.items():
            kept = [
                v[k] for v in row_values if key != "spearman_lfc" or not math.isnan(v[k])
            ]
            mean = sum(kept) / len(kept) if kept else _NAN
            pair_values[pert][key] = mean
            if kept:
                means.append(mean)
        values[key] = sum(means) / len(means) if means else _NAN
    values["spearman_nsig"] = _spearman_nsig(prob, tested, true_de, perts)
    l1 = _context_l1(inp, rows, perts, context)
    values[_L1] = _mean_finite(list(l1.values())) if l1 else _NAN

    pairs: list[dict[str, object]] = [
        {
            "context": context,
            "perturbation": pert,
            "n_rows": len(by_pert[pert]),
            **pair_values[pert],
            _L1: l1.get(pert, _NAN),
        }
        for pert in sorted(by_pert)
    ]
    return _Context(values=values, pairs=pairs, n_rows=len(perts))


def _check(inp: ScoringInputs) -> None:
    n_rows, n_genes = len(inp.perts), len(inp.genes)
    if len(inp.contexts) != n_rows:
        raise ValueError(
            f"{inp.dataset}: {len(inp.contexts)} contexts for {n_rows} perturbations"
        )
    for name in _ARRAYS:
        shape = np.shape(getattr(inp, name))
        if shape != (n_rows, n_genes):
            raise ValueError(
                f"{inp.dataset}: {name} has shape {shape}, expected {(n_rows, n_genes)}"
            )


def score(inputs: Sequence[ScoringInputs]) -> ScoreResult:
    """Score every row: row -> pair -> context -> dataset -> mean over datasets with rows.

    `per_context` values are the evaluation values (DE metrics rounded to float32 per
    context, L1 in float64); `aggregate` is the validation value (each dataset's float32
    mean over its contexts, then the float32 mean over datasets); `context_mean` is the
    NaN-skipping mean of the `per_context` rows. A dataset without rows is left out.
    """
    context_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    dataset_values: list[dict[str, float]] = []
    for inp in inputs:
        _check(inp)
        if not inp.perts:
            continue
        gene_index = {gene: i for i, gene in enumerate(inp.genes)}
        contexts = np.asarray(inp.contexts, dtype=object)
        scored: list[_Context] = []
        for context in sorted(set(inp.contexts)):
            ctx = _score_context(inp, np.flatnonzero(contexts == context), context, gene_index)
            scored.append(ctx)
            row: dict[str, object] = {
                "dataset": inp.dataset,
                "context": context,
                "n_rows": ctx.n_rows,
            }
            for key in METRIC_KEYS:
                value = ctx.values[key]
                row[key] = value if key == _L1 else _f32(value)
            context_rows.append(row)
            pair_rows.extend({"dataset": inp.dataset, **pair} for pair in ctx.pairs)
        dataset_values.append(
            {key: _f32(_mean_defined([c.values[key] for c in scored])) for key in METRIC_KEYS}
        )
    per_context = pd.DataFrame(
        context_rows, columns=["dataset", "context", "n_rows", *METRIC_KEYS]
    )
    per_pair = pd.DataFrame(
        pair_rows, columns=["dataset", "context", "perturbation", "n_rows", *PAIR_METRIC_KEYS]
    )
    aggregate = {key: _dataset_mean([d[key] for d in dataset_values]) for key in METRIC_KEYS}
    context_mean = {
        key: float(per_context[key].astype(float).mean(skipna=True)) if context_rows else _NAN
        for key in METRIC_KEYS
    }
    return ScoreResult(
        per_context=per_context,
        per_pair=per_pair,
        aggregate=aggregate,
        context_mean=context_mean,
    )
