"""Checkpoint loading and batched prediction shared by pie-eval and pie-infer."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pie.config import TrainConfig, resolved_train_config
from pie.data.datamodule import DataStats, PieDataModule
from pie.data.dataset import Batch, GeneGroup, RowRef, collate
from pie.model import PieModel, readout
from pie.utils import canonical_json, configure_determinism

log = logging.getLogger(__name__)

CKPT_FORMAT_VERSION = 1
STATE_PREFIX = "model."
PREDICTION_COLUMNS: tuple[str, ...] = ("p_de", "lfc_pred", "delta_p_pred")
PARQUET_METADATA_KEY = b"pie"
PARQUET_FORMAT_VERSION = 1
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass
class LoadedCheckpoint:
    config: TrainConfig  # paths resolved from portable form
    stats: DataStats
    state_dict: dict[str, Tensor]  # keys without the "model." prefix


@dataclass(frozen=True)
class RowSource:
    kind: Literal["split", "query"]  # split: labelled rows; query: any dir, RowRef.row = -1
    path: Path


@dataclass
class PredictionBlock:
    dataset: str
    genes: list[str]  # that dir's local axis
    contexts: list[str]  # (R,)
    perts: list[str]  # (R,)
    row_index: np.ndarray  # (R,) int64 row in the dir, -1 for query rows
    p_de: np.ndarray  # (R, G) float32
    lfc_pred: np.ndarray  # (R, G) float32
    delta_p_pred: np.ndarray  # (R, G) float32


@dataclass
class Predictions:
    blocks: list[PredictionBlock]  # one per dir with rows, in dir order


def load_checkpoint(ckpt_path: Path) -> LoadedCheckpoint:
    """torch.load(weights_only=False); reads only 'state_dict' and 'pie'."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    block = ckpt.get("pie") if isinstance(ckpt, dict) else None
    if not isinstance(block, dict) or block.get("format_version") != CKPT_FORMAT_VERSION:
        raise ValueError(f"{ckpt_path}: not a pie checkpoint (missing 'pie' block)")
    state = ckpt["state_dict"]
    foreign = [key for key in state if not key.startswith(STATE_PREFIX)]
    if foreign:
        raise ValueError(
            f"{ckpt_path}: state_dict keys without the {STATE_PREFIX!r} prefix: {foreign[:5]}"
        )
    return LoadedCheckpoint(
        config=resolved_train_config(TrainConfig.model_validate(block["config"])),
        stats=DataStats.model_validate(block["data_stats"]),
        state_dict={key[len(STATE_PREFIX) :]: value for key, value in state.items()},
    )


def batch_rows(rows: Sequence[RowRef], batch_size: int) -> list[list[int]]:
    """HARNESS SEAM. Forward batches as positions into `rows`.

    Each maximal run of consecutive rows with the same (dir_index, context) is cut into chunks of
    at most `batch_size`, so a batch never mixes contexts.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    batches: list[list[int]] = []
    start = 0
    for end in range(1, len(rows) + 1):
        run_ends = end == len(rows) or (rows[end].dir_index, rows[end].context) != (
            rows[start].dir_index,
            rows[start].context,
        )
        if run_ends:
            batches.extend(
                list(range(first, min(first + batch_size, end)))
                for first in range(start, end, batch_size)
            )
            start = end
    return batches


def _autocast(precision: str, device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """The training precision's autocast (Lightning's bf16-mixed plugin), else a no-op."""
    if precision == "bf16-mixed":
        return torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _move(tensor: Tensor | None, device: torch.device) -> Tensor | None:
    return None if tensor is None else tensor.to(device)


def _group_to(group: GeneGroup, device: torch.device) -> GeneGroup:
    return dataclasses.replace(
        group,
        rows=group.rows.to(device),
        gene_ids=group.gene_ids.to(device),
        ctrl_means=group.ctrl_means.to(device),
        evidence={key: value.to(device) for key, value in group.evidence.items()},
        fold_changes=_move(group.fold_changes, device),
        de_mask=_move(group.de_mask, device),
        tested=_move(group.tested, device),
        delta_p=_move(group.delta_p, device),
        lfc_true=_move(group.lfc_true, device),
    )


def _batch_to(batch: Batch, device: torch.device, token_dtype: torch.dtype | None) -> Batch:
    """Move a batch to `device`; `token_dtype` casts floating token sources (None keeps them)."""

    def tokens(t: Tensor) -> Tensor:
        t = t.to(device)
        if token_dtype is not None and t.is_floating_point() and t.dtype != token_dtype:
            t = t.to(token_dtype)
        return t

    return dataclasses.replace(
        batch,
        source_tokens={name: tokens(t) for name, t in batch.source_tokens.items()},
        source_masks={name: m.to(device) for name, m in batch.source_masks.items()},
        dataset_ids=batch.dataset_ids.to(device),
        row_index=batch.row_index.to(device),
        target_gene=batch.target_gene.to(device),
        target_gene_idx=batch.target_gene_idx.to(device),
        groups=[_group_to(group, device) for group in batch.groups],
    )


def predict_loaded(
    loaded: LoadedCheckpoint,
    rows: RowSource,
    preprocessed_dirs: list[Path] | None,
    device: str,
    batch_size: int = 16,
) -> Predictions:
    """Pinned stats, evidence from the training config's cache, per-context forward in row order."""
    configure_determinism(loaded.config.seed)
    train_data = loaded.config.data
    data_cfg = train_data
    if preprocessed_dirs is not None:
        data_cfg = train_data.model_copy(
            update={
                "preprocessed_dirs": [str(p) for p in preprocessed_dirs],
                "dataset_weights": None,
            }
        )
    dm = PieDataModule(
        data_cfg, None, loaded.stats, seed=loaded.config.seed, evidence_source=train_data
    )
    dm.setup_stats(rank=0)
    refs = dm.rows_from_split(rows.path) if rows.kind == "split" else dm.rows_from_query(rows.path)
    if not refs:
        raise ValueError(f"{rows.path}: selects no rows")
    dataset = dm.make_dataset(refs, "none")

    by_dir: dict[int, list[int]] = {}
    for position, ref in enumerate(refs):
        by_dir.setdefault(ref.dir_index, []).append(position)
    local = np.empty(len(refs), dtype=np.int64)
    store: dict[int, dict[str, np.ndarray]] = {}
    for di, positions in by_dir.items():
        local[positions] = np.arange(len(positions), dtype=np.int64)
        shape = (len(positions), len(dm.dirs[di].genes))
        store[di] = {name: np.zeros(shape, dtype=np.float32) for name in PREDICTION_COLUMNS}

    batches = batch_rows(refs, batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=data_cfg.num_workers,
        collate_fn=functools.partial(collate, source_dims=dict(dm.source_dims)),
    )
    torch_device = torch.device(device)
    model_cfg = loaded.config.model
    grid = loaded.stats.delta_p
    precision = loaded.config.trainer.precision
    with _deterministic_algorithms():
        model = PieModel(model_cfg, loaded.stats, dict(dm.source_dims), dm.gene_query_text())
        model.load_state_dict(loaded.state_dict, strict=True)
        model.to(torch_device).eval()
        # As in training: token sources arrive in their on-disk half dtype; mixed precision
        # autocasts them, a full-precision run casts them to the parameter dtype.
        token_dtype = None if "mixed" in precision else model.latents.dtype
        with torch.inference_mode():
            for positions, batch in zip(batches, loader, strict=True):
                index = np.asarray(positions, dtype=np.int64)
                with _autocast(precision, torch_device):
                    outputs = model(_batch_to(batch, torch_device, token_dtype))
                    reads = [readout(out, model_cfg, grid) for out in outputs]
                for group, read in zip(batch.groups, reads, strict=True):
                    target = local[index[group.rows.numpy()]]
                    arrays = store[group.dir_index]
                    arrays["p_de"][target] = read.p_de.float().cpu().numpy()
                    arrays["lfc_pred"][target] = read.lfc.float().cpu().numpy()
                    arrays["delta_p_pred"][target] = read.delta_p.float().cpu().numpy()
    log.info("predicted %d rows in %d batches on %s", len(refs), len(batches), torch_device)

    blocks: list[PredictionBlock] = []
    for di in sorted(by_dir):
        d = dm.dirs[di]
        positions = by_dir[di]
        blocks.append(
            PredictionBlock(
                dataset=d.dataset,
                genes=list(d.genes),
                contexts=[refs[p].context for p in positions],
                perts=[refs[p].perturbation for p in positions],
                row_index=np.asarray([refs[p].row for p in positions], dtype=np.int64),
                p_de=store[di]["p_de"],
                lfc_pred=store[di]["lfc_pred"],
                delta_p_pred=store[di]["delta_p_pred"],
            )
        )
    return Predictions(blocks=blocks)


def predict(
    ckpt_path: Path,
    rows: RowSource,
    preprocessed_dirs: list[Path] | None,
    device: str,
    batch_size: int = 16,
) -> Predictions:
    """Load the checkpoint, then predict_loaded()."""
    return predict_loaded(load_checkpoint(ckpt_path), rows, preprocessed_dirs, device, batch_size)


def _list_column(values: np.ndarray) -> pa.ListArray:
    """(R, G) float32 -> list<float32> of R rows, built from offsets (no per-row Python)."""
    rows, genes = values.shape
    if rows * genes > _INT32_MAX:
        raise ValueError(f"{rows} x {genes} values do not fit one parquet list column chunk")
    offsets = pa.array((np.arange(rows + 1, dtype=np.int64) * genes).astype(np.int32))
    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.ListArray.from_arrays(offsets, flat)


def write_predictions_parquet(preds: Predictions, path: Path) -> Path:
    """One row per (dataset, context, perturbation) with list<float32> p_de, lfc_pred,
    delta_p_pred; schema metadata b'pie' = {'format_version': 1, 'genes': {dataset: [...]}}."""
    if not preds.blocks:
        raise ValueError("no prediction blocks to write")
    genes = {block.dataset: list(block.genes) for block in preds.blocks}
    metadata = canonical_json({"format_version": PARQUET_FORMAT_VERSION, "genes": genes})
    schema = pa.schema(
        [
            pa.field("dataset", pa.string()),
            pa.field("context", pa.string()),
            pa.field("perturbation", pa.string()),
            *(pa.field(name, pa.list_(pa.float32())) for name in PREDICTION_COLUMNS),
        ],
        metadata={PARQUET_METADATA_KEY: metadata},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with pq.ParquetWriter(str(tmp), schema) as writer:
            for block in preds.blocks:
                n = len(block.contexts)
                columns = [
                    pa.array([block.dataset] * n, type=pa.string()),
                    pa.array(block.contexts, type=pa.string()),
                    pa.array(block.perts, type=pa.string()),
                    *(_list_column(getattr(block, name)) for name in PREDICTION_COLUMNS),
                ]
                writer.write_table(pa.Table.from_arrays(columns, schema=schema))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
