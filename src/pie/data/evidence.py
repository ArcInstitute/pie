"""Train-split evidence: what each perturbation did in the other training conditions.

The artifact holds (P, G) sufficient statistics (sums, squared sums, counts) over the train rows
of every dataset that has train rows, plus one shard per such dataset with a compact record per
donor row, so a query can remove its own (dataset, context, perturbation) before serving.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pie.data.preprocessed import PreprocessedDir
from pie.data.splits import Split, resolve_split, split_pairs
from pie.utils import StrictModel, atomic_dir, canonical_json, read_json, sha256_bytes, write_json

log = logging.getLogger(__name__)

EVIDENCE_SCHEMA_VERSION = 3
QUANTITY = "evidence_v3"
FOLD_CHANGE_SIGN_THRESHOLD = 1.0
RESPONSE_META = "response_meta.json"
SHARD_META = "shard_meta.json"
CONTEXT_DIR = "context_response"
CONTEXT_SECTION_VERSION = 1

# Global (P, G) arrays: name -> on-disk dtype.
GLOBAL_ARRAYS: dict[str, type] = {
    "dp_sum": np.float32,
    "dp_sumsq": np.float32,
    "dp_cnt": np.uint16,
    "tested_cnt": np.uint16,
    "up_cnt": np.uint16,
    "down_cnt": np.uint16,
    "lfc_sum": np.float32,
    "lfc_sumsq": np.float32,
    "lfc_cnt": np.uint16,
}
SUM_ARRAYS: tuple[str, ...] = ("dp_sum", "dp_sumsq", "lfc_sum", "lfc_sumsq")
COUNT_ARRAYS: tuple[str, ...] = ("dp_cnt", "tested_cnt", "up_cnt", "down_cnt", "lfc_cnt")
# Per-dataset (K, P, G) arrays and the global count each one mirrors.
PER_DATASET_ARRAYS: dict[str, type] = {"dp_cnt_by_ds": np.uint8, "tested_cnt_by_ds": np.uint8}
PER_DATASET_SOURCE: dict[str, str] = {"dp_cnt_by_ds": "dp_cnt", "tested_cnt_by_ds": "tested_cnt"}
# Served block layout: [value columns..., have flag, log1p(count)].
BLOCK_KEYS: tuple[str, ...] = ("evidence_dp", "evidence_de", "evidence_lfc", "evidence_prov")
CONTEXT_BLOCK_KEYS: tuple[str, ...] = ("evidence_ctx_dp", "evidence_ctx_de", "evidence_ctx_lfc")
EVIDENCE_KEYS: tuple[str, ...] = (*BLOCK_KEYS, *CONTEXT_BLOCK_KEYS)
HAVE_COL = -2
LOGCNT_COL = -1

_TESTED = 1
_UP = 2
_DOWN = 4
_LFC_VALID = 8
_KNOWN_FLAGS = _TESTED | _UP | _DOWN | _LFC_VALID
_RECORDS: tuple[tuple[str, type], ...] = (
    ("dp", np.float32),
    ("lfc", np.float32),
    ("flags", np.uint8),
)
_LFC_SAMPLE_PER_CHUNK = 20_000
_LFC_SAMPLE_MAX = 2_000_000


class EvidenceConfig(StrictModel):
    """data.evidence."""

    seed: int
    chunk: int
    lfc_clip_percentile: float


@dataclass(frozen=True)
class Rules:
    """Constants that define a donor observation; frozen into the artifact meta."""

    fdr_threshold: float
    fold_change_sign_threshold: float
    lfc_clip: float


@dataclass(frozen=True)
class Scales:
    """Standardisation constants from the training artifact (population std)."""

    dp: float
    lfc: float

    def to_dict(self) -> dict[str, float]:
        return {"dp": self.dp, "lfc": self.lfc}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scales:
        return cls(dp=float(d["dp"]), lfc=float(d["lfc"]))


@dataclass
class ContextShard:
    """Donor records and per-context totals of one dataset, on that dataset's gene axis."""

    dataset: str
    gene_symbols: tuple[str, ...]
    context_to_index: dict[str, int]
    donor_keys: tuple[tuple[str, str], ...]  # (context, perturbation) per donor row
    donor_rows: dict[tuple[str, str], np.ndarray]
    totals: dict[str, np.ndarray]
    delta_p: np.ndarray
    lfc: np.ndarray
    flags: np.ndarray


@dataclass
class Evidence:
    """A loaded evidence artifact (all arrays mmapped)."""

    path: Path
    key: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]
    by_ds: dict[str, np.ndarray]
    rules: Rules
    scales: Scales
    contributing_datasets: list[str]
    pert_to_index: dict[str, int]
    gene_symbols: list[str]
    train_json_sha256: str
    shards: dict[str, ContextShard]
    donor_keys: frozenset[tuple[str, str, str]]

    @property
    def n_donor_datasets(self) -> int:
        return len(self.contributing_datasets)


# ---------------------------------------------------------------------------------------------
# Statistics of donor rows
# ---------------------------------------------------------------------------------------------


def _own_row_stats(
    *, fc: np.ndarray, fdr: np.ndarray, delta: np.ndarray, tested: np.ndarray, rules: Rules
) -> dict[str, np.ndarray]:
    """Contribution of one row (G,) or a chunk (n, G) to every statistic.

    Float64 for sums, int64 for counts; `delta` must be finite.
    """
    fc64 = np.asarray(fc, dtype=np.float64)
    fdr64 = np.asarray(fdr, dtype=np.float64)
    dp = np.asarray(delta, dtype=np.float64)
    tested_b = np.asarray(tested, dtype=bool)
    sig = tested_b & (fdr64 < rules.fdr_threshold)
    s = rules.fold_change_sign_threshold
    up = sig & (fc64 > s)
    down = sig & (fc64 < s)
    lfc_ok = tested_b & np.isfinite(fc64) & (fc64 > 0.0)
    safe_fc = np.where(lfc_ok, fc64, 1.0)
    lfc = np.where(lfc_ok, np.clip(np.log2(safe_fc), -rules.lfc_clip, rules.lfc_clip), 0.0)
    return {
        "dp_sum": dp,
        "dp_sumsq": dp * dp,
        "dp_cnt": np.ones(tested_b.shape, dtype=np.int64),
        "tested_cnt": tested_b.astype(np.int64),
        "up_cnt": up.astype(np.int64),
        "down_cnt": down.astype(np.int64),
        "lfc_sum": lfc,
        "lfc_sumsq": lfc * lfc,
        "lfc_cnt": lfc_ok.astype(np.int64),
    }


def _pop_std(total: float, total_sq: float, n: float) -> float:
    if n <= 0:
        return 1.0
    mean = total / n
    var = total_sq / n - mean * mean
    std = float(np.sqrt(max(var, 0.0)))
    return std if std > 0.0 else 1.0


def _scales_from_totals(totals: Mapping[str, float]) -> Scales:
    """`totals` holds the sum over every (pert, gene) cell of the named arrays."""
    return Scales(
        dp=_pop_std(totals["dp_sum"], totals["dp_sumsq"], totals["dp_cnt"]),
        lfc=_pop_std(totals["lfc_sum"], totals["lfc_sumsq"], totals["lfc_cnt"]),
    )


def _subtract_donors(
    stats: dict[str, np.ndarray],
    by_ds: dict[str, np.ndarray],
    own: Mapping[str, np.ndarray],
    slot: int,
) -> int:
    """Remove donor contributions in place; slot -1 changes only the global vectors.

    A negative count means artifact and data disagree: that gene is zeroed rather than served
    sign-flipped. Returns how many genes that hit.
    """
    for k in GLOBAL_ARRAYS:
        stats[k] -= own[k]
    if slot >= 0:
        for k, src in PER_DATASET_SOURCE.items():
            by_ds[k][slot] -= own[src]
    broken = np.zeros(stats["dp_cnt"].shape, dtype=bool)
    for k in COUNT_ARRAYS:
        broken |= stats[k] < 0
    for k in PER_DATASET_ARRAYS:
        broken |= (by_ds[k] < 0).any(axis=0)
    if broken.any():
        for k in GLOBAL_ARRAYS:
            stats[k][broken] = 0
        for k in PER_DATASET_ARRAYS:
            by_ds[k][:, broken] = 0
    return int(broken.sum())


def _mean_std(
    total: np.ndarray, total_sq: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Sums are stored as float32, so after an exact leave-one-out that empties a cell a ~1e-8
    # residual can remain: the mean is defined as 0 when there is no donor.
    n_safe = np.maximum(n, 1)
    mean = np.where(n > 0, total / n_safe, 0.0)
    var = total_sq / n_safe - mean * mean
    std = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), 0.0)
    return mean, std


def _derive_blocks(
    stats: Mapping[str, np.ndarray], by_ds: Mapping[str, np.ndarray], scales: Scales
) -> dict[str, np.ndarray]:
    """Post-subtraction statistics -> the served float32 blocks (a gene with no donor is 0)."""
    n = stats["dp_cnt"]
    mean, std = _mean_std(stats["dp_sum"], stats["dp_sumsq"], n)
    dp = np.stack([mean / scales.dp, std / scales.dp, n > 0, np.log1p(n)], axis=-1)

    t = stats["tested_cnt"]
    t_safe = np.maximum(t, 1)
    de = np.stack(
        [stats["up_cnt"] / t_safe, stats["down_cnt"] / t_safe, t > 0, np.log1p(t)], axis=-1
    )

    m = stats["lfc_cnt"]
    mean_l, std_l = _mean_std(stats["lfc_sum"], stats["lfc_sumsq"], m)
    lfc = np.stack([mean_l / scales.lfc, std_l / scales.lfc, m > 0, np.log1p(m)], axis=-1)

    parts: list[np.ndarray] = []
    for k in range(by_ds["dp_cnt_by_ds"].shape[0]):
        c = by_ds["dp_cnt_by_ds"][k]
        tk = by_ds["tested_cnt_by_ds"][k]
        parts.extend([c > 0, np.log1p(c), np.log1p(tk)])
    prov = np.stack(parts, axis=-1)
    out = {"evidence_dp": dp, "evidence_de": de, "evidence_lfc": lfc, "evidence_prov": prov}
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def _pack_rows(own: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialize row contributions without reinterpreting the label rules."""
    shape = own["dp_sum"].shape
    if len(shape) != 2 or any(value.shape != shape for value in own.values()):
        raise ValueError("row contribution shape must be a shared (rows, genes) shape")
    if not np.all(own["dp_cnt"] == 1):
        raise ValueError("dp_cnt must be one for dense delta_p records")
    for key in COUNT_ARRAYS:
        if not np.isin(own[key], (0, 1)).all():
            raise ValueError(f"{key} must contain binary row contributions")
    dp = np.ascontiguousarray(own["dp_sum"], dtype=np.float32)
    lfc = np.ascontiguousarray(own["lfc_sum"], dtype=np.float32)
    flags = np.zeros(shape, dtype=np.uint8)
    for key, bit in (
        ("tested_cnt", _TESTED),
        ("up_cnt", _UP),
        ("down_cnt", _DOWN),
        ("lfc_cnt", _LFC_VALID),
    ):
        flags |= own[key].astype(np.uint8) * bit
    _validate_records(dp, lfc, flags)
    return dp, lfc, flags


def _validate_records(delta_p: np.ndarray, lfc: np.ndarray, flags: np.ndarray) -> None:
    if delta_p.ndim != 2 or delta_p.shape != lfc.shape or delta_p.shape != flags.shape:
        raise ValueError("delta_p, lfc, flags shape must be a shared (rows, genes) shape")
    for name, arr, dtype in (
        ("delta_p", delta_p, np.float32),
        ("lfc", lfc, np.float32),
        ("flags", flags, np.uint8),
    ):
        if arr.dtype != np.dtype(dtype):
            raise ValueError(f"{name} dtype must be {np.dtype(dtype)}")
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
    if np.any(flags & (255 ^ _KNOWN_FLAGS)):
        raise ValueError("flags contain unknown bits")
    tested = (flags & _TESTED) != 0
    up = (flags & _UP) != 0
    down = (flags & _DOWN) != 0
    valid = (flags & _LFC_VALID) != 0
    if np.any(up & down) or np.any((up | down | valid) & ~tested):
        raise ValueError("flags require tested eligibility and mutually exclusive up/down")
    if np.any(lfc[~valid] != 0):
        raise ValueError("lfc must be zero when LFC_VALID is absent")


def _unpack_rows(delta_p: np.ndarray, lfc: np.ndarray, flags: np.ndarray) -> dict[str, np.ndarray]:
    """Recover additive statistics from serialized records, including empty slices."""
    _validate_records(delta_p, lfc, flags)
    dp64 = delta_p.astype(np.float64)
    lfc64 = lfc.astype(np.float64)
    return {
        "dp_sum": dp64,
        "dp_sumsq": dp64 * dp64,
        "dp_cnt": np.ones(flags.shape, dtype=np.int64),
        "tested_cnt": ((flags & _TESTED) != 0).astype(np.int64),
        "up_cnt": ((flags & _UP) != 0).astype(np.int64),
        "down_cnt": ((flags & _DOWN) != 0).astype(np.int64),
        "lfc_sum": lfc64,
        "lfc_sumsq": lfc64 * lfc64,
        "lfc_cnt": ((flags & _LFC_VALID) != 0).astype(np.int64),
    }


def _derive_context_blocks(
    stats: Mapping[str, np.ndarray], scales: Scales
) -> dict[str, np.ndarray]:
    """Context blocks use the donor-block definitions with a single provenance slot."""
    by_ds = {
        "dp_cnt_by_ds": stats["dp_cnt"][None, :],
        "tested_cnt_by_ds": stats["tested_cnt"][None, :],
    }
    blocks = _derive_blocks(stats, by_ds, scales)
    return {
        "evidence_ctx_dp": blocks["evidence_dp"],
        "evidence_ctx_de": blocks["evidence_de"],
        "evidence_ctx_lfc": blocks["evidence_lfc"],
    }


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


@dataclass
class _BuildDir:
    dataset: str
    genes: tuple[str, ...]
    kept: np.ndarray  # sorted train rows of the dir
    fc: np.ndarray
    fdr: np.ndarray
    dp: np.ndarray
    tested: np.ndarray
    pert_names: list[str]  # per kept row
    donor_keys: tuple[tuple[str, str], ...]  # (context, perturbation) per kept row


def _dataset_name(d: PreprocessedDir) -> str:
    return d.dataset


def contributing_dirs(
    dirs: Sequence[PreprocessedDir], train_rows: Mapping[str, np.ndarray]
) -> list[PreprocessedDir]:
    """Dirs with at least one train row, sorted by dataset name (builder input and K order)."""
    keep = [d for d in dirs if len(train_rows.get(d.dataset, ())) > 0]
    return sorted(keep, key=_dataset_name)


def evidence_cache_key(
    dirs: Sequence[PreprocessedDir],
    train_json_sha256: str,
    cfg: EvidenceConfig,
    fdr_threshold: float,
) -> str:
    """sha256 over the given dirs (sorted by name; callers pass the contributing dirs)."""
    payload = {
        "schema": EVIDENCE_SCHEMA_VERSION,
        "dirs": [
            {"dataset": d.dataset, "meta_sha256": d.meta_sha256()}
            for d in sorted(dirs, key=_dataset_name)
        ],
        "train_json_sha256": train_json_sha256,
        "seed": cfg.seed,
        "chunk": cfg.chunk,
        "lfc_clip_percentile": cfg.lfc_clip_percentile,
        "fdr_threshold": float(fdr_threshold),
    }
    return sha256_bytes(canonical_json(payload))


def _open_build_dir(d: PreprocessedDir, kept: np.ndarray) -> _BuildDir:
    keys = d.row_keys()
    donor_keys = tuple(keys[int(r)] for r in kept)
    if len(set(donor_keys)) != len(donor_keys):
        raise ValueError(f"{d.dataset}: more than one train row per (context, perturbation)")
    log.info(
        "%s: %d rows, %d genes; %d train rows",
        d.dataset,
        d.meta.num_rows,
        d.meta.num_genes,
        len(kept),
    )
    return _BuildDir(
        dataset=d.dataset,
        genes=tuple(d.genes),
        kept=np.asarray(kept, dtype=np.int64),
        fc=d.fold_changes,
        fdr=d.fdr,
        dp=d.delta_p,
        tested=d.tested,
        pert_names=[key[1] for key in donor_keys],
        donor_keys=donor_keys,
    )


def _sample_abs_log2fc(b: _BuildDir, chunk: int, rng: np.random.Generator) -> np.ndarray:
    out: list[np.ndarray] = []
    for s in range(0, len(b.kept), chunk):
        idx = b.kept[s : s + chunk]
        fc = np.asarray(b.fc[idx], dtype=np.float64)
        ok = np.asarray(b.tested[idx], dtype=bool) & np.isfinite(fc) & (fc > 0.0)
        vals = np.abs(np.log2(fc[ok]))
        if vals.size:
            take = min(vals.size, _LFC_SAMPLE_PER_CHUNK)
            out.append(rng.choice(vals, size=take, replace=False))
    return np.concatenate(out) if out else np.zeros(0)


def _build_context_response(
    root: Path,
    bdirs: Sequence[_BuildDir],
    *,
    rules: Rules,
    train_json_sha256: str,
    chunk: int,
) -> dict[str, Any]:
    """One shard per contributing dataset, from exactly the rows the global arrays accumulate."""
    entries = []
    for slot, b in enumerate(bdirs):
        relative = f"{CONTEXT_DIR}/{slot:04d}"
        path = root / relative
        path.mkdir(parents=True)
        n, g = len(b.kept), len(b.genes)
        contexts = sorted({key[0] for key in b.donor_keys})
        ctx_index = {name: i for i, name in enumerate(contexts)}
        row_contexts = np.array([ctx_index[key[0]] for key in b.donor_keys], dtype=np.int64)
        totals = {
            key: np.zeros((len(contexts), g), dtype=np.float64 if key in SUM_ARRAYS else np.int64)
            for key in GLOBAL_ARRAYS
        }
        records = {
            name: np.lib.format.open_memmap(
                path / f"donor_{name}.npy", mode="w+", dtype=dtype, shape=(n, g)
            )
            for name, dtype in _RECORDS
        }
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            idx = b.kept[start:stop]
            own = _own_row_stats(
                fc=b.fc[idx], fdr=b.fdr[idx], delta=b.dp[idx], tested=b.tested[idx], rules=rules
            )
            dp, lfc, flags = _pack_rows(own)
            for name, values in (("dp", dp), ("lfc", lfc), ("flags", flags)):
                records[name][start:stop] = values
            for key, values in _unpack_rows(dp, lfc, flags).items():
                np.add.at(totals[key], row_contexts[start:stop], values)
        for record in records.values():
            record.flush()
        for key, values in totals.items():
            dtype = np.float64 if key in SUM_ARRAYS else np.uint32
            if key not in SUM_ARRAYS and np.any(values > np.iinfo(np.uint32).max):
                raise ValueError(f"{path}/{key}: count exceeds uint32")
            np.save(path / f"{key}.npy", values.astype(dtype))
        write_json(
            path / SHARD_META,
            {
                "dataset": b.dataset,
                "gene_symbols": list(b.genes),
                "contexts": contexts,
                "donor_keys": [list(key) for key in b.donor_keys],
                "n_rows": n,
                "n_genes": g,
                "n_contexts": len(contexts),
                "train_json_sha256": train_json_sha256,
            },
        )
        entries.append({"dataset": b.dataset, "path": relative})
    return {"version": CONTEXT_SECTION_VERSION, "shards": entries}


def build_evidence(
    dirs: Sequence[PreprocessedDir],
    train_split: Split,
    train_json_sha256: str,
    cfg: EvidenceConfig,
    fdr_threshold: float,
    out: Path,
) -> Path:
    """Accumulate the train rows of `train_split` into `out` (a missing or empty dir)."""
    if cfg.chunk < 1:
        raise ValueError("data.evidence.chunk must be positive")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    train_rows = resolve_split(train_split, dirs)
    contrib = contributing_dirs(dirs, train_rows)
    if not contrib:
        raise ValueError("no row of any preprocessed dir is listed in the train split")
    key = evidence_cache_key(contrib, train_json_sha256, cfg, fdr_threshold)
    bdirs = [_open_build_dir(d, train_rows[d.dataset]) for d in contrib]
    names = [b.dataset for b in bdirs]

    # Union axes over the train rows.
    genes = sorted({g for b in bdirs for g in b.genes})
    perts = sorted({p for b in bdirs for p in b.pert_names})
    gidx = {g: i for i, g in enumerate(genes)}
    pidx = {p: i for i, p in enumerate(perts)}
    n_p, n_g, n_k = len(perts), len(genes), len(bdirs)
    log.info("evidence axes: %d perts x %d genes, %d datasets", n_p, n_g, n_k)

    # Pass 1: LFC clip from a sample of |log2 fc| over tested cells.
    rng = np.random.default_rng(cfg.seed)
    sample = np.concatenate([_sample_abs_log2fc(b, cfg.chunk, rng) for b in bdirs])
    if sample.size > _LFC_SAMPLE_MAX:
        sample = rng.choice(sample, size=_LFC_SAMPLE_MAX, replace=False)
    lfc_clip = float(np.percentile(sample, cfg.lfc_clip_percentile)) if sample.size else 1.0
    if not lfc_clip > 0.0:
        lfc_clip = 1.0
    rules = Rules(float(fdr_threshold), FOLD_CHANGE_SIGN_THRESHOLD, lfc_clip)
    log.info("lfc_clip = %.4f from %d sampled |log2 fc|", lfc_clip, sample.size)

    # Pass 2: accumulate dir-local first, then one column remap per dir.
    acc: dict[str, NDArray[np.float64]] = {
        k: np.zeros((n_p, n_g), dtype=np.float64 if k in SUM_ARRAYS else np.int64)
        for k in GLOBAL_ARRAYS
    }
    by_ds: dict[str, NDArray[np.int64]] = {
        k: np.zeros((n_k, n_p, n_g), dtype=np.int64) for k in PER_DATASET_ARRAYS
    }
    n_rows_per_dataset: dict[str, int] = {}
    for slot, b in enumerate(bdirs):
        local_perts = sorted(set(b.pert_names))
        lp = {p: i for i, p in enumerate(local_perts)}
        rows_local = np.array([lp[p] for p in b.pert_names], dtype=np.int64)
        g_local = len(b.genes)
        loc: dict[str, NDArray[np.float64]] = {
            k: np.zeros(
                (len(local_perts), g_local), dtype=np.float64 if k in SUM_ARRAYS else np.int64
            )
            for k in GLOBAL_ARRAYS
        }
        for s in range(0, len(b.kept), cfg.chunk):
            idx = b.kept[s : s + cfg.chunk]
            dp = np.asarray(b.dp[idx], dtype=np.float64)
            if not np.isfinite(dp).all():
                raise ValueError(
                    f"{b.dataset}: non-finite delta_p in train rows {idx[0]}..{idx[-1]}"
                )
            own = _own_row_stats(
                fc=np.asarray(b.fc[idx]),
                fdr=np.asarray(b.fdr[idx]),
                delta=dp,
                tested=np.asarray(b.tested[idx]),
                rules=rules,
            )
            for k in GLOBAL_ARRAYS:
                np.add.at(loc[k], rows_local[s : s + cfg.chunk], own[k])
        gp = np.array([pidx[p] for p in local_perts], dtype=np.int64)
        cols = np.array([gidx[g] for g in b.genes], dtype=np.int64)
        for k in GLOBAL_ARRAYS:
            acc[k][gp[:, None], cols[None, :]] += loc[k]
        for k, src in PER_DATASET_SOURCE.items():
            by_ds[k][slot][gp[:, None], cols[None, :]] += loc[src]
        n_rows_per_dataset[b.dataset] = len(b.kept)
        log.info("%s: accumulated %d rows over %d perts", b.dataset, len(b.kept), len(local_perts))

    for k in COUNT_ARRAYS:
        if acc[k].max() > np.iinfo(np.uint16).max:
            raise ValueError(f"{k} exceeds uint16 ({acc[k].max()})")
    for k in PER_DATASET_ARRAYS:
        if by_ds[k].max() > np.iinfo(np.uint8).max:
            raise ValueError(f"{k} exceeds uint8 ({by_ds[k].max()})")

    totals = {
        k: float(acc[k].sum())
        for k in ("dp_sum", "dp_sumsq", "dp_cnt", "lfc_sum", "lfc_sumsq", "lfc_cnt")
    }
    scales = _scales_from_totals(totals)

    for k, dt in GLOBAL_ARRAYS.items():
        np.save(out / f"{k}.npy", acc[k].astype(dt))
    for k, dt in PER_DATASET_ARRAYS.items():
        np.save(out / f"{k}.npy", by_ds[k].astype(dt))
    section = _build_context_response(
        out, bdirs, rules=rules, train_json_sha256=train_json_sha256, chunk=cfg.chunk
    )
    meta = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "quantity": QUANTITY,
        "cache_key": key,
        "contributing_datasets": names,
        "train_json_sha256": train_json_sha256,
        "n_rows_accumulated": sum(n_rows_per_dataset.values()),
        "n_rows_per_dataset": n_rows_per_dataset,
        "rules": asdict(rules),
        "scales": scales.to_dict(),
        "lfc_clip_percentile": cfg.lfc_clip_percentile,
        "seed": cfg.seed,
        "chunk": cfg.chunk,
        "n_perts": n_p,
        "n_genes": n_g,
        "pert_to_index": pidx,
        "gene_symbols": genes,
        "context_response": section,
    }
    write_json(out / RESPONSE_META, meta)

    # Self-check: exact leave-one-out on three train rows of the first dataset.
    first = bdirs[0]
    first_cols = np.array([gidx[g] for g in first.genes], dtype=np.int64)
    for j in sorted({0, len(first.kept) // 2, len(first.kept) - 1}):
        r = int(first.kept[j])
        own = _own_row_stats(
            fc=np.asarray(first.fc[r]),
            fdr=np.asarray(first.fdr[r]),
            delta=np.asarray(first.dp[r], dtype=np.float64),
            tested=np.asarray(first.tested[r]),
            rules=rules,
        )
        p = pidx[first.pert_names[j]]
        for k in COUNT_ARRAYS:
            if (acc[k][p][first_cols] - own[k]).min() < 0:
                raise ValueError(
                    f"evidence self-check failed: {k} negative after leave-one-out on "
                    f"{first.dataset} row {r}"
                )
    log.info("wrote evidence %s", key)
    return out


# ---------------------------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------------------------


def _load_array(path: Path, shape: tuple[int, ...], dtype: type) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape != shape or arr.dtype != np.dtype(dtype):
        raise ValueError(
            f"{path}: shape/dtype {arr.shape}/{arr.dtype}, expected {shape}/{np.dtype(dtype)}"
        )
    return arr


def _string_axis(meta: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    values = meta.get(key)
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"{path}: {key} must be a string list")
    if len(set(values)) != len(values):
        raise ValueError(f"{path}: {key} contains duplicates")
    return tuple(values)


def _load_context_shard(path: Path, dataset: str, n_rows: int, train_hash: str) -> ContextShard:
    meta_path = path / SHARD_META
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")
    if meta.get("dataset") != dataset:
        raise ValueError(f"{meta_path}: dataset disagrees with the artifact")
    if meta.get("train_json_sha256") != train_hash:
        raise ValueError(f"{meta_path}: train_json_sha256 disagrees with the artifact")
    genes = _string_axis(meta, "gene_symbols", meta_path)
    contexts = _string_axis(meta, "contexts", meta_path)
    for field, expected in (
        ("n_rows", n_rows),
        ("n_genes", len(genes)),
        ("n_contexts", len(contexts)),
    ):
        if type(meta.get(field)) is not int or meta[field] != expected:
            raise ValueError(f"{meta_path}: {field} must equal {expected}")
    raw_keys = meta.get("donor_keys")
    if (
        not isinstance(raw_keys, list)
        or len(raw_keys) != n_rows
        or any(
            not isinstance(key, list)
            or len(key) != 2
            or any(not isinstance(v, str) for v in key)
            for key in raw_keys
        )
    ):
        raise ValueError(f"{meta_path}: donor_keys must hold n_rows [context, perturbation] pairs")
    keys = tuple((key[0], key[1]) for key in raw_keys)
    if len(set(keys)) != len(keys) or {key[0] for key in keys} != set(contexts):
        raise ValueError(f"{meta_path}: donor_keys repeat or disagree with contexts")
    ctx_index = {name: i for i, name in enumerate(contexts)}
    donor_groups: dict[tuple[str, str], list[int]] = {}
    for row, key in enumerate(keys):
        donor_groups.setdefault(key, []).append(row)
    donor_rows = {key: np.array(rows, dtype=np.int64) for key, rows in donor_groups.items()}
    g = len(genes)
    dp = _load_array(path / "donor_dp.npy", (n_rows, g), np.float32)
    lfc = _load_array(path / "donor_lfc.npy", (n_rows, g), np.float32)
    flags = _load_array(path / "donor_flags.npy", (n_rows, g), np.uint8)
    totals = {
        key: _load_array(
            path / f"{key}.npy",
            (len(contexts), g),
            np.float64 if key in SUM_ARRAYS else np.uint32,
        )
        for key in GLOBAL_ARRAYS
    }
    reconstructed = {
        key: np.zeros((len(contexts), g), dtype=np.float64 if key in SUM_ARRAYS else np.int64)
        for key in GLOBAL_ARRAYS
    }
    # Bound temporary donor statistics to roughly 1M gene cells, even on large axes.
    chunk = max(1, min(2048, 1_000_000 // max(g, 1)))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        try:
            rows = _unpack_rows(dp[start:stop], lfc[start:stop], flags[start:stop])
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        indices = np.array([ctx_index[key[0]] for key in keys[start:stop]], dtype=np.int64)
        for name, values in rows.items():
            np.add.at(reconstructed[name], indices, values)
    for name, expected_totals in reconstructed.items():
        actual = totals[name]
        equal = (
            np.allclose(actual, expected_totals, rtol=1e-13, atol=1e-12)
            if name in SUM_ARRAYS
            else np.array_equal(actual, expected_totals)
        )
        if not equal:
            raise ValueError(f"{path}/{name}.npy: totals disagree with the donor records")
    return ContextShard(dataset, genes, ctx_index, keys, donor_rows, totals, dp, lfc, flags)


def _load_context_shards(
    root: Path,
    section: object,
    contributing: Sequence[str],
    parent_gene_symbols: Sequence[str],
    n_rows_per_dataset: Mapping[str, int],
    train_hash: str,
) -> dict[str, ContextShard]:
    label = f"{root}: {CONTEXT_DIR}"
    if not isinstance(section, dict) or section.get("version") != CONTEXT_SECTION_VERSION:
        raise ValueError(f"{label}: unsupported or missing section")
    entries = section.get("shards")
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError(f"{label}: shards must be a list of dataset/path entries")
    names = [entry.get("dataset") for entry in entries]
    if names != list(contributing) or len(set(names)) != len(names):
        raise ValueError(f"{label}: shard datasets must equal contributing_datasets in order")
    shards: dict[str, ContextShard] = {}
    seen: set[Path] = set()
    root_resolved = root.resolve()
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"{label}: invalid shard path {relative!r}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root_resolved) or path in seen:
            raise ValueError(f"{label}: shard path escapes the root or repeats: {relative}")
        seen.add(path)
        dataset = str(entry["dataset"])
        if dataset not in n_rows_per_dataset:
            raise ValueError(f"{label}: missing n_rows_per_dataset for {dataset}")
        shards[dataset] = _load_context_shard(
            path, dataset, int(n_rows_per_dataset[dataset]), train_hash
        )
    parent = set(parent_gene_symbols)
    if len(parent) != len(parent_gene_symbols):
        raise ValueError(f"{label}: gene_symbols contain duplicates")
    covered: set[str] = set()
    for name, shard in shards.items():
        local = set(shard.gene_symbols)
        if not local <= parent:
            raise ValueError(f"{label}/{name}: shard genes absent from the artifact gene axis")
        covered |= local
    if covered != parent:
        raise ValueError(f"{label}: shard genes do not cover the artifact gene axis")
    return shards


def load_evidence(path: Path) -> Evidence:
    """Mmap and validate an evidence dir (schema, shapes, shard coverage, shard totals)."""
    path = Path(path)
    meta = read_json(path / RESPONSE_META)
    if not isinstance(meta, dict) or meta.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"{path}: evidence schema_version must be {EVIDENCE_SCHEMA_VERSION}")
    contributing = [str(name) for name in meta["contributing_datasets"]]
    if not contributing:
        raise ValueError(f"{path}: contributing_datasets is empty")
    n_p, n_g, n_k = int(meta["n_perts"]), int(meta["n_genes"]), len(contributing)
    arrays = {
        name: _load_array(path / f"{name}.npy", (n_p, n_g), dtype)
        for name, dtype in GLOBAL_ARRAYS.items()
    }
    by_ds = {
        name: _load_array(path / f"{name}.npy", (n_k, n_p, n_g), dtype)
        for name, dtype in PER_DATASET_ARRAYS.items()
    }
    pert_to_index = {str(k): int(v) for k, v in meta["pert_to_index"].items()}
    gene_symbols = [str(g) for g in meta["gene_symbols"]]
    if len(pert_to_index) != n_p or len(gene_symbols) != n_g:
        raise ValueError(f"{path}: meta axes do not match n_perts={n_p}, n_genes={n_g}")
    scales = Scales.from_dict(meta["scales"])
    if not (scales.dp > 0.0 and scales.lfc > 0.0):
        raise ValueError(f"{path}: scales must be > 0, got {scales}")
    rules = meta["rules"]
    train_hash = str(meta["train_json_sha256"])
    shards = _load_context_shards(
        path,
        meta.get("context_response"),
        contributing,
        gene_symbols,
        meta["n_rows_per_dataset"],
        train_hash,
    )
    donor_keys = frozenset(
        (name, context, pert)
        for name, shard in shards.items()
        for context, pert in shard.donor_keys
    )
    return Evidence(
        path=path,
        key=str(meta["cache_key"]),
        meta=meta,
        arrays=arrays,
        by_ds=by_ds,
        rules=Rules(
            float(rules["fdr_threshold"]),
            float(rules["fold_change_sign_threshold"]),
            float(rules["lfc_clip"]),
        ),
        scales=scales,
        contributing_datasets=contributing,
        pert_to_index=pert_to_index,
        gene_symbols=gene_symbols,
        train_json_sha256=train_hash,
        shards=shards,
        donor_keys=donor_keys,
    )


# ---------------------------------------------------------------------------------------------
# Cache and leakage check
# ---------------------------------------------------------------------------------------------


def check_leakage(
    evidence: Evidence,
    train_pairs: set[tuple[str, str, str]],
    forbidden_pairs: set[tuple[str, str, str]],
) -> None:
    """Every donor must be a train pair and none may be an evaluation pair."""
    not_train = sorted(key for key in evidence.donor_keys if key not in train_pairs)
    if not_train:
        raise ValueError(
            f"{len(not_train)} evidence donors are not in the train split, e.g. {not_train[:3]}; "
            "the evidence was built from another split"
        )
    leaked = sorted(key for key in evidence.donor_keys if key in forbidden_pairs)
    if leaked:
        raise ValueError(
            f"{len(leaked)} evidence donors are evaluation rows (label leak), e.g. {leaked[:3]}"
        )


def load_or_build_evidence(
    dirs: Sequence[PreprocessedDir],
    train_split: Split,
    train_json_sha256: str,
    cfg: EvidenceConfig,
    fdr_threshold: float,
    cache_root: Path,
    forbidden_pairs: set[tuple[str, str, str]],
) -> Evidence:
    """Load `cache_root/evidence/<key>`, building it first when missing; then check leakage."""
    contrib = contributing_dirs(dirs, resolve_split(train_split, dirs))
    if not contrib:
        raise ValueError("no row of any preprocessed dir is listed in the train split")
    key = evidence_cache_key(contrib, train_json_sha256, cfg, fdr_threshold)
    final = Path(cache_root) / "evidence" / key
    if not (final / RESPONSE_META).is_file():
        final.parent.mkdir(parents=True, exist_ok=True)
        log.info("building evidence %s", key)
        with atomic_dir(final) as tmp:
            build_evidence(dirs, train_split, train_json_sha256, cfg, fdr_threshold, tmp)
    loaded = load_evidence(final)
    if loaded.key != key:
        raise ValueError(f"{final}: cache_key {loaded.key} does not match the directory name")
    check_leakage(loaded, split_pairs(train_split), forbidden_pairs)
    return loaded


# ---------------------------------------------------------------------------------------------
# Serving on a run's gene axis
# ---------------------------------------------------------------------------------------------


@dataclass
class EvidenceIndex:
    """Evidence mapped onto a run's gene axis and dataset names (built once per datamodule)."""

    evidence: Evidence
    pert_rows: dict[str, int]
    gene_cols: np.ndarray
    context_gene_cols: dict[str, np.ndarray]
    ds_slot: dict[str, int]


def build_evidence_index(evidence: Evidence, gene_axis: Sequence[str]) -> EvidenceIndex:
    """Artifact column per axis gene, shard column per axis gene and dataset slot, by name."""
    art_col = {g: i for i, g in enumerate(evidence.gene_symbols)}
    gene_cols = np.array([art_col.get(str(g), -1) for g in gene_axis], dtype=np.int64)
    context_gene_cols: dict[str, np.ndarray] = {}
    for dataset, shard in evidence.shards.items():
        local = {g: i for i, g in enumerate(shard.gene_symbols)}
        context_gene_cols[dataset] = np.array(
            [local.get(str(g), -1) for g in gene_axis], dtype=np.int64
        )
    ds_slot = {name: i for i, name in enumerate(evidence.contributing_datasets)}
    return EvidenceIndex(
        evidence=evidence,
        pert_rows=dict(evidence.pert_to_index),
        gene_cols=gene_cols,
        context_gene_cols=context_gene_cols,
        ds_slot=ds_slot,
    )


def _zero_stats(g: int) -> dict[str, np.ndarray]:
    return {
        key: np.zeros(g, dtype=np.float64 if key in SUM_ARRAYS else np.int64)
        for key in GLOBAL_ARRAYS
    }


def _query_response_stats(
    index: EvidenceIndex,
    *,
    dataset: str,
    context: str,
    perturbation: str,
    gene_union_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sum of every stored donor record of the query's (dataset, context, perturbation)."""
    stats = _zero_stats(len(gene_union_ids))
    shard = index.evidence.shards.get(dataset)
    if shard is None:
        return stats
    cols = index.context_gene_cols[dataset][gene_union_ids]
    covered = cols >= 0
    take = cols[covered]
    excluded = shard.donor_rows.get((context, perturbation))
    if excluded is not None and excluded.size:
        if (
            np.any(excluded < 0)
            or np.any(excluded >= len(shard.donor_keys))
            or len(np.unique(excluded)) != len(excluded)
            or any(shard.donor_keys[int(i)] != (context, perturbation) for i in excluded)
        ):
            raise ValueError("evidence: inconsistent donor indices")
        # Bound exclusion intermediates even when one perturbation has many records.
        chunk = max(1, 1_000_000 // max(len(take), 1))
        for start in range(0, len(excluded), chunk):
            ix = np.ix_(excluded[start : start + chunk], take)
            removed = _unpack_rows(shard.delta_p[ix], shard.lfc[ix], shard.flags[ix])
            for key in GLOBAL_ARRAYS:
                stats[key][covered] += removed[key].sum(axis=0)
    return stats


def _donor_blocks(
    index: EvidenceIndex,
    *,
    dataset: str,
    perturbation: str,
    excluded: Mapping[str, np.ndarray],
    gene_union_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    ev = index.evidence
    cols = index.gene_cols[gene_union_ids]
    have = cols >= 0
    stats = _zero_stats(int(cols.shape[0]))
    by_ds = {
        k: np.zeros((ev.n_donor_datasets, int(cols.shape[0])), dtype=np.int64)
        for k in PER_DATASET_ARRAYS
    }
    row = index.pert_rows.get(perturbation, -1)
    if row >= 0 and have.any():
        take = cols[have]
        for k in GLOBAL_ARRAYS:
            stats[k][have] = np.asarray(ev.arrays[k][row], dtype=stats[k].dtype)[take]
        for k in PER_DATASET_ARRAYS:
            by_ds[k][:, have] = np.asarray(ev.by_ds[k][:, row, :], dtype=np.int64)[:, take]
        _subtract_donors(stats, by_ds, excluded, index.ds_slot.get(dataset, -1))
    return _derive_blocks(stats, by_ds, ev.scales)


def _context_blocks(
    index: EvidenceIndex,
    *,
    dataset: str,
    context: str,
    excluded: Mapping[str, np.ndarray],
    gene_union_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    stats = _zero_stats(len(gene_union_ids))
    shard = index.evidence.shards.get(dataset)
    if shard is None or context not in shard.context_to_index:
        return _derive_context_blocks(stats, index.evidence.scales)
    cols = index.context_gene_cols[dataset][gene_union_ids]
    covered = cols >= 0
    row = shard.context_to_index[context]
    for key in GLOBAL_ARRAYS:
        stats[key][covered] = shard.totals[key][row, cols[covered]]
        stats[key] -= excluded[key]
    if any(np.any(stats[key] < 0) for key in COUNT_ARRAYS):
        raise ValueError("evidence: negative context count after excluding the query")
    return _derive_context_blocks(stats, index.evidence.scales)


def serve_evidence(
    index: EvidenceIndex,
    *,
    dataset: str,
    context: str,
    perturbation: str,
    gene_union_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """The 7 EVIDENCE_KEYS float32 blocks on the row's local axis, own records removed."""
    ids = np.asarray(gene_union_ids, dtype=np.int64)
    excluded = _query_response_stats(
        index, dataset=dataset, context=context, perturbation=perturbation, gene_union_ids=ids
    )
    blocks = _donor_blocks(
        index, dataset=dataset, perturbation=perturbation, excluded=excluded, gene_union_ids=ids
    )
    blocks.update(
        _context_blocks(
            index, dataset=dataset, context=context, excluded=excluded, gene_union_ids=ids
        )
    )
    return {key: blocks[key] for key in EVIDENCE_KEYS}
