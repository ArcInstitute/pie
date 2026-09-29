"""pie-prep: DE label tables plus expression h5ads -> one preprocessed dataset dir.

A label table (CSV or parquet) has one row per (context, perturbation, gene) with an FDR and a
fold change. The expression h5ads give the pseudobulk means behind delta_p and ctrl_means.
"""

from __future__ import annotations

import ast
import glob
import logging
import math
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import scipy.sparse as sp
from anndata.io import read_elem
from numpy.typing import NDArray

from pie import __version__
from pie.data import preprocessed as pp
from pie.prep.config import PrepConfig
from pie.utils import resolve_path

log = logging.getLogger(__name__)


def resolve_files(pattern: str, what: str) -> list[str]:
    """Sorted, de-duplicated glob matches; FileNotFoundError when nothing matches."""
    matched = sorted(set(glob.glob(pattern)))
    if not matched:
        raise FileNotFoundError(f"no {what} files matched {pattern!r}")
    return matched


def read_gene_list(path: Path) -> list[str]:
    """One gene symbol per line; blank lines are skipped, duplicates are an error."""
    genes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not genes:
        raise ValueError(f"{path}: gene list is empty")
    duplicates = sorted(g for g, n in Counter(genes).items() if n > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate genes {duplicates[:5]}")
    return genes


def parse_drug_dose(value: str) -> str:
    """Parse ``[('<drug>', <conc>, '<unit>')]`` into ``<drug>_<conc><unit>``.

    Values that do not start with ``[(`` (such as a bare control label) pass through
    unchanged; malformed tuple-shaped values raise ValueError.
    """
    if not value.startswith("[("):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"cannot parse drug-dose value {value!r}: {exc}") from exc
    if not (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], tuple)
        and len(parsed[0]) == 3
    ):
        raise ValueError(f"expected [('<drug>', <conc>, '<unit>')], got {value!r}")
    drug, conc, unit = parsed[0]
    return f"{drug}_{conc}{unit}"


def normalize_obs(
    ctx_arr: np.ndarray,
    pert_arr: np.ndarray,
    *,
    context_map: dict[str, str] | None,
    pert_format: Literal["gene", "drug_dose"],
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map h5ad obs values onto the label encoding (context map, drug-dose parsing).

    Each transform works on the unique values and maps them back. An unmapped context or an
    unparseable drug-dose value raises ValueError naming `source`.
    """
    if context_map is not None:
        ctx_series = pd.Series(ctx_arr).astype(str)
        mapped_ctx = ctx_series.map(context_map)
        missing = mapped_ctx.isna()
        if missing.any():
            first_missing = ctx_series[missing].iloc[0]
            raise ValueError(
                f"{source}: h5ad context value {first_missing!r} is not in the context map"
            )
        new_ctx = mapped_ctx.to_numpy()
    else:
        new_ctx = pd.Series(ctx_arr).astype(str).to_numpy()

    if pert_format == "drug_dose":
        pert_series = pd.Series(pert_arr).astype(str)
        pert_map: dict[str, str] = {}
        for value in pert_series.unique():
            try:
                pert_map[value] = parse_drug_dose(value)
            except ValueError as exc:
                raise ValueError(f"{source}: {exc}") from exc
        new_pert = pert_series.map(pert_map).to_numpy()
    else:
        new_pert = pd.Series(pert_arr).astype(str).to_numpy()
    return new_ctx, new_pert


def normalize_label_frame(
    frame: pd.DataFrame,
    *,
    context_case: Literal["asis", "lower"],
    context_map: dict[str, str] | None,
    pert_format: Literal["plain", "drug_dose"],
    source: str,
) -> pd.DataFrame:
    """Rewrite label contexts (lower-case, then map) and perturbations (drug-dose parsing).

    Works in place on the `context` / `pert` columns that the frame has. A context absent from
    the map or an unparseable drug-dose value raises ValueError naming `source`.
    """
    if "context" in frame.columns and (context_case == "lower" or context_map is not None):
        ctx = frame["context"].astype(str)
        if context_case == "lower":
            ctx = ctx.str.lower()
        if context_map is not None:
            mapped = ctx.map(context_map)
            missing = mapped.isna()
            if missing.any():
                raise ValueError(
                    f"{source}: label context {ctx[missing].iloc[0]!r} is not in the "
                    "label context map"
                )
            ctx = mapped
        frame["context"] = ctx
    if "pert" in frame.columns and pert_format == "drug_dose":
        pert = frame["pert"].astype(str)
        pert_map: dict[str, str] = {}
        for value in pert.unique():
            try:
                pert_map[value] = parse_drug_dose(value)
            except ValueError as exc:
                raise ValueError(f"{source}: {exc}") from exc
        frame["pert"] = pert.map(pert_map)
    return frame


def fold_change_and_lfc(
    values: np.ndarray, space: Literal["linear", "log2", "ln"]
) -> tuple[np.ndarray, np.ndarray]:
    """(linear fold change, log2 fold change), both float64, from values in `space`."""
    values = np.asarray(values, dtype=np.float64)
    if space == "linear":
        with np.errstate(divide="ignore", invalid="ignore"):
            return values, np.log2(values)
    if space == "log2":
        return np.exp2(values), values
    if space == "ln":
        return np.exp(values), values / math.log(2.0)
    raise ValueError(f"unknown fold-change space {space!r}")


class H5CSRReader:
    """Direct h5py reader for CSR-format X matrices in h5ad files."""

    def __init__(self, h5_file: h5py.File, n_cols: int) -> None:
        self._data = h5_file["X"]["data"]
        self._indices = h5_file["X"]["indices"]
        self._indptr = h5_file["X"]["indptr"]
        self._n_cols = n_cols

    def read_rows(self, row_indices: NDArray[np.intp]) -> sp.csr_matrix:
        """Read arbitrary rows by index (returned in sorted order) as one CSR matrix."""
        sorted_idx = np.sort(row_indices)
        n_rows = len(sorted_idx)
        row_min = int(sorted_idx[0])
        row_max = int(sorted_idx[-1])

        indptr_slab = np.array(self._indptr[row_min : row_max + 2], dtype=np.int64)

        local_indptr = np.zeros(n_rows + 1, dtype=np.int64)
        data_ranges: list[tuple[int, int]] = []
        for i, row in enumerate(sorted_idx):
            ri = int(row) - row_min
            start_ptr = indptr_slab[ri]
            end_ptr = indptr_slab[ri + 1]
            length = end_ptr - start_ptr
            local_indptr[i + 1] = local_indptr[i] + length
            data_ranges.append((int(start_ptr), int(end_ptr)))

        total_nnz = int(local_indptr[-1])
        if total_nnz == 0:
            return sp.csr_matrix((n_rows, self._n_cols), dtype=np.float32)

        all_data = np.empty(total_nnz, dtype=np.float32)
        all_indices = np.empty(total_nnz, dtype=np.int32)

        offset = 0
        i = 0
        while i < len(data_ranges):
            start, end = data_ranges[i]
            j = i + 1
            while j < len(data_ranges) and data_ranges[j][0] == end:
                end = data_ranges[j][1]
                j += 1
            length = end - start
            if length > 0:
                all_data[offset : offset + length] = self._data[start:end]
                all_indices[offset : offset + length] = self._indices[start:end]
            offset += length
            i = j

        return sp.csr_matrix(
            (all_data, all_indices, local_indptr),
            shape=(n_rows, self._n_cols),
        )


class H5DenseReader:
    """Row reader for h5ads whose X is a plain dense 2-D array."""

    def __init__(self, x: h5py.Dataset, n_cols: int) -> None:
        self._x = x
        self._n_cols = n_cols

    def read_rows(self, row_indices: NDArray[np.intp]) -> NDArray[np.float32]:
        """Read arbitrary rows by index, returned sorted to match H5CSRReader."""
        sorted_idx = np.sort(np.asarray(row_indices, dtype=np.intp))
        if sorted_idx.size == 0:
            return np.zeros((0, self._n_cols), dtype=np.float32)
        if np.any(sorted_idx[1:] == sorted_idx[:-1]):
            raise ValueError("Duplicate row indices are not supported by H5DenseReader.")
        # A contiguous span reads as one hyperslab; scattered rows need the point-selection
        # path, which h5py only accepts as a plain int list.
        row_min, row_max = int(sorted_idx[0]), int(sorted_idx[-1])
        if row_max - row_min + 1 == sorted_idx.size:
            block = self._x[row_min : row_max + 1]
        else:
            block = self._x[sorted_idx.tolist()]
        return np.asarray(block, dtype=np.float32)


def open_x_reader(h5_file: h5py.File, n_cols: int) -> H5CSRReader | H5DenseReader:
    """Row reader matching the on-disk encoding of ``h5_file["X"]``; CSC raises ValueError."""
    x = h5_file["X"]
    if isinstance(x, h5py.Dataset):
        return H5DenseReader(x, n_cols)
    encoding = x.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding == "csc_matrix":
        raise ValueError(
            "h5ad X is csc_matrix-encoded; only csr_matrix and dense arrays are supported. "
            "Convert X to CSR before pseudobulking."
        )
    return H5CSRReader(h5_file, n_cols)


def h5ad_var_names(h5ad_files: Sequence[str]) -> list[str]:
    """First-seen union of the var names of the files, in file order."""
    seen: dict[str, None] = {}
    for path in h5ad_files:
        with h5py.File(path, "r") as h5:
            var = read_elem(h5["var"])
        if not isinstance(var, pd.DataFrame):
            raise TypeError(f"{path}: var is not a DataFrame")
        for gene in var.index.astype(str):
            seen.setdefault(gene, None)
    return list(seen)


_CSV_BLOCK_SIZE = 256 * 1024 * 1024  # 256 MiB per CSV read block
_PARQUET_BATCH_ROWS = 1 << 20
_INIT_ROW_CAPACITY = 4096  # initial per-file row capacity, doubled on demand
_VALUE_COLUMNS = ("fdr", "fold_change")


def _column_mapping(cfg: PrepConfig) -> dict[str, str]:
    """Label-table column -> canonical name (context, pert, gene_symbol, fdr, fold_change)."""
    mapping = {
        cfg.label.pert_col: "pert",
        cfg.label.gene_col: "gene_symbol",
        cfg.label.fdr_col: "fdr",
        cfg.label.fc_col: "fold_change",
    }
    if cfg.label.context_from == "column":
        mapping = {str(cfg.label.context_col): "context", **mapping}
    expected = 5 if cfg.label.context_from == "column" else 4
    if len(mapping) != expected:
        raise ValueError("the label.* column settings must name distinct columns")
    return mapping


def _file_context(path: str, context_from: str) -> str | None:
    if context_from == "stem":
        return Path(path).stem
    if context_from == "parent":
        return Path(path).parent.name
    return None


def iter_label_frames(
    path: str, cfg: PrepConfig, columns: Sequence[str] | None = None
) -> Iterator[pd.DataFrame]:
    """Chunks of one label table (CSV or parquet) with canonical column names.

    `columns` restricts the canonical columns read (default: all). With
    label.context_from=stem|parent the context column is the file's stem or parent dir name.
    """
    mapping = _column_mapping(cfg)
    if columns is not None:
        mapping = {src: dst for src, dst in mapping.items() if dst in columns}
    sources = list(mapping)
    if Path(path).suffix.lower() in {".parquet", ".pq"}:
        batches = pq.ParquetFile(path).iter_batches(
            batch_size=_PARQUET_BATCH_ROWS, columns=sources
        )
    else:
        types = {
            src: pa.float64() if dst in _VALUE_COLUMNS else pa.string()
            for src, dst in mapping.items()
        }
        batches = pcsv.open_csv(
            path,
            read_options=pcsv.ReadOptions(block_size=_CSV_BLOCK_SIZE),
            convert_options=pcsv.ConvertOptions(column_types=types, include_columns=sources),
        )
    context = _file_context(path, cfg.label.context_from)
    for batch in batches:
        frame = batch.to_pandas().rename(columns=mapping)
        if context is not None and (columns is None or "context" in columns):
            frame["context"] = context
        yield normalize_label_frame(
            frame,
            context_case=cfg.label.context_case,
            context_map=cfg.label.context_map,
            pert_format=cfg.label.pert_format,
            source=path,
        )


def label_gene_set(label_files: Sequence[str], cfg: PrepConfig) -> set[str]:
    """Every gene symbol named by the label tables."""
    genes: set[str] = set()
    for path in label_files:
        for frame in iter_label_frames(path, cfg, columns=("gene_symbol",)):
            genes.update(frame["gene_symbol"].dropna().astype(str).unique().tolist())
    return genes


def default_gene_axis(
    label_files: Sequence[str], h5ad_files: Sequence[str], cfg: PrepConfig
) -> list[str]:
    """h5ad var order (first-seen union over files) restricted to the label genes; never sorted."""
    in_labels = label_gene_set(label_files, cfg)
    axis = [g for g in h5ad_var_names(h5ad_files) if g in in_labels]
    if not axis:
        raise ValueError("no label gene is present in the h5ad var names")
    n_absent = len(in_labels) - len(axis)
    if n_absent:
        log.warning("%d label genes are absent from every h5ad and are dropped", n_absent)
    return axis


@dataclass
class LabelArrays:
    """Dense label arrays on one gene axis; rows sorted by (context, perturbation)."""

    keys: list[tuple[str, str]]
    fold_changes: np.ndarray  # (N, G) float32, 0 where untested
    fdr: np.ndarray  # (N, G) float32, 1 where untested
    tested: np.ndarray  # (N, G) bool
    lfc_true: np.ndarray  # (N, G) float64, NaN where untested


def _empty_rows(
    capacity: int, n_genes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((capacity, n_genes), dtype=np.float32),
        np.ones((capacity, n_genes), dtype=np.float32),
        np.zeros((capacity, n_genes), dtype=bool),
        np.full((capacity, n_genes), np.nan, dtype=np.float64),
    )


def _grow(
    fc: np.ndarray, fdr: np.ndarray, tested: np.ndarray, lfc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Double the row capacity, keeping the existing rows."""
    grown = _empty_rows(2 * fc.shape[0], fc.shape[1])
    for dst, src in zip(grown, (fc, fdr, tested, lfc), strict=True):
        dst[: src.shape[0]] = src
    return grown


def _stack_sorted(chunks: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    stacked = np.concatenate(chunks, axis=0)
    chunks.clear()
    return stacked[order]


def read_labels(label_files: Sequence[str], cfg: PrepConfig, genes: Sequence[str]) -> LabelArrays:
    """Stream every label table into dense arrays on `genes`; rows sorted by (context, pert).

    Label genes off the axis are dropped. A (context, perturbation) pair found in two files is
    an error. Within a file a repeated (context, perturbation, gene) keeps the last value.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    fc_chunks: list[np.ndarray] = []
    fdr_chunks: list[np.ndarray] = []
    tested_chunks: list[np.ndarray] = []
    lfc_chunks: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    owner: dict[tuple[str, str], str] = {}
    for path in label_files:
        log.info("reading labels %s", path)
        local: dict[tuple[str, str], int] = {}
        fc, fdr, tested, lfc = _empty_rows(_INIT_ROW_CAPACITY, n_genes)
        for chunk in iter_label_frames(path, cfg):
            chunk["gene_idx"] = chunk["gene_symbol"].map(gene_to_idx)
            frame = chunk[chunk["gene_idx"].notna()].copy()
            frame["gene_idx"] = frame["gene_idx"].astype(int)
            for (ctx, pert), group in frame.groupby(["context", "pert"]):
                key = (str(ctx), str(pert))
                if key not in local:
                    local[key] = len(local)
                row = local[key]
                if row >= fc.shape[0]:
                    fc, fdr, tested, lfc = _grow(fc, fdr, tested, lfc)
                gidx = np.asarray(group["gene_idx"])
                fc_linear, lfc_values = fold_change_and_lfc(
                    np.asarray(group["fold_change"], dtype=np.float64), cfg.label.fc_space
                )
                fc[row, gidx] = fc_linear.astype(np.float32)
                lfc[row, gidx] = lfc_values
                fdr[row, gidx] = np.asarray(group["fdr"], dtype=np.float32)
                tested[row, gidx] = True
        for key in local:
            if key in owner:
                raise ValueError(
                    f"(context, perturbation) {key} appears in both {owner[key]} and {path}"
                )
            owner[key] = path
        n = len(local)
        fc_chunks.append(fc[:n].copy())
        fdr_chunks.append(fdr[:n].copy())
        tested_chunks.append(tested[:n].copy())
        lfc_chunks.append(lfc[:n].copy())
        keys.extend(local)
        del fc, fdr, tested, lfc
    if not keys:
        raise ValueError("no label row names a gene on the gene axis")
    order = np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)
    return LabelArrays(
        keys=[keys[i] for i in order],
        fold_changes=_stack_sorted(fc_chunks, order),
        fdr=_stack_sorted(fdr_chunks, order),
        tested=_stack_sorted(tested_chunks, order),
        lfc_true=_stack_sorted(lfc_chunks, order),
    )


def pseudobulk_h5ad(
    path: str,
    cfg: PrepConfig,
    gene_to_idx: dict[str, int],
    context_map: dict[str, str] | None,
    *,
    controls_only: bool = False,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Mean expression per (context, perturbation) and per context control, on the gene axis.

    Returns (pseudo {(ctx, pert): (G,) float32}, ctrl {ctx: (G,) float32}). Axis genes absent
    from the h5ad are 0; h5ad genes off the axis are dropped. With controls_only only the
    control groups are read and pseudo is empty.
    """
    n_axis = len(gene_to_idx)
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    for col in (cfg.obs.context_col, cfg.obs.pert_col):
        if col not in obs.columns:
            raise ValueError(f"{path}: missing obs column {col!r}")
    mapped = np.array(
        [gene_to_idx.get(g, -1) for g in adata.var_names.tolist()], dtype=np.int64
    )
    keep_mask = mapped >= 0
    keep_idx_h5ad = np.where(keep_mask)[0]
    keep_idx_axis = mapped[keep_mask]
    # np.asarray, not .values: under pandas 3 an astype(str) column is an extension array.
    ctx_arr = np.asarray(obs[cfg.obs.context_col].astype(str))
    pert_arr = np.asarray(obs[cfg.obs.pert_col].astype(str))
    ctx_arr, pert_arr = normalize_obs(
        ctx_arr, pert_arr, context_map=context_map, pert_format=cfg.obs.pert_format, source=path
    )
    n_genes_h5ad = adata.shape[1]
    adata.file.close()

    pseudo: dict[tuple[str, str], np.ndarray] = {}
    ctrl: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        reader = open_x_reader(h5, n_genes_h5ad)
        groups = (
            pd.DataFrame({"ctx": ctx_arr, "pert": pert_arr})
            .groupby(["ctx", "pert"], sort=True, observed=True)
            .indices
        )
        for (ctx_key, pert_key), row_indices in groups.items():
            ctx, pert = str(ctx_key), str(pert_key)
            is_control = pert == cfg.obs.control_label
            if controls_only and not is_control:
                continue
            mat = reader.read_rows(np.asarray(row_indices, dtype=np.intp))
            mean = np.asarray(mat.mean(axis=0), dtype=np.float32).ravel()
            scattered = np.zeros(n_axis, dtype=np.float32)
            scattered[keep_idx_axis] = mean[keep_idx_h5ad]
            if is_control:
                ctrl[ctx] = scattered
            else:
                pseudo[(ctx, pert)] = scattered
    return pseudo, ctrl


def pseudobulk_delta_p(
    h5ad_files: Sequence[str],
    cfg: PrepConfig,
    genes: Sequence[str],
    keys: Sequence[tuple[str, str]],
    contexts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """(delta_p (N, G) float32, ctrl_means (C, G) float32) in label-row and context order.

    delta_p = pseudobulk(ctx, pert) - control mean(ctx), NaN rows (logged per context) where the
    pair has no cells. A label context with no h5ad cells or no control cells is a ValueError.
    A later h5ad replaces an earlier one's group with the same key.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    all_pseudo: dict[tuple[str, str], np.ndarray] = {}
    all_ctrl: dict[str, np.ndarray] = {}
    for path in h5ad_files:
        log.info("pseudobulk %s", path)
        pseudo, ctrl = pseudobulk_h5ad(path, cfg, gene_to_idx, cfg.obs.context_map)
        all_pseudo.update(pseudo)
        all_ctrl.update(ctrl)

    h5ad_contexts = {ctx for ctx, _ in all_pseudo} | set(all_ctrl)
    for ctx in contexts:
        if ctx not in h5ad_contexts:
            raise ValueError(
                f"label context {ctx!r} has no cells in the h5ad files (h5ad contexts: "
                f"{sorted(h5ad_contexts)[:20]}); check obs.context_map and the label.* settings"
            )
        if ctx not in all_ctrl:
            raise ValueError(
                f"label context {ctx!r} has no control cells ({cfg.obs.control_label!r}) in the "
                "h5ad files"
            )

    delta_p = np.full((len(keys), len(genes)), np.nan, dtype=np.float32)
    missing: Counter[str] = Counter()
    for i, (ctx, pert) in enumerate(keys):
        if (ctx, pert) not in all_pseudo:
            missing[ctx] += 1
            continue
        delta_p[i] = all_pseudo[(ctx, pert)] - all_ctrl[ctx]
    if missing:
        log.warning(
            "%d of %d rows have no pseudobulk; their delta_p is NaN (missing rows per context: %s)",
            sum(missing.values()),
            len(keys),
            ", ".join(f"{ctx}: {n}" for ctx, n in sorted(missing.items())),
        )

    ctrl_means = np.stack([all_ctrl[ctx] for ctx in contexts]).astype(np.float32)
    return delta_p, ctrl_means


_NO_ID = frozenset({"", "nan", "none", "control", "non-targeting", "-"})


def pert_ensembl_ids(
    h5ad_files: Sequence[str], cfg: PrepConfig, perts: Collection[str]
) -> dict[str, str]:
    """{perturbation: Ensembl id} for `perts`, from obs[pert_col] and obs[pert_id_col].

    Ids are stripped; blank, nan, none, control, non-targeting and '-' (any case) and ids that do
    not start with ENSG count as no id. Two different ids for one perturbation, in one file or
    across files, raise ValueError. {} when obs.pert_id_col is null. Keys are sorted.
    """
    id_col = cfg.obs.pert_id_col
    if id_col is None:
        return {}
    wanted = set(perts)
    found: dict[str, tuple[str, str]] = {}
    for path in h5ad_files:
        with h5py.File(path, "r") as h5:
            obs = h5["obs"]
            for col in (cfg.obs.pert_col, id_col):
                if col not in obs:
                    raise ValueError(f"{path}: missing obs column {col!r}")
            pert_arr = pd.Series(read_elem(obs[cfg.obs.pert_col])).astype(str)
            id_arr = pd.Series(read_elem(obs[id_col])).astype(str)
        pairs = pd.DataFrame(
            {"pert": pert_arr.to_numpy(), "id": id_arr.to_numpy()}
        ).drop_duplicates()
        for pert, raw in pairs.itertuples(index=False):
            ident = str(raw).strip()
            if pert not in wanted or ident.casefold() in _NO_ID or not ident.startswith("ENSG"):
                continue
            previous = found.get(pert)
            if previous is not None and previous[0] != ident:
                raise ValueError(
                    f"perturbation {pert!r} has conflicting Ensembl ids {previous[0]!r} "
                    f"({previous[1]}) and {ident!r} ({path})"
                )
            found.setdefault(pert, (ident, path))
    return {pert: found[pert][0] for pert in sorted(found)}


def _meta(
    cfg: PrepConfig,
    genes: Sequence[str],
    context_to_id: dict[str, int],
    pert_to_id: dict[str, int],
    num_rows: int,
    pert_ensembl: dict[str, str],
) -> pp.PreprocessedMeta:
    return pp.PreprocessedMeta(
        format_version=pp.FORMAT_VERSION,
        dataset=cfg.name,
        genes=list(genes),
        context_to_id=context_to_id,
        pert_to_id=pert_to_id,
        pert_kind="drug" if cfg.obs.pert_format == "drug_dose" else "gene",
        control_label=cfg.obs.control_label,
        num_rows=num_rows,
        num_genes=len(genes),
        num_contexts=len(context_to_id),
        num_perts=len(pert_to_id),
        controls_only=cfg.controls_only,
        tool_version=__version__,
        array_sha256={},
        pert_ensembl=pert_ensembl,
    )


def _run_controls_only(cfg: PrepConfig, out: Path, h5ad_files: Sequence[str]) -> Path:
    genes = (
        read_gene_list(resolve_path(cfg.genes))
        if cfg.genes is not None
        else h5ad_var_names(h5ad_files)
    )
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    ctrl: dict[str, np.ndarray] = {}
    for path in h5ad_files:
        log.info("control pseudobulk %s", path)
        _, file_ctrl = pseudobulk_h5ad(
            path, cfg, gene_to_idx, cfg.obs.context_map, controls_only=True
        )
        ctrl.update(file_ctrl)
    if not ctrl:
        raise ValueError(f"no cells with control label {cfg.obs.control_label!r} in {cfg.h5ad}")
    contexts = sorted(ctrl)
    ctrl_means = np.stack([ctrl[c] for c in contexts]).astype(np.float32)
    meta = _meta(cfg, genes, {c: i for i, c in enumerate(contexts)}, {}, 0, {})
    return pp.write_preprocessed(out, meta, {pp.CTRL_MEANS: ctrl_means}, overwrite=cfg.overwrite)


def run_prep(cfg: PrepConfig) -> Path:
    """Labels + h5ad -> preprocessed dir (via write_preprocessed); returns the output dir."""
    out = resolve_path(cfg.output_dir)
    if not cfg.overwrite and out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not empty; set overwrite=true to replace it")
    h5ad_files = resolve_files(str(resolve_path(cfg.h5ad)), "h5ad")
    if cfg.controls_only:
        return _run_controls_only(cfg, out, h5ad_files)
    if cfg.labels is None:
        raise ValueError("labels is required unless controls_only=true")
    label_files = resolve_files(str(resolve_path(cfg.labels)), "labels")
    genes = (
        read_gene_list(resolve_path(cfg.genes))
        if cfg.genes is not None
        else default_gene_axis(label_files, h5ad_files, cfg)
    )
    labels = read_labels(label_files, cfg, genes)
    contexts = sorted({ctx for ctx, _ in labels.keys})
    perts = sorted({pert for _, pert in labels.keys})
    context_to_id = {c: i for i, c in enumerate(contexts)}
    pert_to_id = {p: i for i, p in enumerate(perts)}
    pert_ensembl = pert_ensembl_ids(h5ad_files, cfg, pert_to_id)
    delta_p, ctrl_means = pseudobulk_delta_p(h5ad_files, cfg, genes, labels.keys, contexts)
    arrays = {
        pp.FOLD_CHANGES: labels.fold_changes,
        pp.FDR: labels.fdr,
        pp.TESTED: labels.tested,
        pp.LFC_TRUE: labels.lfc_true,
        pp.DELTA_P: delta_p,
        pp.CTRL_MEANS: ctrl_means,
        pp.CTX_IDS: np.array([context_to_id[c] for c, _ in labels.keys], dtype=np.int32),
        pp.PERT_IDS: np.array([pert_to_id[p] for _, p in labels.keys], dtype=np.int32),
    }
    log.info(
        "%s: %d rows x %d genes, %d contexts, %d perturbations",
        cfg.name,
        len(labels.keys),
        len(genes),
        len(contexts),
        len(perts),
    )
    meta = _meta(cfg, genes, context_to_id, pert_to_id, len(labels.keys), pert_ensembl)
    return pp.write_preprocessed(out, meta, arrays, overwrite=cfg.overwrite)
