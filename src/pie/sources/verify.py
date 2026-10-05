"""pie-sources mode=verify: key coverage of datasets by sources, and diffs to a reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pie.data.dataset import load_aliases
from pie.data.preprocessed import PreprocessedDir
from pie.sources.contract import (
    DESCRIPTIONS,
    Source,
    read_descriptions,
    read_source,
    read_source_aliases,
)

MISSING_EXAMPLES = 20
TOKEN_SAMPLE = 256


def needed_keys(source: Source, dataset: PreprocessedDir) -> list[str]:
    """The keys a dataset asks of a source: genes of gene_text, else contexts or perturbations."""
    if source.meta.name == "gene_text":
        return list(dataset.genes)
    if source.meta.index == "context":
        return list(dataset.contexts)
    return list(dataset.perts)


def coverage(
    source: Source, aliases: Mapping[str, str], dataset: PreprocessedDir
) -> dict[str, Any]:
    """Direct hits, alias hits (only on a direct miss, target must exist) and misses."""
    table: Mapping[str, str] = {} if source.meta.name == "gene_text" else aliases
    direct = alias = 0
    missing: list[str] = []
    keys = needed_keys(source, dataset)
    for key in keys:
        if key in source.key_to_row:
            direct += 1
        elif table.get(key, "") in source.key_to_row:
            alias += 1
        else:
            missing.append(key)
    return {
        "n": len(keys),
        "direct": direct,
        "alias": alias,
        "missing": len(missing),
        "missing_keys": missing[:MISSING_EXAMPLES],
    }


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of matching rows in float64: 1 when both rows are zero, 0 when only one is."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    out = np.zeros(len(x))
    both = (norm_x > 0) & (norm_y > 0)
    out[both] = (x[both] * y[both]).sum(axis=1) / (norm_x[both] * norm_y[both])
    out[(norm_x == 0) & (norm_y == 0)] = 1.0
    return out


def _equal_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype != b.dtype:
        return np.zeros(len(a), dtype=bool)
    return np.all(np.asarray(a) == np.asarray(b), axis=1)


def _cosine_summary(cosines: np.ndarray) -> dict[str, float | None]:
    if cosines.size == 0:
        return {"cosine_min": None, "cosine_mean": None}
    return {"cosine_min": float(cosines.min()), "cosine_mean": float(cosines.mean())}


def _lengths(source: Source, keys: Sequence[str]) -> np.ndarray:
    assert source.offsets is not None
    rows = np.array([source.key_to_row[key] for key in keys], dtype=np.int64)
    return source.offsets[rows + 1] - source.offsets[rows]


def _compare_tokens(new: Source, reference: Source, common: Sequence[str]) -> dict[str, Any]:
    same = _lengths(new, common) == _lengths(reference, common)
    candidates = [key for key, equal in zip(common, same, strict=True) if equal]
    step = max(1, -(-len(candidates) // TOKEN_SAMPLE))
    sample = candidates[::step]
    cosines: list[np.ndarray] = []
    bitwise = 0
    for key in sample:
        a = np.asarray(new.tokens(key))
        b = np.asarray(reference.tokens(key))
        cosines.append(row_cosines(a, b))
        bitwise += int(bool(_equal_rows(a, b).all()))
    joined = np.concatenate(cosines) if cosines else np.zeros(0)
    return {
        "n_length_equal": int(same.sum()),
        "n_sampled": len(sample),
        "n_sampled_bitwise_equal": bitwise,
        **_cosine_summary(joined),
    }


def _compare_descriptions(
    new: Source, reference: Source, common: Sequence[str]
) -> dict[str, int] | None:
    if not ((new.path / DESCRIPTIONS).exists() and (reference.path / DESCRIPTIONS).exists()):
        return None
    ours = read_descriptions(new.path)
    theirs = read_descriptions(reference.path)
    return {"n_compared": len(common), "n_equal": sum(ours[k] == theirs[k] for k in common)}


def compare_to_reference(new: Source, reference: Source) -> dict[str, Any]:
    """Key agreement and per-key numeric agreement of a source with its reference."""
    common = [key for key in new.meta.keys if key in reference.key_to_row]
    result: dict[str, Any] = {
        "layout": new.meta.layout,
        "n_keys": len(new.meta.keys),
        "n_reference_keys": len(reference.meta.keys),
        "n_common": len(common),
        "keys_equal": new.meta.keys == reference.meta.keys,
    }
    if new.meta.layout != reference.meta.layout:
        result["layout_equal"] = False
        return result
    if new.meta.layout == "dense":
        a = np.asarray(new.embeddings[[new.key_to_row[key] for key in common]])
        b = np.asarray(reference.embeddings[[reference.key_to_row[key] for key in common]])
        result["n_bitwise_equal"] = int(_equal_rows(a, b).sum())
        result.update(_cosine_summary(row_cosines(a, b)))
    else:
        result.update(_compare_tokens(new, reference, common))
    result["descriptions_equal"] = _compare_descriptions(new, reference, common)
    return result


def verify_sources(
    sources: Mapping[str, Path],
    preprocessed: Sequence[PreprocessedDir],
    aliases_path: Path | None,
    reference_root: Path | None,
) -> dict[str, object]:
    """{"coverage": {source: {dataset: counts}}, "reference": {source: diff}} for every source."""
    aliases = load_aliases(aliases_path) if aliases_path is not None else {}
    covered: dict[str, dict[str, Any]] = {}
    diffs: dict[str, dict[str, Any]] = {}
    for name, path in sources.items():
        source = read_source(Path(path))
        table = {**read_source_aliases(Path(path)), **aliases.get(source.meta.name, {})}
        covered[name] = {d.dataset: coverage(source, table, d) for d in preprocessed}
        if reference_root is not None and (Path(reference_root) / name).exists():
            diffs[name] = compare_to_reference(source, read_source(Path(reference_root) / name))
    return {"coverage": covered, "reference": diffs}
