"""Split files: strict JSON objects {"<dataset>.<context>": [perturbation, ...]}."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from pie.data.preprocessed import PreprocessedDir

Split = dict[str, list[str]]
SPLIT_FILES: tuple[str, ...] = ("train.json", "val.json")
_MAX_LISTED = 20


class SplitKeyError(ValueError):
    """Unknown dataset, context or (context, perturbation) in a split; lists every offender."""

    def __init__(self, offenders: list[str]) -> None:
        self.offenders = offenders
        shown = "; ".join(offenders[:_MAX_LISTED])
        extra = len(offenders) - _MAX_LISTED
        more = f" (and {extra} more)" if extra > 0 else ""
        super().__init__(f"{len(offenders)} unresolved split entries: {shown}{more}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate split key {key!r}")
        out[key] = value
    return out


def split_key(dataset: str, context: str) -> str:
    """'<dataset>.<context>'."""
    if not dataset or "." in dataset or not context:
        raise ValueError(f"invalid split key parts: dataset={dataset!r}, context={context!r}")
    return f"{dataset}.{context}"


def parse_split_key(key: str) -> tuple[str, str]:
    """Split on the first '.' into (dataset, context)."""
    dataset, sep, context = key.partition(".")
    if not sep or not dataset or not context:
        raise ValueError(f"split key {key!r} is not '<dataset>.<context>'")
    return dataset, context


def load_split(path: Path) -> Split:
    """Strict load: a JSON object whose keys contain a '.', values are lists of unique str."""
    raw = json.loads(Path(path).read_text(), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a split file must hold a JSON object")
    split: Split = {}
    for key, perts in raw.items():
        try:
            parse_split_key(key)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if not isinstance(perts, list) or not all(isinstance(p, str) for p in perts):
            raise ValueError(f"{path}: value of {key!r} must be a list of strings")
        if len(set(perts)) != len(perts):
            raise ValueError(f"{path}: duplicate perturbations under {key!r}")
        split[key] = list(perts)
    return split


def split_pairs(split: Split) -> set[tuple[str, str, str]]:
    """{(dataset, context, perturbation)} for every entry."""
    pairs: set[tuple[str, str, str]] = set()
    for key, perts in split.items():
        dataset, context = parse_split_key(key)
        pairs.update((dataset, context, pert) for pert in perts)
    return pairs


def resolve_split(split: Split, dirs: Sequence[PreprocessedDir]) -> dict[str, np.ndarray]:
    """Row indices per dataset name (sorted int64; empty array for a dataset with no key).

    Raises SplitKeyError if a key names an unknown dataset/context, or a pert with no row.
    """
    by_name = {d.dataset: d for d in dirs}
    if len(by_name) != len(dirs):
        names = [d.dataset for d in dirs]
        raise ValueError(f"duplicate dataset names among preprocessed dirs: {names}")
    rows: dict[str, list[int]] = {d.dataset: [] for d in dirs}
    offenders: list[str] = []
    for key, perts in split.items():
        dataset, context = parse_split_key(key)
        pdir = by_name.get(dataset)
        if pdir is None:
            offenders.append(f"{key}: unknown dataset {dataset!r}")
            continue
        if pdir.controls_only:
            offenders.append(f"{key}: dataset {dataset!r} is controls-only")
            continue
        if context not in pdir.meta.context_to_id:
            offenders.append(f"{key}: unknown context {context!r}")
            continue
        index = pdir.row_index()
        for pert in perts:
            row = index.get((context, pert))
            if row is None:
                offenders.append(f"{key}: no row for perturbation {pert!r}")
            else:
                rows[dataset].append(row)
    if offenders:
        raise SplitKeyError(offenders)
    return {name: np.array(sorted(r), dtype=np.int64) for name, r in rows.items()}


def check_disjoint(splits: Mapping[str, Split]) -> None:
    """Raise ValueError if any (dataset, context, pert) appears in two of the named splits."""
    pairs = {name: split_pairs(split) for name, split in splits.items()}
    for (name_a, pairs_a), (name_b, pairs_b) in combinations(pairs.items(), 2):
        shared = pairs_a & pairs_b
        if shared:
            example = ", ".join(".".join(t) for t in sorted(shared)[:5])
            raise ValueError(
                f"splits {name_a!r} and {name_b!r} share {len(shared)} entries, e.g. {example}"
            )
