"""On-disk format of a knowledge source.

A source dir holds ``meta.json`` (``SourceMeta``) and ``embeddings.npy``. The token layout adds
``offsets.npy``, and text sources add ``descriptions.json``. Readers dispatch on ``meta.json``
only, never on file names.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from omegaconf import OmegaConf
from pydantic import field_validator

from pie.utils import StrictModel, atomic_dir, write_json

FORMAT_VERSION = 1
META = "meta.json"
EMBEDDINGS = "embeddings.npy"  # dense (N, D) or token (T, D)
OFFSETS = "offsets.npy"  # token layout only: (N + 1,) int64, key i = rows offsets[i]:offsets[i+1]
DESCRIPTIONS = "descriptions.json"  # text sources only: {key: text}, key order = row order
ALIASES_FILE = "aliases.yaml"  # optional: {key: canonical key}, used only on a direct miss
# The reviewed aliases per source (multi-source format); write_source ships them with each source.
CURATED_ALIASES_FILE = Path(__file__).resolve().parent / "curated_aliases.yaml"
SOURCE_NAMES: tuple[str, ...] = (
    "context_text",
    "perturbation_text",
    "gene_text",
    "ncbi_text",
    "esm2",
    "string_space",
    "depmap_gene_effect",
    "smiles",
    "l1000_tas",
    "prism_secondary",
    "jump_morphology",
)

_ON_CONFLICT = ("error", "keep-prior", "replace")


class SourceMeta(StrictModel):
    """meta.json of a source dir. The loader dispatches on this, never on file names."""

    format_version: int
    name: str
    layout: Literal["dense", "token"]
    index: Literal["pert", "context", "gene"]
    keys: list[str]
    dim: int
    dtype: Literal["float16", "float32"]
    provenance: dict[str, Any]

    @field_validator("format_version")
    @classmethod
    def check_format_version(cls, value: int) -> int:
        if value != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version {value}; expected {FORMAT_VERSION}")
        return value

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        if value not in SOURCE_NAMES:
            raise ValueError(f"unknown source name {value!r}; expected one of {SOURCE_NAMES}")
        return value

    @field_validator("keys")
    @classmethod
    def check_keys(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("keys must not be empty")
        if any(not key for key in value):
            raise ValueError("keys must be non-empty strings")
        duplicates = sorted(key for key, count in Counter(value).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate keys: {duplicates[:10]}")
        return value

    @field_validator("dim")
    @classmethod
    def check_dim(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"dim must be positive, got {value}")
        return value


class DescriptionConflictError(ValueError):
    """A key exists in the prior with a different text and on_conflict='error'."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        shown = ", ".join(keys[:10])
        more = f" (+{len(keys) - 10} more)" if len(keys) > 10 else ""
        super().__init__(
            f"{len(keys)} keys have a text that differs from the prior: {shown}{more}; "
            "use on_conflict='keep-prior' or 'replace'"
        )


@dataclass
class Source:
    """An opened source dir. ``embeddings`` is memory-mapped in the dtype given by meta."""

    path: Path
    meta: SourceMeta
    embeddings: np.ndarray
    offsets: np.ndarray | None
    key_to_row: dict[str, int]

    def tokens(self, key: str) -> np.ndarray:
        """(m, D) view for key; m == 1 for dense. KeyError when absent."""
        row = self.key_to_row[key]
        if self.offsets is None:
            return self.embeddings[row : row + 1]
        return self.embeddings[int(self.offsets[row]) : int(self.offsets[row + 1])]


def _check_arrays(meta: SourceMeta, embeddings: np.ndarray, offsets: np.ndarray | None) -> None:
    where = f"source {meta.name!r}"
    if embeddings.ndim != 2:
        raise ValueError(f"{where}: embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.dtype != np.dtype(meta.dtype):
        raise ValueError(
            f"{where}: embeddings dtype {embeddings.dtype} does not match meta dtype {meta.dtype}"
        )
    if embeddings.shape[1] != meta.dim:
        raise ValueError(
            f"{where}: embeddings dim {embeddings.shape[1]} does not match meta dim {meta.dim}"
        )
    n_keys = len(meta.keys)
    if meta.layout == "dense":
        if offsets is not None:
            raise ValueError(f"{where}: dense layout takes no offsets")
        if embeddings.shape[0] != n_keys:
            raise ValueError(
                f"{where}: dense embeddings have {embeddings.shape[0]} rows for {n_keys} keys"
            )
        return
    if offsets is None:
        raise ValueError(f"{where}: token layout requires offsets")
    if offsets.dtype != np.int64 or offsets.shape != (n_keys + 1,):
        raise ValueError(
            f"{where}: token offsets must be int64 of shape ({n_keys + 1},), "
            f"got {offsets.dtype} {offsets.shape}"
        )
    n_tokens = embeddings.shape[0]
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_tokens:
        raise ValueError(
            f"{where}: token offsets must start at 0 and end at {n_tokens} (the token rows), "
            f"got {int(offsets[0])}..{int(offsets[-1])}"
        )
    if np.any(np.diff(offsets) <= 0):
        raise ValueError(f"{where}: token offsets must be strictly increasing")


def _check_descriptions(meta: SourceMeta, descriptions: Mapping[str, object]) -> None:
    if list(descriptions) != meta.keys:
        raise ValueError(
            f"source {meta.name!r}: descriptions keys must equal meta keys in row order"
        )
    bad = [key for key, text in descriptions.items() if not isinstance(text, str)]
    if bad:
        raise ValueError(f"source {meta.name!r}: descriptions must be strings, not for {bad[:10]}")


def _read_meta(path: Path) -> SourceMeta:
    return SourceMeta.model_validate_json((path / META).read_bytes())


def read_source(path: Path) -> Source:
    """Open a source dir: strict meta, mmap embeddings, check shape, dtype and offsets."""
    path = Path(path)
    meta = _read_meta(path)
    embeddings = np.load(path / EMBEDDINGS, mmap_mode="r", allow_pickle=False)
    offsets_path = path / OFFSETS
    offsets: np.ndarray | None = None
    if meta.layout == "token":
        offsets = np.load(offsets_path, allow_pickle=False)
    elif offsets_path.exists():
        raise ValueError(f"source {meta.name!r}: a dense source dir must not contain {OFFSETS}")
    _check_arrays(meta, embeddings, offsets)
    key_to_row = {key: row for row, key in enumerate(meta.keys)}
    return Source(
        path=path, meta=meta, embeddings=embeddings, offsets=offsets, key_to_row=key_to_row
    )


def read_descriptions(path: Path) -> dict[str, str]:
    """Load descriptions.json (insertion order preserved), checked against meta keys."""
    path = Path(path)
    meta = _read_meta(path)
    data = json.loads((path / DESCRIPTIONS).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"source {meta.name!r}: descriptions must be a JSON object")
    _check_descriptions(meta, data)
    return data


def write_source(
    out: Path,
    meta: SourceMeta,
    embeddings: np.ndarray,
    offsets: np.ndarray | None = None,
    descriptions: Mapping[str, str] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate against meta and write the dir atomically via atomic_dir(out, overwrite=...)."""
    out = Path(out)
    _check_arrays(meta, embeddings, offsets)
    if descriptions is not None:
        _check_descriptions(meta, descriptions)
    with atomic_dir(out, overwrite=overwrite) as tmp:
        np.save(tmp / EMBEDDINGS, embeddings, allow_pickle=False)
        if offsets is not None:
            np.save(tmp / OFFSETS, offsets, allow_pickle=False)
        if descriptions is not None:
            write_json(tmp / DESCRIPTIONS, dict(descriptions))
        aliases = curated_aliases(meta.name)
        if aliases:
            OmegaConf.save(OmegaConf.create(aliases), tmp / ALIASES_FILE)
        write_json(tmp / META, meta.model_dump(mode="json"))
    read_source(out)
    return out


def extend_descriptions(
    prior: Mapping[str, str] | None,
    new: Mapping[str, str],
    on_conflict: Literal["error", "keep-prior", "replace"] = "error",
) -> dict[str, str]:
    """Prior entries verbatim in prior order, then keys of `new` not in prior in `new` order.

    A key in both with a different text raises DescriptionConflictError (all such keys listed)
    unless on_conflict is 'keep-prior' (prior text kept) or 'replace' (new text, prior position
    kept).
    """
    if on_conflict not in _ON_CONFLICT:
        raise ValueError(f"on_conflict must be one of {_ON_CONFLICT}, got {on_conflict!r}")
    out = dict(prior) if prior is not None else {}
    conflicts = [key for key, text in new.items() if key in out and out[key] != text]
    if conflicts and on_conflict == "error":
        raise DescriptionConflictError(conflicts)
    for key, text in new.items():
        if key not in out or on_conflict == "replace":
            out[key] = text
    return out


def extend_embeddings(
    prior: Source | None,
    keys: Sequence[str],
    embed_new: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Dense only. Prior rows byte-for-byte in prior order, then new `keys` in `keys` order.

    Calls embed_new once with the new keys (never when there are none); its result must be
    (len(new), prior.meta.dim) in prior's dtype. Returns (embeddings, row_keys).
    """
    if prior is not None and prior.meta.layout != "dense":
        raise ValueError(
            f"extend_embeddings supports dense sources only, not {prior.meta.layout!r}"
        )
    known = set(prior.meta.keys) if prior is not None else set()
    new_keys = list(dict.fromkeys(key for key in keys if key not in known))
    if prior is None and not new_keys:
        raise ValueError("nothing to embed: no prior source and no keys")
    blocks: list[np.ndarray] = []
    row_keys: list[str] = []
    if prior is not None:
        blocks.append(np.asarray(prior.embeddings))
        row_keys.extend(prior.meta.keys)
    if new_keys:
        fresh = np.asarray(embed_new(list(new_keys)))
        if prior is not None:
            expected_shape = (len(new_keys), prior.meta.dim)
            expected_dtype = np.dtype(prior.meta.dtype)
            if fresh.shape != expected_shape or fresh.dtype != expected_dtype:
                raise ValueError(
                    f"embed_new returned {fresh.dtype} {fresh.shape}, "
                    f"expected {expected_dtype} {expected_shape}"
                )
        elif (
            fresh.ndim != 2
            or fresh.shape[0] != len(new_keys)
            or fresh.dtype not in (np.dtype("float16"), np.dtype("float32"))
        ):
            raise ValueError(
                f"embed_new returned {fresh.dtype} {fresh.shape}, "
                f"expected float16/float32 ({len(new_keys)}, D)"
            )
        blocks.append(fresh)
        row_keys.extend(new_keys)
    embeddings = np.concatenate(blocks, axis=0) if len(blocks) > 1 else np.array(blocks[0])
    return embeddings, row_keys


def curated_aliases(name: str) -> dict[str, str]:
    """The reviewed {key: canonical key} table of source `name` ({} when it has none)."""
    raw = OmegaConf.to_container(OmegaConf.load(CURATED_ALIASES_FILE), resolve=True)
    table = raw.get(name) if isinstance(raw, dict) else None
    return {str(k): str(v) for k, v in (table or {}).items()}


def read_source_aliases(path: Path) -> dict[str, str]:
    """<source dir>/aliases.yaml as {key: canonical key}; {} when the source has none."""
    file = Path(path) / ALIASES_FILE
    if not file.is_file():
        return {}
    raw = OmegaConf.to_container(OmegaConf.load(file), resolve=True)
    table = {} if raw is None else raw
    if not isinstance(table, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in table.items()
    ):
        raise ValueError(f"{file}: aliases must map str -> str")
    return dict(table)
