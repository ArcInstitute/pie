"""Resolve immutable Hub assets into local PIE-format directories under PIE_DATA_ROOT."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote

import numpy as np
from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download

from pie.utils import (
    canonical_json,
    require_env,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
)

log = logging.getLogger(__name__)
AssetKind = Literal["preprocessed", "source", "splits"]
_SHA = re.compile(r"[0-9a-f]{40}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class HubReference:
    repo_id: str
    revision: str
    subdir: str

    @property
    def uri(self) -> str:
        revision = quote(self.revision, safe="")
        suffix = f"/{self.subdir}" if self.subdir else ""
        return f"hf://datasets/{self.repo_id}@{revision}{suffix}"


def parse_hf_reference(value: str) -> HubReference:
    """Explicit dataset URIs only; revisions containing '/' must be percent-encoded."""
    prefix = "hf://datasets/"
    error = f"invalid HF reference {value!r}; expected hf://datasets/owner/repo[@revision]/dir"
    if not value.startswith(prefix) or any(c in value for c in "?#\\"):
        raise ValueError(error)
    parts = value[len(prefix):].rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(error)
    owner, repo_revision, *dirs = parts
    repo, sep, revision = repo_revision.partition("@")
    revision = unquote(revision) if sep else "main"
    dirs = [unquote(p) for p in dirs]
    if (
        not _COMPONENT.fullmatch(owner)
        or not _COMPONENT.fullmatch(repo)
        or not revision
        or any(c in revision for c in "@?#\\")
        or any(not _COMPONENT.fullmatch(p) for p in dirs)
    ):
        raise ValueError(error)
    return HubReference(f"{owner}/{repo}", revision, "/".join(dirs))


def _root() -> Path:
    root = resolve_path(require_env("PIE_DATA_ROOT")["PIE_DATA_ROOT"]).resolve() / "hf"
    # hf_xet reads this when its download runtime initializes. Keep transfer caches with data.
    os.environ["HF_XET_CACHE"] = str(root / ".xet")
    return root


@contextlib.contextmanager
def _repo_lock(root: Path, repo_id: str) -> Iterator[None]:
    """One lock per repository, shared by processes/ranks using the same data root."""
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    key = sha256_bytes(repo_id.encode())
    log.info("waiting for HF asset lock: %s", repo_id)
    with FileLock(locks / f"{key}.lock"):
        yield


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}


def _pin(ref: HubReference, root: Path) -> HubReference:
    if _SHA.fullmatch(ref.revision):
        return ref
    key = sha256_bytes(f"{ref.repo_id}@{ref.revision}".encode())
    record = root / ".refs" / f"{key}.json"
    if record.is_file():
        try:
            payload = json.loads(record.read_text())
            if not isinstance(payload, dict):
                raise ValueError("revision record must be an object")
            commit = payload["commit"]
            if (
                payload.get("repo_id") != ref.repo_id
                or payload.get("revision") != ref.revision
                or not isinstance(commit, str)
                or not _SHA.fullmatch(commit)
            ):
                raise ValueError("revision record does not match the requested reference")
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                f"invalid HF revision record {record}; use an explicit commit from a saved "
                "run config, or intentionally delete the record to resolve the revision again"
            ) from exc
    else:
        if _offline():
            raise FileNotFoundError(f"HF asset {ref.uri} is not pinned locally (offline mode)")
        try:
            commit = HfApi().repo_info(
                repo_id=ref.repo_id, repo_type="dataset", revision=ref.revision
            ).sha
        except Exception as exc:
            raise RuntimeError(
                f"cannot resolve {ref.uri}: {exc}. Check the repository/revision and HF login."
            ) from exc
        if not isinstance(commit, str) or not _SHA.fullmatch(commit):
            raise ValueError(f"HF returned an invalid commit for {ref.uri}: {commit!r}")
        record.parent.mkdir(parents=True, exist_ok=True)
        write_json(record, {"repo_id": ref.repo_id, "revision": ref.revision, "commit": commit})
    return replace(ref, revision=commit)


def pin_asset_reference(value: str) -> str:
    """Freeze a remote revision once per data root; local strings remain untouched."""
    if not value.startswith("hf://"):
        return value
    ref = parse_hf_reference(value)
    if _SHA.fullmatch(ref.revision):
        return ref.uri
    root = _root()
    with _repo_lock(root, ref.repo_id):
        return _pin(ref, root).uri


def _split_files(path: Path) -> list[Path]:
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".json")
    if not files:
        raise ValueError(f"{path}: no split .json files")
    return files


def _splits_digest(path: Path) -> str:
    return sha256_bytes(canonical_json({p.name: sha256_file(p) for p in _split_files(path)}))


def _key_digest(path: Path, kind: AssetKind) -> str:
    """meta.json's sha256 for the PIE formats; a digest of every split file for splits."""
    return _splits_digest(path) if kind == "splits" else sha256_file(path / "meta.json")


def _validate_asset(path: Path, kind: AssetKind) -> list[Path]:
    """Validate the native format before recording download completion."""
    if kind == "splits":
        from pie.data.splits import load_split

        files = _split_files(path)
        for f in files:
            load_split(f)
        return files
    if kind == "preprocessed":
        from pie.data.preprocessed import ARRAY_DTYPES, CONTEXTS_FILE, PreprocessedDir
        from pie.sources.text.context_file import load_context_file

        meta = PreprocessedDir.open(path).meta
        files = [path / "meta.json"]
        for name in meta.array_sha256:
            array = np.load(path / name, mmap_mode="r", allow_pickle=False)
            if array.dtype != ARRAY_DTYPES[name]:
                raise ValueError(
                    f"{path / name}: dtype {array.dtype}, expected {ARRAY_DTYPES[name]}"
                )
            shape = (
                (meta.num_contexts, meta.num_genes) if name == "ctrl_means.npy"
                else (meta.num_rows,) if name in {"ctx_ids.npy", "pert_ids.npy"}
                else (meta.num_rows, meta.num_genes)
            )
            if array.shape != shape:
                raise ValueError(f"{path / name}: shape {array.shape}, expected {shape}")
            files.append(path / name)
        contexts = path / CONTEXTS_FILE
        if contexts.is_file():
            load_context_file(contexts)
            files.append(contexts)
        return files
    from pie.sources.contract import (
        ALIASES_FILE,
        read_descriptions,
        read_source,
        read_source_aliases,
    )

    source = read_source(path)
    files = [path / "meta.json", path / "embeddings.npy"]
    if source.meta.layout == "token":
        files.append(path / "offsets.npy")
    descriptions = path / "descriptions.json"
    if descriptions.is_file():
        read_descriptions(path)
        files.append(descriptions)
    aliases = path / ALIASES_FILE
    if aliases.is_file():
        read_source_aliases(path)
        files.append(aliases)
    return files


def _complete(manifest: Path, path: Path, ref: HubReference, kind: AssetKind) -> bool:
    try:
        payload = json.loads(manifest.read_text())
        return (
            payload["reference"] == ref.uri
            and payload["kind"] == kind
            and bool(payload["sizes"])
            and all((path / name).stat().st_size == size for name, size in payload["sizes"].items())
            and _key_digest(path, kind) == payload["meta_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def resolve_asset(value: str | Path, *, kind: AssetKind) -> Path:
    """Return a local asset, downloading only its directory if needed.

    Complete pinned downloads are usable without network access. A repository lock and
    completion manifest prevent concurrent consumers from opening partially downloaded files.
    HF's local-directory metadata is retained for recovering interrupted transfers.
    """
    text = str(value)
    if not text.startswith("hf://"):
        return resolve_path(value)
    ref = parse_hf_reference(text)
    root = _root()
    with _repo_lock(root, ref.repo_id):
        ref = _pin(ref, root)
        repo_dir = root / "datasets" / ref.repo_id / ref.revision
        path = repo_dir / ref.subdir
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"HF reference {text!r} resolves outside PIE_DATA_ROOT")
        key = sha256_bytes(f"{ref.uri}:{kind}".encode())
        manifest = repo_dir / ".pie-assets" / f"{key}.json"
        if _complete(manifest, path, ref, kind):
            log.info("using downloaded HF asset: %s -> %s", ref.uri, path)
            return path
        if _offline():
            raise FileNotFoundError(
                f"HF asset {ref.uri} is missing or incomplete at {path} (offline mode); "
                "run once with HF_HUB_OFFLINE unset to download it"
            )
        log.info("downloading HF asset: %s -> %s", ref.uri, path)
        try:
            snapshot_download(
                repo_id=ref.repo_id,
                repo_type="dataset",
                revision=ref.revision,
                local_dir=repo_dir,
                cache_dir=root / ".hub",
                allow_patterns=f"{ref.subdir}/*" if ref.subdir else "*",
                force_download=manifest.exists(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot download {ref.uri} to {path}: {exc}. "
                "Check HF login/HF_TOKEN and network access; rerun to recover the download."
            ) from exc
        files = _validate_asset(path, kind)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {
            "reference": ref.uri,
            "kind": kind,
            "meta_sha256": _key_digest(path, kind),
            "sizes": {p.name: p.stat().st_size for p in files},
        })
        return path


def resolve_split_file(value: str | Path) -> Path:
    """One split file: a local path, or hf://datasets/<owner>/<repo>@<rev>/<dir>/<name>.json."""
    text = str(value)
    if not text.startswith("hf://"):
        return resolve_path(value)
    ref = parse_hf_reference(text)
    parent, _, name = ref.subdir.rpartition("/")
    if not name.endswith(".json"):
        raise ValueError(f"{text!r} must name a .json split file inside a split dir")
    directory = resolve_asset(replace(ref, subdir=parent).uri, kind="splits")
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"{text}: {name} is not in the split dir {directory}")
    return path
