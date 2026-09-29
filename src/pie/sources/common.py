"""Shared helpers of the source tools: cached downloads and provenance records."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pie
from pie.utils import sha256_file

if TYPE_CHECKING:
    import requests

RECORDED_HEADERS: tuple[str, ...] = ("last-modified", "etag", "x-uniprot-release")
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS: tuple[int, int] = (30, 600)


def http_session(session: requests.Session | None = None) -> requests.Session:
    """Return `session`, or a new ``requests.Session`` (requests is imported only here)."""
    if session is not None:
        return session
    import requests

    return requests.Session()


def headers_path(dest: Path) -> Path:
    """Sidecar file holding the recorded response headers of a download."""
    return dest.with_name(dest.name + ".headers.json")


def _check_sha256(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name}: sha256 {actual} does not match the pinned {expected}")


def download_file(
    url: str,
    dest: Path,
    *,
    session: requests.Session | None = None,
    params: Mapping[str, str] | None = None,
    sha256: str | None = None,
    offline: bool = False,
) -> Path:
    """Download `url` to `dest` unless `dest` exists; with `sha256`, verify the file either way.

    The body streams into a temporary sibling that is renamed only after the checksum passed,
    so an interrupted or rejected download never leaves a partial `dest`. With `offline`, a
    missing `dest` raises FileNotFoundError and no request is made.
    """
    dest = Path(dest)
    if dest.exists():
        _check_sha256(dest, sha256)
        return dest
    if offline:
        raise FileNotFoundError(f"offline run: {dest} is not in the cache (url {url})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part-{os.getpid()}")
    try:
        with http_session(session).get(
            url, params=params, stream=True, timeout=TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    handle.write(chunk)
            recorded = {
                name: str(response.headers[name])
                for name in RECORDED_HEADERS
                if name in response.headers
            }
        _check_sha256(part, sha256)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    headers_path(dest).write_text(json.dumps(recorded, sort_keys=True) + "\n")
    os.replace(part, dest)
    return dest


def input_record(path: Path, url: str | None, release: str | None = None) -> dict[str, Any]:
    """Provenance of one input file: url, sha256 and release.

    Without an explicit release, the UniProt release header or the Last-Modified header recorded
    at download time is used; a file that was not downloaded here has release None.
    """
    path = Path(path)
    sidecar = headers_path(path)
    headers = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return {
        "url": url,
        "sha256": sha256_file(path),
        "release": release or headers.get("x-uniprot-release") or headers.get("last-modified"),
    }


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def environment_record(device: str) -> dict[str, str | None]:
    """Library versions and the accelerator an embedding step ran on."""
    import torch

    record: dict[str, str | None] = {
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        record["accelerator"] = torch.cuda.get_device_name(0)
    return record


def base_provenance(
    inputs: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any], **extra: Any
) -> dict[str, Any]:
    """The provenance block every tool writes: tool version, inputs, params, then extras."""
    return {
        "tool_version": pie.__version__,
        "inputs": {name: dict(record) for name, record in inputs.items()},
        "params": dict(params),
        **extra,
    }
