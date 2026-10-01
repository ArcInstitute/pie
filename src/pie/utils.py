"""Shared helpers: environment loading, config composition, logging, determinism, hashing and
atomic writes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
ENV_ROOTS: tuple[str, ...] = ("PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR")
CUBLAS_WORKSPACE = ":4096:8"

_ASSIGN = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}(?:/(.*))?$")


class StrictModel(BaseModel):
    """Base for every pydantic model in the package: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid")


class MissingEnvError(RuntimeError):
    """Raised when required environment variables are unset or empty."""

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        super().__init__(
            f"missing environment variables: {', '.join(self.names)} (set them in common.sh)"
        )


def _unquote(value: str) -> str:
    """Strip one pair of matching quotes; ValueError on shell syntax the loader does not run.

    Single-quoted values are literal. `$` in an unquoted or double-quoted value, a leading `~`
    and unquoted whitespace (which covers an inline `# comment`) would mean something else
    to bash, so they are rejected instead of loaded verbatim.
    """
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        if "$" in value:
            raise ValueError("'$' expansion is not supported; values are literal")
        return value[1:-1]
    if "$" in value:
        raise ValueError("'$' expansion is not supported; values are literal")
    if value.startswith("~"):
        raise ValueError("'~' expansion is not supported; write the absolute path")
    if any(ch.isspace() for ch in value):
        raise ValueError("unquoted whitespace (or an inline '#' comment) is not supported")
    return value


def load_common_env(path: Path | None = None) -> dict[str, str]:
    """Parse `common.sh` and set the variables that are not already in the environment.

    Accepts `NAME=value` and `export NAME=value`, strips one pair of matching quotes and ignores
    blank lines and `#` comments. Values are literal: `$` expansion, a leading `~` and unquoted
    whitespace or inline comments are rejected. Any other line is a ValueError, and then nothing
    is applied.
    Returns only the variables it set. A missing file returns {}.
    """
    source = REPO_ROOT / "common.sh" if path is None else Path(path)
    if not source.is_file():
        return {}
    parsed: dict[str, str] = {}
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGN.match(line)
        if match is None:
            raise ValueError(f"{source}:{lineno}: expected NAME=value or export NAME=value")
        try:
            parsed[match.group(1)] = _unquote(match.group(2).strip())
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: {exc}") from None
    applied = {name: value for name, value in parsed.items() if name not in os.environ}
    os.environ.update(applied)
    return applied


def require_env(*names: str) -> dict[str, str]:
    """Return {name: value} for every name; raise MissingEnvError listing all unset/empty ones."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise MissingEnvError(missing)
    return {name: os.environ[name] for name in names}


def setup_logging(level: str = "INFO") -> None:
    """Stdlib logging to stderr; keeps existing root handlers and quiets HTTP client loggers."""
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT, stream=sys.stderr)
    logging.getLogger().setLevel(level.upper())
    for name in ("httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def compose_config(
    config_name: str,
    overrides: Sequence[str],
    *,
    groups: Mapping[str, str] | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Compose <config_dir>/<config_name>.yaml with Hydra overrides into resolved plain data.

    `groups` rewrites a short `key=value` override to its config-group path (`dataset=x` ->
    `prep/dataset=x`); any other override passes through. Interpolations are resolved, a `???`
    left anywhere is an error and a `hydra` node is dropped. Hydra's compose API writes no files
    and leaves logging alone.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    aliases = dict(groups or {})
    rewritten: list[str] = []
    for item in overrides:
        key, sep, value = item.partition("=")
        rewritten.append(f"{aliases[key]}={value}" if sep and key in aliases else item)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir or CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=rewritten)
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise TypeError(f"{config_name}.yaml must hold a mapping")
    container.pop("hydra", None)
    return {str(key): value for key, value in container.items()}


def configure_determinism(seed: int) -> None:
    """Set CUBLAS_WORKSPACE_CONFIG (before any CUDA init) and seed every RNG, workers included."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=True)


def env_global_rank(devices_per_node: int) -> int:
    """Global rank before a Trainer exists: RANK, else NODE_RANK * devices + LOCAL_RANK."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return node_rank * devices_per_node + local_rank


def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> bytes:
    """Sorted-key, compact, UTF-8 JSON bytes (the input to every content hash)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@contextlib.contextmanager
def atomic_dir(final: Path, *, overwrite: bool = False) -> Iterator[Path]:
    """Yield a fresh sibling tmp dir and rename it onto `final` when the body succeeds.

    On an exception the tmp dir is removed and the exception re-raised. Without `overwrite`:
    raises FileExistsError before yielding if `final` exists and is not an empty dir, and if a
    non-empty `final` appears while the body runs, the tmp dir is discarded and `final` is kept
    (first writer wins). With `overwrite`, an existing `final` is replaced only after the body
    succeeds.
    """
    final = Path(final)
    if not overwrite and final.exists() and (not final.is_dir() or any(final.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if overwrite and (final.exists() or final.is_symlink()):
        _replace_existing(tmp, final)
        return
    try:
        os.replace(tmp, final)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not final.exists():
            raise


def _replace_existing(tmp: Path, final: Path) -> None:
    """Move `final` aside, rename `tmp` onto it, then drop the old copy (restored on failure)."""
    trash = final.parent / f".{final.name}.old-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    os.replace(final, trash)
    try:
        os.replace(tmp, final)
    except BaseException:
        os.replace(trash, final)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if trash.is_dir() and not trash.is_symlink():
        shutil.rmtree(trash)
    else:
        trash.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to a sibling tmp file and os.replace it onto `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: object) -> None:
    """Atomically write `obj` as indented JSON with a trailing newline."""
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=False) + "\n")


def read_json(path: Path) -> object:
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root_depth(item: tuple[Path, str]) -> int:
    return len(item[0].parts)


def _env_root_prefixes() -> list[tuple[Path, str]]:
    """(root, '${NAME}') for every set env root, deepest root first."""
    roots = [
        (Path(os.path.normpath(value)), f"${{{name}}}")
        for name in ENV_ROOTS
        if (value := os.environ.get(name))
    ]
    return sorted(roots, key=_root_depth, reverse=True)


def to_portable(path: Path | str) -> str:
    """Rewrite a path under an env root to '${NAME}/rel', under REPO_ROOT to a relative path.

    Symlinks are not resolved. Relative paths and strings already in '${NAME}' form are returned
    normalized or unchanged; any other absolute path is returned as-is.
    """
    text = str(path)
    if text.startswith("hf://"):
        return text
    if _ENV_REF.match(text):
        return text
    candidate = Path(os.path.normpath(text))
    if not candidate.is_absolute():
        return candidate.as_posix()
    for root, prefix in _env_root_prefixes():
        if candidate == root:
            return prefix
        if candidate.is_relative_to(root):
            return f"{prefix}/{candidate.relative_to(root).as_posix()}"
    if candidate.is_relative_to(REPO_ROOT):
        return candidate.relative_to(REPO_ROOT).as_posix()
    return str(candidate)


def resolve_path(value: Path | str) -> Path:
    """Inverse of to_portable: expand '${NAME}' (require_env); relative paths join REPO_ROOT."""
    text = str(value)
    match = _ENV_REF.match(text)
    if match is not None:
        name, rest = match.group(1), match.group(2)
        root = Path(require_env(name)[name])
        return root / rest if rest else root
    path = Path(text)
    return path if path.is_absolute() else REPO_ROOT / path
