"""Token-level text embeddings from a causal language model: every final hidden state."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pie.utils import atomic_write_text, canonical_json, sha256_bytes

log = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.5-35B-A3B-Base"
REVISION = "0f0813072d2358973511097385626f21fcb6d422"
MAX_LENGTH = 16384
ATTN_IMPLEMENTATION = "sdpa"
FLUSH_EVERY = 256
LOG_EVERY = 1000
TOKENS_FILE = "tokens.npy"
OFFSETS_FILE = "offsets.npy"
STATE_FILE = "state.json"
PARAMS: dict[str, Any] = {
    "max_length": MAX_LENGTH,
    "add_special_tokens": False,
    "model_dtype": "bfloat16",
    "storage_dtype": "float16",
    "attn_implementation": ATTN_IMPLEMENTATION,
}


def _load_model(model_id: str, revision: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return tokenizer, model.to(device).eval()


def cache_key(texts: Sequence[str], model_id: str = MODEL, revision: str = REVISION) -> str:
    """16 hex chars naming the resumable work dir of one (texts, model, revision, params) run."""
    texts_sha = sha256_bytes("\x1e".join(texts).encode("utf-8"))
    payload = {"texts": texts_sha, "model": model_id, "revision": revision, "params": PARAMS}
    return sha256_bytes(canonical_json(payload))[:16]


def _prepare_work_dir(work_dir: Path | None, offsets: np.ndarray) -> dict[str, int]:
    if work_dir is None:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    offsets_path = work_dir / OFFSETS_FILE
    if offsets_path.exists():
        if not np.array_equal(np.load(offsets_path), offsets):
            raise ValueError(f"{work_dir} holds a run over different texts; remove it first")
    else:
        np.save(offsets_path, offsets)
    state_path = work_dir / STATE_FILE
    return json.loads(state_path.read_text()) if state_path.exists() else {}


def _open_store(work_dir: Path | None, shape: tuple[int, int], create: bool) -> np.ndarray:
    if work_dir is None:
        return np.zeros(shape, dtype=np.float16)
    path = work_dir / TOKENS_FILE
    if create:
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    return np.lib.format.open_memmap(path, mode="r+")


def _save_state(work_dir: Path, store: np.ndarray, done: int, dim: int) -> None:
    if isinstance(store, np.memmap):
        store.flush()
    atomic_write_text(work_dir / STATE_FILE, json.dumps({"done": done, "dim": dim}) + "\n")


def _encode(model: Any, ids: Sequence[int], device: str) -> np.ndarray:
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    return output.last_hidden_state[0].to(torch.float16).cpu().numpy()


def embed_tokens(
    texts: Sequence[str],
    *,
    device: str,
    model_id: str = MODEL,
    revision: str = REVISION,
    work_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Final hidden states of every token of every text: (T, D) float16 and (N + 1,) offsets.

    Each text is tokenized without special tokens, truncated at MAX_LENGTH tokens and encoded
    alone (no padding). Text i owns rows offsets[i]:offsets[i + 1]. With `work_dir`, the rows go
    to a memory-mapped file there and a rerun resumes after the last flushed text.
    """
    if not texts:
        raise ValueError("no texts to embed")
    tokenizer, model = _load_model(model_id, revision, device)
    encoded = tokenizer(
        list(texts), add_special_tokens=False, truncation=True, max_length=MAX_LENGTH
    )
    ids = [list(row) for row in encoded["input_ids"]]
    lengths = np.array([len(row) for row in ids], dtype=np.int64)
    if (lengths == 0).any():
        raise ValueError(f"{int((lengths == 0).sum())} text(s) have no tokens")
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    state = _prepare_work_dir(work_dir, offsets)
    done = int(state.get("done", 0))
    store = _open_store(work_dir, (total, int(state["dim"])), create=False) if done else None
    for i in range(done, len(ids)):
        hidden = _encode(model, ids[i], device)
        if hidden.shape[0] != lengths[i]:
            raise RuntimeError(f"text {i}: {hidden.shape[0]} hidden states for {lengths[i]} tokens")
        if store is None:
            store = _open_store(work_dir, (total, int(hidden.shape[1])), create=True)
        store[offsets[i] : offsets[i + 1]] = hidden
        if work_dir is not None and ((i + 1) % FLUSH_EVERY == 0 or i + 1 == len(ids)):
            _save_state(work_dir, store, i + 1, int(hidden.shape[1]))
        if (i + 1) % LOG_EVERY == 0:
            log.info("qwen: embedded %d/%d texts", i + 1, len(ids))
    if store is None:
        raise RuntimeError("no hidden states were produced")
    return store, offsets
