"""OpenAI text-embedding-3-large: input-order batches of 64, 8191-token window mean pooling."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pie.utils import MissingEnvError, canonical_json, read_json, sha256_bytes, write_json

MODEL = "text-embedding-3-large"
BATCH = 64
MAX_TOKENS = 8191
DIM = 3072
CHECKPOINT = "checkpoint.json"


def require_openai_key() -> str:
    """The key comes only from the environment; it is never logged or written anywhere."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise MissingEnvError(["OPENAI_API_KEY"])
    return key


def _tiktoken_encoder() -> Any:
    import tiktoken

    return tiktoken.encoding_for_model(MODEL)


def window_splits(text: str, encoder: Any, max_tokens: int | None = None) -> list[str]:
    """Token windows of `max_tokens` with stride max_tokens // 2; a short text is one window."""
    limit = MAX_TOKENS if max_tokens is None else max_tokens
    tokens = encoder.encode(text)
    if len(tokens) <= limit:
        return [text]
    stride = limit // 2
    splits: list[str] = []
    for start in range(0, len(tokens), stride):
        end = min(start + limit, len(tokens))
        splits.append(encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
    return splits


def _create(client: Any, inputs: list[str]) -> np.ndarray:
    response = client.embeddings.create(input=inputs, model=MODEL)
    data = sorted(response.data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in data], dtype=np.float32)


def _embed_batch(client: Any, encoder: Any, texts: list[str]) -> np.ndarray:
    lengths = [len(encoder.encode(text)) for text in texts]
    if all(length <= MAX_TOKENS for length in lengths):
        return _create(client, texts)
    rows = []
    for text, length in zip(texts, lengths, strict=True):
        splits = [text] if length <= MAX_TOKENS else window_splits(text, encoder)
        vectors = _create(client, splits)
        rows.append(vectors[0] if len(splits) == 1 else vectors.mean(axis=0, dtype=np.float32))
    return np.stack(rows).astype(np.float32, copy=False)


def _completed(resume_dir: Path, digest: str) -> set[int]:
    resume_dir.mkdir(parents=True, exist_ok=True)
    path = resume_dir / CHECKPOINT
    if path.exists():
        state = read_json(path)
        if isinstance(state, dict) and state.get("sha256") == digest:
            return set(state["completed_batches"])
    for stale in resume_dir.glob("batch_*.npy"):
        stale.unlink()
    return set()


def embed_texts(
    texts: Sequence[str],
    *,
    client: Any | None = None,
    resume_dir: Path | None = None,
    encoder: Any | None = None,
) -> np.ndarray:
    """(N, 3072) float32 rows in input order; finished batches are reused from `resume_dir`."""
    require_openai_key()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    texts = list(texts)
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    encoder = _tiktoken_encoder() if encoder is None else encoder
    total = math.ceil(len(texts) / BATCH)
    digest = sha256_bytes(canonical_json({"model": MODEL, "batch": BATCH, "texts": texts}))
    done = _completed(resume_dir, digest) if resume_dir is not None else set()
    parts: list[np.ndarray] = []
    for index in range(total):
        batch = texts[index * BATCH : (index + 1) * BATCH]
        path = resume_dir / f"batch_{index:06d}.npy" if resume_dir is not None else None
        if path is not None and index in done:
            parts.append(np.load(path))
            continue
        rows = _embed_batch(client, encoder, batch)
        if rows.shape != (len(batch), DIM):
            raise ValueError(f"OpenAI returned shape {rows.shape} for a batch of {len(batch)}")
        if path is not None and resume_dir is not None:
            np.save(path, rows)
            done.add(index)
            state = {"sha256": digest, "total_batches": total, "completed_batches": sorted(done)}
            write_json(resume_dir / CHECKPOINT, state)
        parts.append(rows)
    return np.concatenate(parts)
