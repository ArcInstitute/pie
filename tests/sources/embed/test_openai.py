"""OpenAI encoder: key check first, batch 64, token-window mean pooling, resume, extend."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pie.sources.contract import (
    FORMAT_VERSION,
    SourceMeta,
    extend_embeddings,
    read_source,
    write_source,
)
from pie.sources.embed import openai as oai
from pie.utils import MissingEnvError


class FakeEncoder:
    """One token per whitespace-separated word."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def vector(text: str) -> np.ndarray:
    v = np.zeros(oai.DIM, dtype=np.float32)
    v[0] = len(text)
    v[1] = sum(map(ord, text)) % 997
    return v


class FakeEmbeddings:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_after = fail_after

    def create(self, *, input: list[str], model: str) -> Any:
        assert model == oai.MODEL
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("API down")
        self.calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=vector(t).tolist()) for i, t in enumerate(input)]
        return SimpleNamespace(data=data[::-1])  # out of order on purpose


class FakeClient:
    def __init__(self, fail_after: int | None = None) -> None:
        self.embeddings = FakeEmbeddings(fail_after)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_missing_key_fails_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    client = FakeClient()
    with pytest.raises(MissingEnvError):
        oai.embed_texts(["a"], client=client, encoder=FakeEncoder())
    assert client.embeddings.calls == []
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert oai.require_openai_key() == "k"


def test_batches_of_64_in_input_order() -> None:
    texts = [f"text number {i}" for i in range(130)]
    client = FakeClient()
    out = oai.embed_texts(texts, client=client, encoder=FakeEncoder())
    assert [len(c) for c in client.embeddings.calls] == [64, 64, 2]
    assert out.dtype == np.float32 and out.shape == (130, oai.DIM)
    np.testing.assert_array_equal(out, np.stack([vector(t) for t in texts]))
    assert logging.getLogger("httpx").level == logging.WARNING
    assert oai.embed_texts([], client=client, encoder=FakeEncoder()).shape == (0, oai.DIM)


def test_long_text_is_window_mean_pooled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oai, "MAX_TOKENS", 4)
    long = " ".join(f"w{i}" for i in range(10))
    splits = oai.window_splits(long, FakeEncoder())
    assert splits == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
    client = FakeClient()
    out = oai.embed_texts(["a b", long], client=client, encoder=FakeEncoder())
    assert client.embeddings.calls == [["a b"], splits]
    np.testing.assert_array_equal(out[0], vector("a b"))
    expected = np.stack([vector(s) for s in splits]).mean(axis=0, dtype=np.float32)
    np.testing.assert_array_equal(out[1], expected)


def test_resume_skips_finished_batches(tmp_path: Path) -> None:
    texts = [f"t {i}" for i in range(150)]
    with pytest.raises(RuntimeError, match="API down"):
        oai.embed_texts(
            texts, client=FakeClient(fail_after=2), resume_dir=tmp_path, encoder=FakeEncoder()
        )
    client = FakeClient()
    out = oai.embed_texts(texts, client=client, resume_dir=tmp_path, encoder=FakeEncoder())
    assert [len(c) for c in client.embeddings.calls] == [22]
    np.testing.assert_array_equal(out, np.stack([vector(t) for t in texts]))
    other = FakeClient()
    oai.embed_texts(texts[:3], client=other, resume_dir=tmp_path, encoder=FakeEncoder())
    assert other.embeddings.calls == [texts[:3]]


def test_extend_sends_only_new_keys(tmp_path: Path) -> None:
    texts = {"A": "alpha text", "B": "beta text", "C": "gamma text"}
    prior_rows = np.stack([vector("old a"), vector("old b")])
    meta = SourceMeta(
        format_version=FORMAT_VERSION, name="perturbation_text", layout="dense", index="pert",
        keys=["A", "B"], dim=oai.DIM, dtype="float32", provenance={},
    )
    prior_dir = write_source(
        tmp_path / "prior", meta, prior_rows, descriptions={"A": "a", "B": "b"}
    )
    prior = read_source(prior_dir)
    client = FakeClient()
    rows, keys = extend_embeddings(
        prior,
        ["C", "A", "B"],
        lambda new: oai.embed_texts([texts[k] for k in new], client=client, encoder=FakeEncoder()),
    )
    assert keys == ["A", "B", "C"]
    assert client.embeddings.calls == [["gamma text"]]
    assert rows[:2].tobytes() == prior_rows.tobytes()
    np.testing.assert_array_equal(rows[2], vector("gamma text"))
