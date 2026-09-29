"""Tests for the token-level Qwen encoder (tokenizer and model are faked)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.sources.inputs import FakeQwenModel, FakeQwenTokenizer

from pie.sources.embed import qwen


def _patch(monkeypatch: pytest.MonkeyPatch, model: FakeQwenModel) -> FakeQwenTokenizer:
    tokenizer = FakeQwenTokenizer()

    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        assert (model_id, revision, device) == (qwen.MODEL, qwen.REVISION, "cpu")
        return tokenizer, model

    monkeypatch.setattr(qwen, "_load_model", load)
    return tokenizer


def _expected(texts: list[str]) -> np.ndarray:
    rows = [[ord(c), 2 * ord(c), pos] for text in texts for pos, c in enumerate(text)]
    return np.array(rows, dtype=np.float16)


def test_embed_tokens_packs_every_hidden_state(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _patch(monkeypatch, FakeQwenModel())
    tokens, offsets = qwen.embed_tokens(["ab", "c"], device="cpu")
    assert tokens.dtype == np.float16
    assert tokens.shape == (3, 3)
    np.testing.assert_array_equal(offsets, np.array([0, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(tokens, _expected(["ab", "c"]))
    assert tokenizer.kwargs == {
        "add_special_tokens": False,
        "truncation": True,
        "max_length": qwen.MAX_LENGTH,
    }


def test_embed_tokens_truncates_at_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeQwenModel())
    monkeypatch.setattr(qwen, "MAX_LENGTH", 2)
    tokens, offsets = qwen.embed_tokens(["abcd"], device="cpu")
    assert offsets.tolist() == [0, 2]
    assert tokens.shape == (2, 3)


def test_embed_tokens_rejects_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, FakeQwenModel())
    with pytest.raises(ValueError, match="no texts"):
        qwen.embed_tokens([], device="cpu")
    with pytest.raises(ValueError, match="no tokens"):
        qwen.embed_tokens(["ab", ""], device="cpu")


def test_embed_tokens_resumes_from_its_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    texts = ["ab", "c", "def"]
    monkeypatch.setattr(qwen, "FLUSH_EVERY", 1)
    _patch(monkeypatch, FakeQwenModel(fail_on_call=3))
    with pytest.raises(RuntimeError, match="simulated failure"):
        qwen.embed_tokens(texts, device="cpu", work_dir=tmp_path / "work")
    model = FakeQwenModel()
    _patch(monkeypatch, model)
    tokens, offsets = qwen.embed_tokens(texts, device="cpu", work_dir=tmp_path / "work")
    assert model.calls == 1
    assert isinstance(tokens, np.memmap)
    np.testing.assert_array_equal(tokens, _expected(texts))
    assert offsets.tolist() == [0, 2, 3, 6]


def test_work_dir_of_other_texts_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch, FakeQwenModel())
    qwen.embed_tokens(["ab"], device="cpu", work_dir=tmp_path / "work")
    with pytest.raises(ValueError, match="different texts"):
        qwen.embed_tokens(["abc"], device="cpu", work_dir=tmp_path / "work")


def test_cache_key_depends_on_texts_and_revision() -> None:
    key = qwen.cache_key(["a", "b"])
    assert len(key) == 16
    assert key == qwen.cache_key(["a", "b"])
    assert key != qwen.cache_key(["a", "c"])
    assert key != qwen.cache_key(["a", "b"], revision="other")
