"""Tests for the knowledge-source on-disk contract."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pie.sources.contract import (
    DESCRIPTIONS,
    EMBEDDINGS,
    FORMAT_VERSION,
    META,
    OFFSETS,
    SOURCE_NAMES,
    DescriptionConflictError,
    Source,
    SourceMeta,
    extend_descriptions,
    extend_embeddings,
    read_descriptions,
    read_source,
    write_source,
)


def make_meta(
    keys: Sequence[str],
    *,
    dim: int = 4,
    dtype: str = "float32",
    layout: str = "dense",
    index: str = "pert",
    name: str = "esm2",
) -> SourceMeta:
    return SourceMeta(
        format_version=FORMAT_VERSION,
        name=name,
        layout=layout,
        index=index,
        keys=list(keys),
        dim=dim,
        dtype=dtype,
        provenance={"tool_version": "test"},
    )


def rand(n: int, d: int, dtype: str = "float32", seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(dtype)


def test_source_names_are_the_model_slots_plus_gene_text() -> None:
    assert set(SOURCE_NAMES) == {
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
    }
    assert len(SOURCE_NAMES) == 11


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_dense_roundtrip_preserves_bytes(tmp_path: Path, dtype: str) -> None:
    emb = rand(3, 4, dtype)
    meta = make_meta(["B", "A", "C"], dtype=dtype)
    out = write_source(tmp_path / "esm2", meta, emb)
    assert out == tmp_path / "esm2"
    assert sorted(p.name for p in out.iterdir()) == [EMBEDDINGS, META]
    src = read_source(out)
    assert isinstance(src.embeddings, np.memmap)
    assert src.meta == meta
    assert src.offsets is None
    assert src.key_to_row == {"B": 0, "A": 1, "C": 2}
    assert src.embeddings.dtype == np.dtype(dtype)
    assert src.embeddings.tobytes() == emb.tobytes()
    tok = src.tokens("A")
    assert tok.shape == (1, 4)
    assert tok.tobytes() == emb[1:2].tobytes()


def test_meta_json_holds_exactly_the_contract_fields(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a", "b"]), rand(2, 4))
    on_disk = json.loads((out / META).read_text())
    assert set(on_disk) == {
        "format_version",
        "name",
        "layout",
        "index",
        "keys",
        "dim",
        "dtype",
        "provenance",
    }
    assert on_disk["keys"] == ["a", "b"]
    assert on_disk["format_version"] == FORMAT_VERSION


def test_token_layout_segments_follow_offsets(tmp_path: Path) -> None:
    emb = rand(6, 3, "float16")
    offsets = np.array([0, 2, 5, 6], dtype=np.int64)
    meta = make_meta(["g1", "g2", "g3"], dim=3, dtype="float16", layout="token", name="ncbi_text")
    out = write_source(tmp_path / "ncbi_text", meta, emb, offsets=offsets)
    assert (out / OFFSETS).is_file()
    src = read_source(out)
    assert src.offsets is not None
    assert src.offsets.tolist() == [0, 2, 5, 6]
    assert src.key_to_row == {"g1": 0, "g2": 1, "g3": 2}
    assert src.tokens("g1").tobytes() == emb[0:2].tobytes()
    assert src.tokens("g2").shape == (3, 3)
    assert src.tokens("g2").tobytes() == emb[2:5].tobytes()
    assert src.tokens("g3").shape == (1, 3)
    assert src.embeddings.tobytes() == emb.tobytes()


def test_read_source_views_are_read_only(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a", "b"]), rand(2, 4))
    src = read_source(out)
    assert not src.embeddings.flags.writeable
    tok = src.tokens("b")
    with pytest.raises(ValueError):
        tok[0, 0] = 1.0
    assert read_source(out).embeddings.tobytes() == rand(2, 4).tobytes()


def test_tokens_unknown_key_raises_key_error(tmp_path: Path) -> None:
    src = read_source(write_source(tmp_path / "esm2", make_meta(["a"]), rand(1, 4)))
    with pytest.raises(KeyError):
        src.tokens("missing")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"embeddings": rand(3, 4, "float64")}, "dtype"),
        ({"embeddings": rand(3, 5)}, "dim"),
        ({"embeddings": rand(2, 4)}, "rows"),
        ({"embeddings": rand(3, 4).reshape(3, 4, 1)}, "2-D"),
        ({"offsets": np.array([0, 1, 2, 3], dtype=np.int64)}, "dense layout takes no offsets"),
        ({"descriptions": {"a": "x", "b": "y"}}, "descriptions"),
        ({"descriptions": {"b": "y", "a": "x", "c": "z"}}, "descriptions"),
        ({"descriptions": {"a": "x", "b": 3, "c": "z"}}, "descriptions"),
    ],
)
def test_write_rejects_inconsistent_dense_inputs(
    tmp_path: Path, override: dict[str, object], match: str
) -> None:
    args: dict[str, object] = {"embeddings": rand(3, 4), "offsets": None, "descriptions": None}
    args |= override
    with pytest.raises(ValueError, match=match):
        write_source(tmp_path / "esm2", make_meta(["a", "b", "c"]), **args)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("offsets", "match"),
    [
        (None, "requires offsets"),
        (np.array([0, 2, 5], dtype=np.int64), "offsets"),
        (np.array([0, 2, 5, 6], dtype=np.int32), "offsets"),
        (np.array([1, 2, 5, 6], dtype=np.int64), "offsets"),
        (np.array([0, 2, 5, 7], dtype=np.int64), "offsets"),
        (np.array([0, 3, 3, 6], dtype=np.int64), "strictly increasing"),
        (np.array([0, 4, 2, 6], dtype=np.int64), "strictly increasing"),
    ],
)
def test_write_rejects_bad_token_offsets(
    tmp_path: Path, offsets: np.ndarray | None, match: str
) -> None:
    meta = make_meta(["g1", "g2", "g3"], dim=3, dtype="float16", layout="token", name="ncbi_text")
    with pytest.raises(ValueError, match=match):
        write_source(tmp_path / "ncbi_text", meta, rand(6, 3, "float16"), offsets=offsets)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("name", "genept"),
        ("layout", "packed"),
        ("index", "batch"),
        ("dtype", "float64"),
        ("keys", []),
        ("keys", ["a", "a"]),
        ("keys", ["a", ""]),
        ("dim", 0),
    ],
)
def test_meta_rejects_invalid_fields(field: str, value: object) -> None:
    fields = make_meta(["a"]).model_dump() | {field: value}
    with pytest.raises(ValidationError):
        SourceMeta(**fields)


def test_meta_rejects_unknown_key() -> None:
    fields = make_meta(["a"]).model_dump() | {"symbol_to_index": {"a": 0}}
    with pytest.raises(ValidationError):
        SourceMeta(**fields)


def test_read_rejects_meta_that_disagrees_with_array(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a", "b", "c"]), rand(3, 4))
    meta = json.loads((out / META).read_text())
    meta["keys"] = ["a", "b"]
    (out / META).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="rows"):
        read_source(out)


def test_read_rejects_unknown_meta_key_on_disk(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a"]), rand(1, 4))
    meta = json.loads((out / META).read_text())
    meta["symbol_to_index"] = {"a": 0}
    (out / META).write_text(json.dumps(meta))
    with pytest.raises(ValidationError):
        read_source(out)


def test_read_rejects_offsets_file_in_dense_dir(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a", "b"]), rand(2, 4))
    np.save(out / OFFSETS, np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="must not contain"):
        read_source(out)


def test_write_refuses_existing_non_empty_dir(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a"]), rand(1, 4))
    with pytest.raises(FileExistsError):
        write_source(out, make_meta(["a"]), rand(1, 4, seed=1))
    assert read_source(out).embeddings.tobytes() == rand(1, 4).tobytes()


def test_write_with_overwrite_replaces_the_dir(tmp_path: Path) -> None:
    out = write_source(tmp_path / "esm2", make_meta(["a"]), rand(1, 4))
    assert write_source(out, make_meta(["a"]), rand(1, 4, seed=1), overwrite=True) == out
    assert read_source(out).embeddings.tobytes() == rand(1, 4, seed=1).tobytes()


def test_descriptions_roundtrip_keeps_row_order_and_text(tmp_path: Path) -> None:
    keys = ["zeta", "alpha", "Δ-gene"]
    desc = {"zeta": "Gene Name: zeta ;\nSummary: ü", "alpha": "  spaced  ", "Δ-gene": "x"}
    meta = make_meta(keys, name="gene_text", index="gene")
    out = write_source(tmp_path / "gene_text", meta, rand(3, 4), descriptions=desc)
    assert (out / DESCRIPTIONS).is_file()
    got = read_descriptions(out)
    assert list(got) == keys
    assert got == desc


def test_read_descriptions_rejects_keys_out_of_row_order(tmp_path: Path) -> None:
    meta = make_meta(["a", "b"], name="context_text", index="context")
    desc = {"a": "1", "b": "2"}
    out = write_source(tmp_path / "context_text", meta, rand(2, 4), descriptions=desc)
    (out / DESCRIPTIONS).write_text(json.dumps({"b": "2", "a": "1"}))
    with pytest.raises(ValueError, match="descriptions"):
        read_descriptions(out)


def test_sources_package_is_empty_and_contract_import_is_light() -> None:
    import pie.sources

    assert Path(pie.sources.__file__).read_text() == ""
    code = (
        "import sys, pie.sources.contract\n"
        "heavy = ('openai', 'tiktoken', 'transformers', 'rdkit', 'huggingface_hub')\n"
        "bad = [m for m in heavy if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


PRIOR_TEXT = {"b": "text b", "a": "text a"}


def test_extend_descriptions_appends_new_keys_after_prior() -> None:
    out = extend_descriptions(PRIOR_TEXT, {"c": "text c", "a": "text a", "d": "text d"})
    assert list(out.items()) == [
        ("b", "text b"),
        ("a", "text a"),
        ("c", "text c"),
        ("d", "text d"),
    ]
    assert PRIOR_TEXT == {"b": "text b", "a": "text a"}


def test_extend_descriptions_without_prior_keeps_new_order() -> None:
    out = extend_descriptions(None, {"y": "2", "x": "1"})
    assert list(out.items()) == [("y", "2"), ("x", "1")]


def test_extend_descriptions_conflict_lists_every_changed_key() -> None:
    with pytest.raises(DescriptionConflictError) as excinfo:
        extend_descriptions(PRIOR_TEXT, {"a": "changed a", "b": "changed b", "c": "new"})
    assert excinfo.value.keys == ["a", "b"]
    assert isinstance(excinfo.value, ValueError)


def test_extend_descriptions_keep_prior_keeps_prior_text() -> None:
    out = extend_descriptions(
        PRIOR_TEXT, {"a": "changed a", "b": "changed b", "c": "new"}, on_conflict="keep-prior"
    )
    assert list(out.items()) == [("b", "text b"), ("a", "text a"), ("c", "new")]


def test_extend_descriptions_replace_keeps_prior_position() -> None:
    out = extend_descriptions(
        PRIOR_TEXT, {"a": "changed a", "b": "changed b", "c": "new"}, on_conflict="replace"
    )
    assert list(out.items()) == [("b", "changed b"), ("a", "changed a"), ("c", "new")]


def test_extend_descriptions_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="on_conflict"):
        extend_descriptions(PRIOR_TEXT, {}, on_conflict="prefer-new")  # type: ignore[arg-type]


def _prior_source(tmp_path: Path) -> tuple[Source, np.ndarray]:
    emb = rand(3, 4)
    emb[0, 0] = -0.0
    emb[1, 1] = np.float32(1e-45)
    emb.view(np.uint32)[2, 2] = 0x7FC00123
    meta = make_meta(["p0", "p1", "p2"], name="perturbation_text")
    src = read_source(write_source(tmp_path / "prior", meta, emb))
    return src, emb


def test_extend_embeddings_copies_prior_rows_byte_for_byte(tmp_path: Path) -> None:
    prior, emb = _prior_source(tmp_path)
    calls: list[list[str]] = []

    def embed_new(keys: list[str]) -> np.ndarray:
        calls.append(list(keys))
        return np.full((len(keys), 4), 7.0, dtype=np.float32)

    out, row_keys = extend_embeddings(prior, ["n1", "p1", "n0", "n1", "p0"], embed_new)
    assert calls == [["n1", "n0"]]
    assert row_keys == ["p0", "p1", "p2", "n1", "n0"]
    assert out.dtype == np.float32
    assert out.shape == (5, 4)
    assert out[:3].tobytes() == emb.tobytes()
    assert np.all(out[3:] == 7.0)

    meta = make_meta(row_keys, name="perturbation_text")
    extended = read_source(write_source(tmp_path / "extended", meta, out))
    assert extended.embeddings[:3].tobytes() == prior.embeddings.tobytes()
    assert extended.key_to_row["n0"] == 4


def test_extend_embeddings_skips_embedder_when_nothing_is_new(tmp_path: Path) -> None:
    prior, emb = _prior_source(tmp_path)

    def embed_new(keys: list[str]) -> np.ndarray:
        raise AssertionError(f"embedder called with {keys}")

    out, row_keys = extend_embeddings(prior, ["p2", "p0"], embed_new)
    assert row_keys == ["p0", "p1", "p2"]
    assert out.tobytes() == emb.tobytes()
    assert not isinstance(out, np.memmap)


def test_extend_embeddings_without_prior_uses_embedder_output() -> None:
    def embed_new(keys: list[str]) -> np.ndarray:
        return np.arange(len(keys) * 4, dtype=np.float16).reshape(len(keys), 4)

    out, row_keys = extend_embeddings(None, ["x", "y", "x"], embed_new)
    assert row_keys == ["x", "y"]
    assert out.dtype == np.float16
    assert out.shape == (2, 4)


def test_extend_embeddings_without_prior_or_keys_raises() -> None:
    with pytest.raises(ValueError, match="nothing to embed"):
        extend_embeddings(None, [], lambda keys: np.zeros((0, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "fresh",
    [
        np.zeros((2, 4), dtype=np.float64),
        np.zeros((2, 5), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
    ],
)
def test_extend_embeddings_rejects_mismatched_new_rows(tmp_path: Path, fresh: np.ndarray) -> None:
    prior, _ = _prior_source(tmp_path)
    with pytest.raises(ValueError, match="embed_new"):
        extend_embeddings(prior, ["n0", "n1"], lambda keys: fresh)


def test_extend_helpers_compose_into_a_valid_text_source(tmp_path: Path) -> None:
    meta = make_meta(["p1", "p0"], name="perturbation_text")
    prior_dir = write_source(
        tmp_path / "prior", meta, rand(2, 4), descriptions={"p1": "one", "p0": "zero"}
    )
    prior = read_source(prior_dir)
    new_texts = {"n9": "nine", "p0": "zero", "n2": "two"}
    desc = extend_descriptions(read_descriptions(prior_dir), new_texts)
    emb, row_keys = extend_embeddings(
        prior, list(new_texts), lambda keys: rand(len(keys), 4, seed=5)
    )
    assert row_keys == list(desc) == ["p1", "p0", "n9", "n2"]
    out_meta = make_meta(row_keys, name="perturbation_text")
    out = write_source(tmp_path / "extended", out_meta, emb, descriptions=desc)
    assert list(read_descriptions(out)) == row_keys
    assert read_source(out).embeddings[:2].tobytes() == prior.embeddings.tobytes()


def test_extend_embeddings_rejects_token_prior(tmp_path: Path) -> None:
    meta = make_meta(["g1", "g2"], dim=3, dtype="float16", layout="token", name="ncbi_text")
    offsets = np.array([0, 1, 3], dtype=np.int64)
    prior = read_source(write_source(tmp_path / "tok", meta, rand(3, 3, "float16"), offsets))
    with pytest.raises(ValueError, match="dense"):
        extend_embeddings(prior, ["g3"], lambda keys: np.zeros((1, 3), dtype=np.float16))
