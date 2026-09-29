"""Tests for pie.data.splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pie
from pie.data import preprocessed as pp
from pie.data.splits import (
    SPLIT_FILES,
    SplitKeyError,
    check_disjoint,
    load_split,
    parse_split_key,
    resolve_split,
    split_key,
    split_pairs,
)


def _write_dir(
    root: Path, dataset: str, contexts: list[str], rows: list[tuple[str, str]]
) -> pp.PreprocessedDir:
    """Tiny labelled preprocessed dir; `rows` must be sorted by (context, perturbation)."""
    ctx_to_id = {c: i for i, c in enumerate(contexts)}
    perts = sorted({p for _, p in rows})
    pert_to_id = {p: i for i, p in enumerate(perts)}
    n, g, c = len(rows), 2, len(contexts)
    arrays = {
        pp.FOLD_CHANGES: np.ones((n, g), dtype=np.float32),
        pp.FDR: np.ones((n, g), dtype=np.float32),
        pp.TESTED: np.ones((n, g), dtype=bool),
        pp.LFC_TRUE: np.zeros((n, g), dtype=np.float64),
        pp.DELTA_P: np.zeros((n, g), dtype=np.float32),
        pp.CTRL_MEANS: np.zeros((c, g), dtype=np.float32),
        pp.CTX_IDS: np.array([ctx_to_id[x] for x, _ in rows], dtype=np.int32),
        pp.PERT_IDS: np.array([pert_to_id[p] for _, p in rows], dtype=np.int32),
    }
    meta = pp.PreprocessedMeta(
        format_version=pp.FORMAT_VERSION,
        dataset=dataset,
        genes=["G0", "G1"],
        context_to_id=ctx_to_id,
        pert_to_id=pert_to_id,
        pert_kind="gene",
        control_label="non-targeting",
        num_rows=n,
        num_genes=g,
        num_contexts=c,
        num_perts=len(perts),
        controls_only=False,
        tool_version=pie.__version__,
        array_sha256={},
    )
    return pp.PreprocessedDir.open(pp.write_preprocessed(root / dataset, meta, arrays))


def _write_controls_only(root: Path, dataset: str, contexts: list[str]) -> pp.PreprocessedDir:
    meta = pp.PreprocessedMeta(
        format_version=pp.FORMAT_VERSION,
        dataset=dataset,
        genes=["G0", "G1"],
        context_to_id={c: i for i, c in enumerate(contexts)},
        pert_to_id={},
        pert_kind="gene",
        control_label="non-targeting",
        num_rows=0,
        num_genes=2,
        num_contexts=len(contexts),
        num_perts=0,
        controls_only=True,
        tool_version=pie.__version__,
        array_sha256={},
    )
    arrays = {pp.CTRL_MEANS: np.zeros((len(contexts), 2), dtype=np.float32)}
    return pp.PreprocessedDir.open(pp.write_preprocessed(root / dataset, meta, arrays))


@pytest.fixture
def dirs(tmp_path: Path) -> list[pp.PreprocessedDir]:
    replogle = _write_dir(
        tmp_path,
        "replogle",
        ["k562", "rpe1"],
        [("k562", "AAA"), ("k562", "BBB"), ("k562", "CCC"), ("rpe1", "AAA"), ("rpe1", "DDD")],
    )
    tahoe = _write_dir(tmp_path, "tahoe", ["a549"], [("a549", "drugX_1uM"), ("a549", "drugY_1uM")])
    return [replogle, tahoe]


def _write_json(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "split.json"
    path.write_text(text)
    return path


def test_split_files_constant() -> None:
    assert SPLIT_FILES == ("train.json", "val.json")


def test_load_split_valid_and_empty(tmp_path: Path) -> None:
    path = _write_json(tmp_path, json.dumps({"replogle.k562": ["BBB", "AAA"], "tahoe.a549": []}))
    assert load_split(path) == {"replogle.k562": ["BBB", "AAA"], "tahoe.a549": []}
    assert load_split(_write_json(tmp_path, "{}")) == {}


@pytest.mark.parametrize(
    "text",
    [
        '["replogle.k562"]',
        '{"k562": ["AAA"]}',
        '{".k562": ["AAA"]}',
        '{"replogle.": ["AAA"]}',
        '{"replogle.k562": {"batch1": ["AAA"]}}',
        '{"replogle.k562": ["AAA", 3]}',
        '{"replogle.k562": ["AAA", "AAA"]}',
        '{"replogle.k562": ["AAA"], "replogle.k562": ["BBB"]}',
    ],
)
def test_load_split_rejects_non_strict_forms(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_split(_write_json(tmp_path, text))


def test_split_key_round_trip_uses_first_dot() -> None:
    assert split_key("replogle", "k562") == "replogle.k562"
    assert parse_split_key("replogle.k562") == ("replogle", "k562")
    assert parse_split_key("tahoe.HepG2.C3A") == ("tahoe", "HepG2.C3A")
    with pytest.raises(ValueError):
        parse_split_key("k562")
    with pytest.raises(ValueError):
        split_key("rep.logle", "k562")


def test_split_pairs() -> None:
    split = {"replogle.k562": ["AAA", "BBB"], "tahoe.a549": ["drugX_1uM"]}
    assert split_pairs(split) == {
        ("replogle", "k562", "AAA"),
        ("replogle", "k562", "BBB"),
        ("tahoe", "a549", "drugX_1uM"),
    }


def test_resolve_split_rows_sorted_per_dataset(dirs: list[pp.PreprocessedDir]) -> None:
    split = {"replogle.rpe1": ["DDD"], "replogle.k562": ["CCC", "AAA"]}
    out = resolve_split(split, dirs)
    assert list(out) == ["replogle", "tahoe"]
    np.testing.assert_array_equal(out["replogle"], np.array([0, 2, 4]))
    assert out["replogle"].dtype == np.int64
    assert out["tahoe"].dtype == np.int64
    assert out["tahoe"].size == 0


def test_resolve_split_lists_every_offender(dirs: list[pp.PreprocessedDir]) -> None:
    split = {
        "jiang.k562": ["AAA"],
        "replogle.hepg2": ["AAA"],
        "replogle.k562": ["AAA", "ZZZ"],
        "tahoe.a549": ["drugZ_1uM"],
    }
    with pytest.raises(SplitKeyError) as info:
        resolve_split(split, dirs)
    offenders = info.value.offenders
    assert len(offenders) == 4
    text = str(info.value)
    for needle in ("jiang", "hepg2", "ZZZ", "drugZ_1uM"):
        assert needle in text


def test_resolve_split_checks_the_context_pert_pair(dirs: list[pp.PreprocessedDir]) -> None:
    # BBB has a row in k562 only and DDD in rpe1 only: each is an offender in the other context.
    with pytest.raises(SplitKeyError) as info:
        resolve_split({"replogle.rpe1": ["BBB"], "replogle.k562": ["DDD", "AAA"]}, dirs)
    assert len(info.value.offenders) == 2
    assert "BBB" in str(info.value) and "DDD" in str(info.value)


def test_resolve_split_rejects_controls_only_dir(tmp_path: Path) -> None:
    query_dir = _write_controls_only(tmp_path, "custom", ["k562"])
    with pytest.raises(SplitKeyError):
        resolve_split({"custom.k562": ["AAA"]}, [query_dir])
    np.testing.assert_array_equal(resolve_split({}, [query_dir])["custom"], np.array([]))


def test_check_disjoint() -> None:
    train = {"replogle.k562": ["AAA", "BBB"]}
    val = {"replogle.k562": ["CCC"], "replogle.rpe1": ["AAA"]}
    check_disjoint({"train": train, "val": val})
    with pytest.raises(ValueError, match="'train' and 'test'"):
        check_disjoint({"train": train, "val": val, "test": {"replogle.k562": ["BBB"]}})
