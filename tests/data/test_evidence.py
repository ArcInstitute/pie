from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pie.data import evidence
from pie.data.evidence import (
    EVIDENCE_KEYS,
    EVIDENCE_SCHEMA_VERSION,
    HAVE_COL,
    LOGCNT_COL,
    EvidenceConfig,
    EvidenceIndex,
    build_evidence,
    build_evidence_index,
    check_leakage,
    contributing_dirs,
    evidence_cache_key,
    load_evidence,
    load_or_build_evidence,
    serve_evidence,
)
from pie.data.preprocessed import PreprocessedDir
from pie.data.splits import load_split, resolve_split, split_pairs
from pie.utils import sha256_file
from tests.fixtures import GENE_AXIS, TinyData

CFG = EvidenceConfig(seed=0, chunk=2, lfc_clip_percentile=95.0)
FDR = 0.05


def _inputs(tiny: TinyData):
    dirs = [PreprocessedDir.open(tiny.preprocessed[name]) for name in ("alpha", "beta")]
    train_path = tiny.split_dir / "train.json"
    return dirs, load_split(train_path), sha256_file(train_path)


def _build(tiny: TinyData, out: Path, cfg: EvidenceConfig = CFG):
    dirs, split, sha = _inputs(tiny)
    build_evidence(dirs, split, sha, cfg, FDR, out)
    return dirs, split, sha, load_evidence(out)


def test_sums_and_counts_match_train_rows(tiny_data: TinyData, tmp_path: Path) -> None:
    dirs, split, _, ev = _build(tiny_data, tmp_path / "ev")
    assert ev.contributing_datasets == ["alpha", "beta"]
    assert ev.n_donor_datasets == 2
    assert ev.gene_symbols == sorted(GENE_AXIS)
    assert sorted(ev.pert_to_index) == sorted({p for _, _, p in split_pairs(split)})
    shape = ev.arrays["dp_cnt"].shape
    dp_sum = np.zeros(shape)
    dp_cnt = np.zeros(shape, dtype=np.int64)
    tested_cnt = np.zeros(shape, dtype=np.int64)
    up_cnt = np.zeros(shape, dtype=np.int64)
    by_ds = np.zeros((2, *shape), dtype=np.int64)
    gene_col = {g: i for i, g in enumerate(ev.gene_symbols)}
    rows = resolve_split(split, dirs)
    for slot, d in enumerate(dirs):
        keys = d.row_keys()
        cols = np.array([gene_col[g] for g in d.genes])
        for r in rows[d.dataset]:
            p = ev.pert_to_index[keys[int(r)][1]]
            fc = np.asarray(d.fold_changes[r], dtype=np.float64)
            fdr = np.asarray(d.fdr[r], dtype=np.float64)
            tested = np.asarray(d.tested[r], dtype=bool)
            dp_sum[p, cols] += np.asarray(d.delta_p[r], dtype=np.float64)
            dp_cnt[p, cols] += 1
            by_ds[slot, p, cols] += 1
            tested_cnt[p, cols] += tested
            up_cnt[p, cols] += tested & (fdr < FDR) & (fc > 1.0)
    np.testing.assert_array_equal(ev.arrays["dp_cnt"], dp_cnt)
    np.testing.assert_array_equal(ev.arrays["tested_cnt"], tested_cnt)
    np.testing.assert_array_equal(ev.arrays["up_cnt"], up_cnt)
    np.testing.assert_array_equal(ev.by_ds["dp_cnt_by_ds"], by_ds)
    np.testing.assert_allclose(ev.arrays["dp_sum"], dp_sum, rtol=1e-6, atol=1e-7)
    assert ev.arrays["dp_sum"].dtype == np.float32
    assert ev.arrays["dp_cnt"].dtype == np.uint16
    assert ev.by_ds["dp_cnt_by_ds"].dtype == np.uint8


def test_build_is_byte_deterministic(tiny_data: TinyData, tmp_path: Path) -> None:
    _, _, _, first = _build(tiny_data, tmp_path / "one")
    _, _, _, second = _build(tiny_data, tmp_path / "two")
    names = sorted(p.relative_to(first.path) for p in first.path.rglob("*.npy"))
    assert names == sorted(p.relative_to(second.path) for p in second.path.rglob("*.npy"))
    assert len(names) == 11 + 2 * 12
    for name in names:
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()


def test_meta_is_path_free_and_keyed(tiny_data: TinyData, tmp_path: Path) -> None:
    dirs, split, sha, ev = _build(tiny_data, tmp_path / "ev")
    text = (ev.path / "response_meta.json").read_text()
    assert str(tmp_path) not in text
    for banned in ("argv", "built_at", "builder", "contributing_dirs", "split_dir"):
        assert f'"{banned}"' not in text
    meta = json.loads(text)
    assert meta["schema_version"] == EVIDENCE_SCHEMA_VERSION
    contrib = contributing_dirs(dirs, resolve_split(split, dirs))
    expected = evidence_cache_key(contrib, sha, CFG, FDR)
    assert meta["cache_key"] == expected == ev.key
    assert ev.train_json_sha256 == sha


def test_donor_keys_are_train_pairs_without_batch(tiny_data: TinyData, tmp_path: Path) -> None:
    _, split, _, ev = _build(tiny_data, tmp_path / "ev")
    assert ev.donor_keys == frozenset(split_pairs(split))
    shard_meta = json.loads(
        (ev.path / "context_response" / "0000" / "shard_meta.json").read_text()
    )
    assert shard_meta["dataset"] == "alpha"
    assert shard_meta["contexts"] == ["a1"]
    assert all(len(key) == 2 for key in shard_meta["donor_keys"])
    assert ev.shards["beta"].context_to_index == {"b1": 0}


def test_cache_key_covers_every_input(tiny_data: TinyData) -> None:
    dirs, _, sha = _inputs(tiny_data)
    key = evidence_cache_key(dirs, sha, CFG, FDR)
    assert key == evidence_cache_key(list(reversed(dirs)), sha, CFG, FDR)
    variants = [
        evidence_cache_key(dirs, "0" * 64, CFG, FDR),
        evidence_cache_key(dirs, sha, CFG.model_copy(update={"seed": 1}), FDR),
        evidence_cache_key(dirs, sha, CFG.model_copy(update={"chunk": 3}), FDR),
        evidence_cache_key(dirs, sha, CFG.model_copy(update={"lfc_clip_percentile": 99.0}), FDR),
        evidence_cache_key(dirs, sha, CFG, 0.1),
        evidence_cache_key(dirs[:1], sha, CFG, FDR),
    ]
    assert len({key, *variants}) == len(variants) + 1


def test_contributing_dirs_are_sorted_and_skip_dirs_without_train_rows(
    tiny_data: TinyData,
) -> None:
    dirs, split, _ = _inputs(tiny_data)
    alpha, beta = dirs
    flipped = [beta, alpha]
    got = contributing_dirs(flipped, resolve_split(split, flipped))
    assert [d.dataset for d in got] == ["alpha", "beta"]
    only_beta = {"beta.b1": ["drugA"]}
    assert [d.dataset for d in contributing_dirs(dirs, resolve_split(only_beta, dirs))] == ["beta"]


def test_load_or_build_reuses_the_cache(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs, split, sha = _inputs(tiny_data)
    calls: list[Path] = []
    real = evidence.build_evidence

    def spy(*args: object, **kwargs: object) -> Path:
        calls.append(Path(str(args[-1])))
        return real(*args, **kwargs)

    monkeypatch.setattr(evidence, "build_evidence", spy)
    root = tmp_path / "cache"
    first = load_or_build_evidence(dirs, split, sha, CFG, FDR, root, set())
    second = load_or_build_evidence(dirs, split, sha, CFG, FDR, root, set())
    assert len(calls) == 1
    assert first.path == second.path == root / "evidence" / first.key
    assert not [p for p in (root / "evidence").iterdir() if p.name != first.key]


def test_leakage_check_runs_on_build_and_load(tiny_data: TinyData, tmp_path: Path) -> None:
    dirs, split, sha = _inputs(tiny_data)
    root = tmp_path / "cache"
    ev = load_or_build_evidence(dirs, split, sha, CFG, FDR, root, set())
    train = split_pairs(split)
    check_leakage(ev, train, split_pairs(load_split(tiny_data.split_dir / "val.json")))
    with pytest.raises(ValueError, match="leak"):
        check_leakage(ev, train, {("alpha", "a1", "GA")})
    with pytest.raises(ValueError, match="not in the train split"):
        check_leakage(ev, train - {("beta", "b1", "drugA")}, set())
    with pytest.raises(ValueError, match="leak"):
        load_or_build_evidence(dirs, split, sha, CFG, FDR, root, {("beta", "b1", "drugB")})


ALPHA_IDS = np.arange(0, 6, dtype=np.int64)
BETA_IDS = np.arange(2, 8, dtype=np.int64)
DONOR_BLOCKS = ("evidence_dp", "evidence_de", "evidence_lfc", "evidence_prov")
CONTEXT_BLOCKS = ("evidence_ctx_dp", "evidence_ctx_de", "evidence_ctx_lfc")


def _index(tiny: TinyData, tmp_path: Path) -> tuple[list[PreprocessedDir], EvidenceIndex]:
    dirs, split, sha = _inputs(tiny)
    ev = load_or_build_evidence(dirs, split, sha, CFG, FDR, tmp_path / "cache", set())
    return dirs, build_evidence_index(ev, GENE_AXIS)


def test_index_maps_axes_by_name(tiny_data: TinyData, tmp_path: Path) -> None:
    _, index = _index(tiny_data, tmp_path)
    np.testing.assert_array_equal(index.gene_cols, np.arange(8))
    np.testing.assert_array_equal(index.context_gene_cols["alpha"], [0, 1, 2, 3, 4, 5, -1, -1])
    np.testing.assert_array_equal(index.context_gene_cols["beta"], [-1, -1, 0, 1, 2, 3, 4, 5])
    assert index.ds_slot == {"alpha": 0, "beta": 1}
    assert index.pert_rows == index.evidence.pert_to_index


def test_train_row_excludes_its_own_contribution(tiny_data: TinyData, tmp_path: Path) -> None:
    _, index = _index(tiny_data, tmp_path)
    blocks = serve_evidence(
        index, dataset="alpha", context="a1", perturbation="GA", gene_union_ids=ALPHA_IDS
    )
    assert list(blocks) == list(EVIDENCE_KEYS)
    widths = {key: value.shape for key, value in blocks.items()}
    assert widths == {**{k: (6, 4) for k in EVIDENCE_KEYS}, "evidence_prov": (6, 6)}
    assert all(value.dtype == np.float32 for value in blocks.values())
    for key in DONOR_BLOCKS:
        np.testing.assert_array_equal(blocks[key], 0.0)
    np.testing.assert_array_equal(blocks["evidence_ctx_dp"][:, HAVE_COL], 1.0)
    np.testing.assert_allclose(blocks["evidence_ctx_dp"][:, LOGCNT_COL], np.log1p(3.0), rtol=1e-6)


def test_val_row_sees_the_train_donor(tiny_data: TinyData, tmp_path: Path) -> None:
    dirs, index = _index(tiny_data, tmp_path)
    alpha = dirs[0]
    donor = alpha.row_index()[("a1", "GA")]
    blocks = serve_evidence(
        index, dataset="alpha", context="a2", perturbation="GA", gene_union_ids=ALPHA_IDS
    )
    scale = index.evidence.scales.dp
    expected = (np.asarray(alpha.delta_p[donor], dtype=np.float64) / scale).astype(np.float32)
    np.testing.assert_allclose(blocks["evidence_dp"][:, 0], expected, rtol=1e-6)
    np.testing.assert_array_equal(blocks["evidence_dp"][:, 1], 0.0)
    np.testing.assert_array_equal(blocks["evidence_dp"][:, HAVE_COL], 1.0)
    np.testing.assert_allclose(blocks["evidence_dp"][:, LOGCNT_COL], np.log1p(1.0), rtol=1e-6)
    np.testing.assert_array_equal(blocks["evidence_prov"][:, 0], 1.0)
    np.testing.assert_array_equal(blocks["evidence_prov"][:, 3:], 0.0)
    for key in CONTEXT_BLOCKS:
        np.testing.assert_array_equal(blocks[key], 0.0)


def test_uncovered_perturbation_on_a_local_axis(tiny_data: TinyData, tmp_path: Path) -> None:
    _, index = _index(tiny_data, tmp_path)
    blocks = serve_evidence(
        index, dataset="beta", context="b1", perturbation="drugC", gene_union_ids=BETA_IDS
    )
    for key in DONOR_BLOCKS:
        np.testing.assert_array_equal(blocks[key], 0.0)
    np.testing.assert_allclose(blocks["evidence_ctx_dp"][:, LOGCNT_COL], np.log1p(2.0), rtol=1e-6)
    own = serve_evidence(
        index, dataset="beta", context="b1", perturbation="drugA", gene_union_ids=BETA_IDS
    )
    np.testing.assert_allclose(own["evidence_ctx_dp"][:, LOGCNT_COL], np.log1p(1.0), rtol=1e-6)
