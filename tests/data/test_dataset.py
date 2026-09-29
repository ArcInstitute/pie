from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pie.data.dataset import (
    NO_SOURCE,
    Batch,
    PieDataset,
    RowRef,
    SourceLookup,
    collate,
    load_aliases,
)
from pie.data.evidence import (
    EVIDENCE_KEYS,
    EvidenceConfig,
    EvidenceIndex,
    build_evidence_index,
    load_or_build_evidence,
)
from pie.data.preprocessed import PreprocessedDir
from pie.data.splits import load_split
from pie.sources.contract import Source, read_source
from pie.utils import sha256_file
from tests.fixtures import ALIASES, GENE_AXIS, SOURCE_DIMS, TinyData

CFG = EvidenceConfig(seed=0, chunk=2, lfc_clip_percentile=95.0)
UNION_IDS = [np.arange(0, 6, dtype=np.int64), np.arange(2, 8, dtype=np.int64)]


def _dirs(tiny: TinyData) -> list[PreprocessedDir]:
    return [PreprocessedDir.open(tiny.preprocessed[name]) for name in ("alpha", "beta")]


def _lookup(tiny: TinyData) -> tuple[dict[str, Source], SourceLookup]:
    sources = {name: read_source(path) for name, path in tiny.sources.items()}
    return sources, SourceLookup(sources, load_aliases(tiny.aliases))


def _index(tiny: TinyData) -> EvidenceIndex:
    train_path = tiny.split_dir / "train.json"
    ev = load_or_build_evidence(
        _dirs(tiny), load_split(train_path), sha256_file(train_path), CFG, 0.05,
        tiny.cache_dir, set(),
    )
    return build_evidence_index(ev, GENE_AXIS)


def _ref(dirs: list[PreprocessedDir], dir_index: int, context: str, pert: str) -> RowRef:
    return RowRef(dir_index, dirs[dir_index].row_index()[(context, pert)], context, pert)


def _dataset(tiny: TinyData, rows: list[RowRef], labels: str) -> PieDataset:
    _, lookup = _lookup(tiny)
    return PieDataset(_dirs(tiny), rows, UNION_IDS, lookup, _index(tiny), 0.05, labels)


def test_load_aliases_is_strict(tiny_data: TinyData, tmp_path: Path) -> None:
    assert load_aliases(tiny_data.aliases) == ALIASES
    bad_name = tmp_path / "bad_name.yaml"
    OmegaConf.save(OmegaConf.create({"not_a_source": {"A": "B"}}), bad_name)
    with pytest.raises(ValueError, match="unknown source"):
        load_aliases(bad_name)
    bad_value = tmp_path / "bad_value.yaml"
    OmegaConf.save(OmegaConf.create({"esm2": {"A": 1}}), bad_value)
    with pytest.raises(ValueError, match="str -> str"):
        load_aliases(bad_value)


def test_alias_is_used_only_on_a_direct_miss_with_an_existing_target(
    tiny_data: TinyData,
) -> None:
    sources, lookup = _lookup(tiny_data)
    esm2 = sources["esm2"]
    direct = lookup.tokens("esm2", context="a1", perturbation="GC")
    np.testing.assert_array_equal(direct, np.asarray(esm2.tokens("GC"), dtype=np.float32))
    aliased = lookup.tokens("esm2", context="a1", perturbation="OLDX")
    np.testing.assert_array_equal(aliased, np.asarray(esm2.tokens("GD"), dtype=np.float32))
    assert lookup.tokens("ncbi_text", context="a1", perturbation="OLDX") is None
    assert lookup.tokens("esm2", context="b1", perturbation="drugA") is None


def test_layouts_dtypes_and_context_index(tiny_data: TinyData) -> None:
    sources, lookup = _lookup(tiny_data)
    ncbi = lookup.tokens("ncbi_text", context="a1", perturbation="GB")
    assert ncbi is not None and ncbi.shape == (3, 3) and ncbi.dtype == np.float16
    smiles = lookup.tokens("smiles", context="b1", perturbation="drugA")
    assert smiles is not None and smiles.shape == (1, 3) and smiles.dtype == np.float32
    np.testing.assert_array_equal(smiles, sources["smiles"].tokens("drugA").astype(np.float32))
    ctx = lookup.tokens("context_text", context="b2", perturbation="drugA")
    np.testing.assert_array_equal(
        ctx, np.asarray(sources["context_text"].tokens("b2"), dtype=np.float32)
    )
    assert lookup.dims == SOURCE_DIMS
    assert list(lookup.dims) == list(tiny_data.sources)


def test_train_sample_serves_dense_labels_and_masks(tiny_data: TinyData) -> None:
    dirs = _dirs(tiny_data)
    rows = [_ref(dirs, 0, "a1", "GB"), _ref(dirs, 0, "a1", "GC")]
    ds = _dataset(tiny_data, rows, "train")
    alpha, r = dirs[0], rows[0].row
    s = ds[0]
    tested = np.asarray(alpha.tested[r], dtype=bool)
    np.testing.assert_array_equal(s["fold_changes"].numpy(), alpha.fold_changes[r])
    np.testing.assert_array_equal(s["tested"].numpy(), tested)
    np.testing.assert_array_equal(
        s["de_mask"].numpy(), (np.asarray(alpha.fdr[r]) < 0.05) & tested
    )
    assert np.all(s["fold_changes"].numpy()[~tested] == 0.0)
    np.testing.assert_array_equal(s["delta_p"].numpy(), alpha.delta_p[r])
    assert "lfc_true" not in s
    assert s["target_gene"] == 1 and s["target_gene_idx"] == 1
    gated = ds[1]
    assert gated["target_gene"] == -1 and gated["target_gene_idx"] == 2
    assert set(s["evidence"]) == set(EVIDENCE_KEYS)
    assert s["evidence"]["evidence_prov"].shape == (6, 6)
    assert s["gene_ids"].tolist() == list(range(6))


def test_eval_and_query_samples(tiny_data: TinyData) -> None:
    dirs = _dirs(tiny_data)
    labelled = _ref(dirs, 0, "a2", "GC")
    s = _dataset(tiny_data, [labelled], "eval")[0]
    assert s["lfc_true"].dtype == torch.float64
    np.testing.assert_array_equal(s["lfc_true"].numpy(), dirs[0].lfc_true[labelled.row])
    q = _dataset(tiny_data, [RowRef(1, -1, "b2", "drugZ")], "none")[0]
    for key in ("fold_changes", "de_mask", "tested", "delta_p", "lfc_true"):
        assert key not in q
    assert set(q["source_tokens"]) == {"context_text"}
    assert torch.isnan(q["ctrl_mean"]).all()
    assert q["target_gene"] == -1 and q["target_gene_idx"] == -1


def test_labelled_dataset_rejects_query_rows(tiny_data: TinyData) -> None:
    with pytest.raises(ValueError, match="query rows"):
        _dataset(tiny_data, [RowRef(1, -1, "b2", "drugZ")], "train")


def test_collate_groups_rows_per_dir(tiny_data: TinyData) -> None:
    dirs = _dirs(tiny_data)
    rows = [_ref(dirs, 1, "b1", "drugA"), _ref(dirs, 0, "a1", "GA"), _ref(dirs, 0, "a1", "GB")]
    ds = _dataset(tiny_data, rows, "train")
    batch = collate([ds[i] for i in range(3)], ds.sources.dims)
    assert isinstance(batch, Batch)
    assert list(batch.source_tokens) == [
        "context_text", "esm2", "ncbi_text", "perturbation_text", "smiles",
    ]
    assert batch.source_tokens["ncbi_text"].shape == (3, 3, 3)
    assert batch.source_tokens["ncbi_text"].dtype == torch.float16
    assert batch.source_masks["ncbi_text"].tolist() == [
        [False, False, False], [True, True, False], [True, True, True],
    ]
    assert batch.source_tokens["smiles"].dtype == torch.float32
    assert batch.source_masks["smiles"].tolist() == [[True], [False], [False]]
    assert batch.source_masks["esm2"].tolist() == [[False], [True], [True]]
    assert NO_SOURCE not in batch.source_tokens
    assert batch.dataset_ids.tolist() == [1, 0, 0]
    assert batch.row_index.tolist() == [r.row for r in rows]
    assert batch.ctx_names == ["b1", "a1", "a1"]
    assert batch.pert_names == ["drugA", "GA", "GB"]
    alpha_group, beta_group = batch.groups
    assert (alpha_group.dir_index, beta_group.dir_index) == (0, 1)
    assert alpha_group.rows.tolist() == [1, 2] and beta_group.rows.tolist() == [0]
    assert alpha_group.gene_ids.tolist() == list(range(6))
    assert beta_group.gene_ids.tolist() == list(range(2, 8))
    assert alpha_group.fold_changes is not None and alpha_group.fold_changes.shape == (2, 6)
    assert beta_group.fold_changes is not None and beta_group.fold_changes.shape == (1, 6)
    assert alpha_group.ctrl_means.shape == (2, 6)
    assert alpha_group.evidence["evidence_prov"].shape == (2, 6, 6)
    assert set(alpha_group.evidence) == set(EVIDENCE_KEYS)
    assert alpha_group.lfc_true is None
    expected_ga = 0 if bool(dirs[0].tested[rows[1].row][0]) else -1
    assert batch.target_gene.tolist() == [-1, expected_ga, 1]
    assert batch.target_gene_idx.tolist() == [-1, 0, 1]


def test_rows_without_sources_get_the_carrier(tiny_data: TinyData) -> None:
    dirs = _dirs(tiny_data)
    sources, _ = _lookup(tiny_data)
    only_smiles = SourceLookup({"smiles": sources["smiles"]}, {})
    rows = [_ref(dirs, 0, "a1", "GA"), _ref(dirs, 1, "b1", "drugA")]
    ds = PieDataset(dirs, rows, UNION_IDS, only_smiles, _index(tiny_data), 0.05, "train")
    batch = collate([ds[0], ds[1]], only_smiles.dims)
    assert list(batch.source_tokens) == ["smiles", NO_SOURCE]
    assert batch.source_tokens[NO_SOURCE].shape == (2, 1, 1)
    assert batch.source_masks[NO_SOURCE].tolist() == [[True], [False]]
    assert batch.source_masks["smiles"].tolist() == [[False], [True]]
