from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from pie.data import datamodule, samplers
from pie.data.datamodule import DATA_STATS, DataStats, PieDataModule
from pie.data.dataset import Batch, RowRef
from pie.data.delta_p import fit_grid, train_percentile
from pie.data.preprocessed import PreprocessedDir
from pie.data.splits import load_split, resolve_split
from pie.sources.contract import FORMAT_VERSION, SourceMeta, read_source, write_source
from pie.utils import read_json
from tests.fixtures import BETA_GENES, GENE_AXIS, SOURCE_DIMS, TinyData, tiny_data_config


def _fitted(tiny: TinyData, tmp_path: Path, **overrides: object) -> PieDataModule:
    dm = PieDataModule(tiny_data_config(tiny, **overrides), tmp_path / "run")
    dm.setup_stats(rank=0)
    return dm


def _boom(*args: object, **kwargs: object) -> None:
    raise AssertionError("pinned stats must not be refitted")


def test_gene_axis_is_the_first_seen_union(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    assert dm.gene_axis == GENE_AXIS
    assert [ids.tolist() for ids in dm.gene_union_ids] == [list(range(6)), list(range(2, 8))]
    assert dm.source_dims == SOURCE_DIMS
    assert list(dm.source_dims) == list(tiny_data.sources)
    flipped = tiny_data_config(
        tiny_data,
        preprocessed_dirs=[
            str(tiny_data.preprocessed["beta"]), str(tiny_data.preprocessed["alpha"])
        ],
    )
    assert PieDataModule(flipped, tmp_path / "run2").gene_axis == [*BETA_GENES, "GA", "GB"]


def test_dataset_weights_are_validated(tiny_data: TinyData, tmp_path: Path) -> None:
    for weights in (
        {"alpha": 1.0},
        {"alpha": -1.0, "beta": 1.0},
        {"alpha": 0.0, "beta": 0.0},
        {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    ):
        with pytest.raises(ValueError, match="dataset_weights"):
            PieDataModule(tiny_data_config(tiny_data, dataset_weights=weights), tmp_path / "run")


def test_partially_missing_control_means_are_rejected(tiny_data: TinyData, tmp_path: Path) -> None:
    broken = tmp_path / "alpha_broken"
    shutil.copytree(tiny_data.preprocessed["alpha"], broken)
    ctrl = np.load(broken / "ctrl_means.npy")
    ctrl[0, 0] = np.nan
    np.save(broken / "ctrl_means.npy", ctrl)
    cfg = tiny_data_config(
        tiny_data, preprocessed_dirs=[str(broken), str(tiny_data.preprocessed["beta"])]
    )
    with pytest.raises(ValueError, match="partially missing"):
        PieDataModule(cfg, tmp_path / "run")


def test_gene_query_text_follows_the_gene_axis(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    text = dm.gene_query_text()
    src = read_source(tiny_data.gene_text)
    assert text.dtype == torch.float32 and tuple(text.shape) == (8, 6)
    expected = np.stack([np.asarray(src.tokens(g), dtype=np.float32)[0] for g in GENE_AXIS])
    np.testing.assert_array_equal(text.numpy(), expected)
    partial = tmp_path / "gene_text_partial"
    keys = GENE_AXIS[:-1]
    meta = SourceMeta(
        format_version=FORMAT_VERSION, name="gene_text", layout="dense", index="gene",
        keys=keys, dim=6, dtype="float32", provenance={},
    )
    write_source(partial, meta, np.ones((len(keys), 6), dtype=np.float32))
    dm2 = PieDataModule(tiny_data_config(tiny_data, gene_text_dir=str(partial)), tmp_path / "r2")
    with pytest.raises(KeyError, match="GH"):
        dm2.gene_query_text()


def test_rank0_fits_stats_and_writes_the_handoff(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    cfg = tiny_data_config(tiny_data)
    dm = PieDataModule(cfg, tmp_path / "run")
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    stats = dm.setup_stats(rank=0)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert random.getstate() == py_state
    dirs = [PreprocessedDir.open(tiny_data.preprocessed[n]) for n in ("alpha", "beta")]
    rows = resolve_split(load_split(tiny_data.split_dir / "train.json"), dirs)
    expected = {d.dataset: train_percentile(d.delta_p, rows[d.dataset], 99.9) for d in dirs}
    assert stats.per_dir_percentile == expected
    assert stats.delta_p == fit_grid(expected, cfg.delta_p)
    assert stats.delta_p.n_bins % 2 == 1
    assert stats.datasets == ["alpha", "beta"]
    assert stats.evidence_datasets == ["alpha", "beta"] and stats.n_donor_datasets == 2
    assert stats.source_dims == SOURCE_DIMS and stats.gene_query_dim == 6
    payload = read_json(tmp_path / "run" / DATA_STATS)
    assert isinstance(payload, dict) and payload["launch_id"] is None
    assert DataStats.model_validate(payload["stats"]) == stats
    assert (tiny_data.cache_dir / "evidence" / stats.evidence_key / "response_meta.json").is_file()


def test_other_ranks_wait_for_the_matching_launch(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "launch-1")
    stats = _fitted(tiny_data, tmp_path).stats
    follower = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    assert follower.setup_stats(rank=1, timeout_s=1.0) == stats
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "launch-2")
    monkeypatch.setattr(datamodule, "_POLL_S", 0.01)
    stale = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    with pytest.raises(TimeoutError):
        stale.setup_stats(rank=1, timeout_s=0.05)


@pytest.mark.parametrize("run_id", [None, "none"])
def test_followers_reject_a_stale_handoff_without_a_launch_id(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: str | None
) -> None:
    if run_id is None:
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", run_id)
    monkeypatch.setattr(datamodule, "_POLL_S", 0.01)
    stats = _fitted(tiny_data, tmp_path).stats
    path = tmp_path / "run" / DATA_STATS
    # This launch started after the previous launch wrote its handoff file.
    monkeypatch.setattr(datamodule, "_LAUNCH_T0", time.time() + 1.0)
    stale = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    with pytest.raises(TimeoutError):
        stale.setup_stats(rank=1, timeout_s=0.05)

    monkeypatch.setattr(datamodule, "_LAUNCH_T0", time.time())
    rank0 = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    fit = rank0._fit_stats
    seen: list[bool] = []

    def fit_after_removal() -> DataStats:
        seen.append(path.exists())
        return fit()

    monkeypatch.setattr(rank0, "_fit_stats", fit_after_removal)
    assert rank0.setup_stats(rank=0) == stats
    assert seen == [False]
    follower = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    assert follower.setup_stats(rank=1, timeout_s=1.0) == stats


def test_a_follower_started_after_the_handoff_accepts_it(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A subprocess DDP launcher starts rank 1 as a new process after rank 0 wrote its handoff.
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    _fitted(tiny_data, tmp_path)
    path = tmp_path / "run" / DATA_STATS
    stale = tmp_path / "stale.json"
    payload = read_json(path)
    payload["written_at"] = datamodule._LAUNCH_T0 - 1.0
    stale.write_text(json.dumps(payload))
    code = (
        "import sys, time\n"
        "time.sleep(0.05)\n"
        "from pie.data import datamodule\n"
        "from pie.utils import read_json\n"
        "ok = datamodule._is_this_launch(read_json(sys.argv[1]), None)\n"
        "stale = datamodule._is_this_launch(read_json(sys.argv[2]), None)\n"
        "sys.exit(0 if ok and not stale else 1)\n"
    )
    subprocess.run([sys.executable, "-c", code, str(path), str(stale)], check=True)


def test_pinned_stats_are_never_refitted(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stats = _fitted(tiny_data, tmp_path).stats
    monkeypatch.setattr(datamodule, "fit_grid", _boom)
    monkeypatch.setattr(datamodule, "train_percentile", _boom)
    monkeypatch.setattr("pie.data.evidence.build_evidence", _boom)
    pinned = PieDataModule(tiny_data_config(tiny_data), None, stats=stats)
    assert pinned.setup_stats(rank=0) is stats
    test_rows = pinned.rows_from_split(tiny_data.split_dir / "test.json")
    ds = pinned.make_dataset(test_rows, "eval")
    assert len(ds) == 5 and "lfc_true" in ds[0]
    wrong = stats.model_copy(update={"evidence_key": "0" * 64})
    bad = PieDataModule(tiny_data_config(tiny_data), None, stats=wrong)
    with pytest.raises(ValueError, match="evidence key"):
        bad.make_dataset(test_rows, "eval")


def test_setup_requires_stats(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run")
    with pytest.raises(RuntimeError, match="setup_stats"):
        dm.setup("fit")


def test_fit_datasets_and_loaders(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = _fitted(tiny_data, tmp_path)
    dm.setup("fit")
    assert dm.train_dataset is not None and dm.val_dataset is not None
    assert len(dm.train_dataset) == 6 and len(dm.val_dataset) == 3
    batches = list(dm.train_dataloader())
    assert len(batches) == 1 and isinstance(batches[0], Batch)
    assert tuple(batches[0].row_index.shape) == (4,)
    assert all(g.lfc_true is None for g in batches[0].groups)
    val = list(dm.val_dataloader())
    assert dm.val_sharded is False
    assert sum(len(b.ctx_names) for b in val) == 3
    assert all(g.lfc_true is not None for b in val for g in b.groups)


def test_weight_zero_dir_is_never_sampled(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = _fitted(tiny_data, tmp_path, dataset_weights={"alpha": 1.0, "beta": 0.0})
    dm.setup("fit")
    batches = list(dm.train_dataloader())
    assert len(batches) == 1
    assert batches[0].dataset_ids.tolist() == [0, 0, 0, 0]
    assert dm.gene_axis == GENE_AXIS


def test_train_sampler_seam_is_resolved_at_call_time(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake(
        group_ids: np.ndarray, weights: dict[int, float], num_replicas: int, rank: int, seed: int
    ) -> SequentialSampler:
        seen.update(
            group_ids=group_ids.tolist(), weights=dict(weights), num_replicas=num_replicas,
            rank=rank, seed=seed,
        )
        return SequentialSampler(range(len(group_ids)))

    monkeypatch.setattr(samplers, "build_train_sampler", fake)
    dm = PieDataModule(tiny_data_config(tiny_data), tmp_path / "run", seed=7)
    dm.setup_stats(rank=0)
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, SequentialSampler)
    assert seen == {
        "group_ids": [0, 0, 0, 0, 1, 1],
        "weights": {0: 1.0, 1: 1.0},
        "num_replicas": 1,
        "rank": 0,
        "seed": 7,
    }


def test_rows_from_split_and_query(tiny_data: TinyData, tmp_path: Path) -> None:
    dm = _fitted(tiny_data, tmp_path)
    rows = dm.rows_from_split(tiny_data.split_dir / "val.json")
    assert [(r.dir_index, r.context, r.perturbation) for r in rows] == [
        (0, "a2", "GA"), (0, "a2", "GB"), (1, "b1", "drugC"),
    ]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"alpha.zz": ["GA"]}))
    with pytest.raises(ValueError, match="zz"):
        dm.rows_from_split(bad)
    no_row = tmp_path / "no_row.json"
    no_row.write_text(json.dumps({"alpha.a2": ["GA", "NOPE"]}))
    with pytest.raises(ValueError, match="NOPE"):
        dm.rows_from_split(no_row)
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"beta.b2": ["drugZ"], "alpha.a1": ["NEWG"]}))
    queried = dm.rows_from_query(query)
    assert queried == [RowRef(0, -1, "a1", "NEWG"), RowRef(1, -1, "b2", "drugZ")]
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"beta.b9": ["drugZ"]}))
    with pytest.raises(ValueError, match=r"beta\.b9"):
        dm.rows_from_query(unknown)
    sample = dm.make_dataset(queried, "none")[1]
    assert "fold_changes" not in sample
    assert set(sample["source_tokens"]) == {"context_text"}
