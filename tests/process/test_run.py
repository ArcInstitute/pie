from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pie.process.config import compose_process_config
from pie.process.de import OUTPUT_COLUMNS
from pie.process.knockdown import DROPPED_FILE, STATS_FILE
from pie.process.run import ProcessOutputs, run_process
from tests.process.conftest import FakeGpudge
from tests.process.helpers import CTRL, KEPT_ROWS, N_CELLS, make_counts, stack, write_counts


def _screen(tmp_path: Path) -> Path:
    """A jiang-like file: contexts c1 and c2 in one h5ad."""
    adata = stack(make_counts("c1"), make_counts("c2"))
    return write_counts(adata, tmp_path / "counts" / "screen.h5ad")


def _dirs(tmp_path: Path) -> list[str]:
    return [
        f"filter.output_dir={tmp_path / 'filtered'}",
        f"normalize.output_dir={tmp_path / 'expression'}",
        f"de.output_dir={tmp_path / 'de'}",
    ]


def _jiang(tmp_path: Path, *extra: str) -> list[str]:
    # min_cells=3: the synthetic screen has at most 4 cells per perturbation.
    return ["dataset=jiang", f"input={_screen(tmp_path)}", "filter.min_cells=3", *extra]


def test_filter_feeds_normalize_and_de(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    out = run_process(compose_process_config(_jiang(tmp_path, *_dirs(tmp_path))))
    assert out == ProcessOutputs(
        inputs=[tmp_path / "counts" / "screen.h5ad"],
        filtered=[tmp_path / "filtered" / "screen.h5ad"],
        expression=[tmp_path / "expression" / "screen.h5ad"],
        de=[tmp_path / "de" / "c1.parquet", tmp_path / "de" / "c2.parquet"],
    )
    assert (tmp_path / "filtered" / STATS_FILE).is_file()
    assert (tmp_path / "filtered" / DROPPED_FILE).is_file()
    filtered = ad.read_h5ad(out.filtered[0])
    assert filtered.n_obs == 2 * len(KEPT_ROWS)
    expression = ad.read_h5ad(out.expression[0])
    assert expression.obs_names.tolist() == filtered.obs_names.tolist()
    assert expression.X.dtype == np.float32
    assert not np.array_equal(expression.X.data, np.floor(expression.X.data))
    # de reads the filtered raw counts, never the normalized expression.
    assert [sub.n_obs for sub, _ in fake_gpudge.calls] == [len(KEPT_ROWS)] * 2
    for sub, _ in fake_gpudge.calls:
        context = sub.obs["context"].astype(str).iloc[0]
        rows = (filtered.obs["context"].astype(str) == context).to_numpy()
        np.testing.assert_array_equal(sub.X.toarray(), filtered.X.toarray()[rows])


def test_without_the_filter_normalize_and_de_read_the_input(
    tmp_path: Path, fake_gpudge: FakeGpudge
) -> None:
    src = write_counts(make_counts("x"), tmp_path / "counts" / "lineA.h5ad")
    overrides = [
        "dataset=tahoe",
        f"input={src}",
        "de.groupby=gene",
        "de.reference=non-targeting",
        *_dirs(tmp_path)[1:],
    ]
    out = run_process(compose_process_config(overrides))
    assert out.filtered == []
    assert out.expression == [tmp_path / "expression" / "lineA.h5ad"]
    assert out.de == [tmp_path / "de" / "lineA.parquet"]
    assert [sub.n_obs for sub, _ in fake_gpudge.calls] == [N_CELLS]
    assert not (tmp_path / "filtered").exists()


def test_de_only(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    overrides = _jiang(
        tmp_path, "filter.enabled=false", "normalize.enabled=false", f"de.output_dir={tmp_path}/de"
    )
    out = run_process(compose_process_config(overrides))
    assert (out.filtered, out.expression) == ([], [])
    assert [sub.n_obs for sub, _ in fake_gpudge.calls] == [N_CELLS, N_CELLS]


def test_missing_gpudge_fails_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "gpudge", None)
    cfg = compose_process_config(_jiang(tmp_path, *_dirs(tmp_path)))
    with pytest.raises(RuntimeError, match="uv sync --extra process"):
        run_process(cfg)
    assert not (tmp_path / "filtered").exists()


def test_existing_output_fails_before_the_filter_runs(
    tmp_path: Path, fake_gpudge: FakeGpudge
) -> None:
    cfg = compose_process_config(_jiang(tmp_path, *_dirs(tmp_path)))
    (tmp_path / "expression").mkdir()
    (tmp_path / "expression" / "screen.h5ad").write_bytes(b"")
    with pytest.raises(FileExistsError):
        run_process(cfg)
    assert not (tmp_path / "filtered").exists()
    assert fake_gpudge.calls == []
    run_process(compose_process_config(_jiang(tmp_path, *_dirs(tmp_path), "overwrite=true")))
    assert ad.read_h5ad(tmp_path / "expression" / "screen.h5ad").n_obs == 2 * len(KEPT_ROWS)


def test_outputs_follow_the_pie_prep_contract(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    out = run_process(compose_process_config(_jiang(tmp_path, *_dirs(tmp_path))))
    expression = ad.read_h5ad(out.expression[0])
    assert {"context", "gene"} <= set(expression.obs.columns)
    contexts = sorted(expression.obs["context"].astype(str).unique())
    assert sorted(path.stem for path in out.de) == contexts
    for path in out.de:
        assert list(pd.read_parquet(path).columns) == list(OUTPUT_COLUMNS)


def test_filter_and_normalize_keep_every_obs_column(tmp_path: Path) -> None:
    counts = make_counts("c1")
    counts.obs["gene_id"] = [
        "nan" if pert == CTRL else f"ENSG{index:011d}"
        for index, pert in enumerate(counts.obs["gene"])
    ]
    counts.obs["gene_id"] = counts.obs["gene_id"].astype("category")
    src = write_counts(counts, tmp_path / "counts" / "s.h5ad")
    cfg = compose_process_config(
        [
            f"input={src}",
            "filter.enabled=true",
            "filter.context_column=context",
            f"filter.output_dir={tmp_path / 'filtered'}",
            "normalize.enabled=true",
            f"normalize.output_dir={tmp_path / 'expression'}",
        ]
    )
    run_process(cfg)
    for out in (tmp_path / "filtered" / "s.h5ad", tmp_path / "expression" / "s.h5ad"):
        got = ad.read_h5ad(out)
        assert list(got.obs.columns) == list(counts.obs.columns)
        expected = counts.obs.loc[got.obs_names, "gene_id"].astype(str)
        pd.testing.assert_series_equal(got.obs["gene_id"].astype(str), expected)
