from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from pie.process.io import (
    check_writable,
    context_masks,
    resolve_inputs,
    write_csv_atomic,
    write_h5ad_atomic,
)
from tests.process.helpers import N_CELLS, make_counts, stack


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_resolve_inputs_expands_a_glob_sorted(tmp_path: Path) -> None:
    b = _touch(tmp_path / "b.h5ad")
    a = _touch(tmp_path / "a.h5ad")
    _touch(tmp_path / "c.txt")
    assert resolve_inputs(str(tmp_path / "*.h5ad")) == [a, b]


def test_resolve_inputs_accepts_a_list_and_dedupes(tmp_path: Path) -> None:
    a = _touch(tmp_path / "x" / "a.h5ad")
    b = _touch(tmp_path / "y" / "b.h5ad")
    assert resolve_inputs([str(b), str(a), str(tmp_path / "x" / "*.h5ad")]) == [a, b]


def test_resolve_inputs_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no files"):
        resolve_inputs(str(tmp_path / "*.h5ad"))
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_inputs(str(tmp_path / "absent.h5ad"))


def test_resolve_inputs_rejects_colliding_names(tmp_path: Path) -> None:
    a = _touch(tmp_path / "x" / "a.h5ad")
    b = _touch(tmp_path / "y" / "a.h5ad")
    with pytest.raises(ValueError, match="share a file name"):
        resolve_inputs([str(a), str(b)])


def test_check_writable(tmp_path: Path) -> None:
    path = _touch(tmp_path / "out.h5ad")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        check_writable(path, overwrite=False)
    check_writable(path, overwrite=True)
    check_writable(tmp_path / "new.h5ad", overwrite=False)


def test_write_h5ad_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    out = tmp_path / "out" / "a.h5ad"
    assert write_h5ad_atomic(make_counts(), out, overwrite=False) == out
    assert sorted(p.name for p in out.parent.iterdir()) == ["a.h5ad"]
    with pytest.raises(FileExistsError):
        write_h5ad_atomic(make_counts(), out, overwrite=False)
    write_h5ad_atomic(make_counts("c2"), out, overwrite=True)
    assert ad.read_h5ad(out).obs_names[0] == "c2_0"


def test_write_h5ad_atomic_cleans_up_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: ad.AnnData, filename: Path, **kwargs: object) -> None:
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_h5ad_atomic(make_counts(), tmp_path / "a.h5ad", overwrite=False)
    assert list(tmp_path.iterdir()) == []


def test_write_csv_atomic(tmp_path: Path) -> None:
    path = write_csv_atomic("a,b\n1,2\n", tmp_path / "s.csv", overwrite=False)
    assert path.read_bytes() == b"a,b\n1,2\n"
    with pytest.raises(FileExistsError):
        write_csv_atomic("x\n", path, overwrite=False)


def test_context_masks_are_sorted_by_value() -> None:
    masks = context_masks(stack(make_counts("c2"), make_counts("c1")), "context", "stem")
    assert [ctx for ctx, _ in masks] == ["c1", "c2"]
    assert masks[0][1].tolist() == [False] * N_CELLS + [True] * N_CELLS


def test_context_masks_without_a_column_use_the_stem() -> None:
    masks = context_masks(make_counts(), None, "lineA")
    assert [ctx for ctx, _ in masks] == ["lineA"]
    assert masks[0][1].all()


def test_context_masks_errors() -> None:
    adata = make_counts()
    with pytest.raises(KeyError, match="cell_line"):
        context_masks(adata, "cell_line", "s")
    adata.obs["context"] = pd.Series(
        [None] + ["c1"] * (N_CELLS - 1), index=adata.obs_names, dtype=object
    )
    with pytest.raises(ValueError, match="missing"):
        context_masks(adata, "context", "s")
