"""Tests for the STRING SPACE builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.sources.inputs import STRING_EMBEDDINGS, FakeSession, write_string_files

from pie.sources import string as string_source


def test_build_keeps_the_first_protein_per_name_sorted_by_name(tmp_path: Path) -> None:
    files = write_string_files(tmp_path, "v12.0")
    keys, embeddings = string_source.build_string_space(files["space_h5"], files["protein_info"])
    assert keys == ["AAA", "BBB", "CCC"]
    assert embeddings.dtype == np.float16
    # AAA = P2 (h5 row 2), BBB = P3 (row 0, first of P3/P1), CCC = P4 (row 3); PX has no name
    np.testing.assert_array_equal(embeddings, STRING_EMBEDDINGS[[2, 0, 3]])


def test_string_urls_and_fetch(tmp_path: Path) -> None:
    urls = string_source.string_urls("v12.0")
    base = "https://stringdb-downloads.org/download"
    assert urls == {
        "space_h5": f"{base}/protein.network.embeddings.v12.0/"
        "9606.protein.network.embeddings.v12.0.h5",
        "protein_info": f"{base}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    }
    session = FakeSession({url: b"x" for url in urls.values()})
    files = string_source.fetch_string("v12.0", tmp_path, session=session)
    assert files == {
        "space_h5": tmp_path / "9606.protein.network.embeddings.v12.0.h5",
        "protein_info": tmp_path / "9606.protein.info.v12.0.txt.gz",
    }
