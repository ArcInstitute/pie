"""Tests for the shared download and provenance helpers of the source tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.sources.inputs import FakeSession

import pie
from pie.sources.common import (
    base_provenance,
    download_file,
    environment_record,
    headers_path,
    input_record,
    provenance_path,
)
from pie.utils import REPO_ROOT, sha256_bytes

URL = "https://x.test/a.txt"


def test_download_writes_the_file_and_recorded_headers(tmp_path: Path) -> None:
    session = FakeSession({URL: b"hello"}, headers={"last-modified": "Mon", "server": "x"})
    dest = download_file(URL, tmp_path / "c" / "a.txt", session=session)
    assert dest == tmp_path / "c" / "a.txt"
    assert dest.read_bytes() == b"hello"
    assert json.loads(headers_path(dest).read_text()) == {"last-modified": "Mon"}
    assert session.calls == [(URL, None)]
    assert sorted(p.name for p in dest.parent.iterdir()) == ["a.txt", "a.txt.headers.json"]


def test_download_skips_an_existing_file(tmp_path: Path) -> None:
    dest = tmp_path / "a.txt"
    dest.write_bytes(b"cached")
    session = FakeSession({})
    assert download_file(URL, dest, session=session) == dest
    assert session.calls == []
    assert dest.read_bytes() == b"cached"


def test_download_passes_query_params(tmp_path: Path) -> None:
    session = FakeSession({"https://x.test/q": b"rows"})
    download_file("https://x.test/q", tmp_path / "q.tsv", session=session, params={"format": "tsv"})
    assert session.calls == [("https://x.test/q", {"format": "tsv"})]


def test_download_rejects_a_bad_checksum_and_leaves_nothing(tmp_path: Path) -> None:
    session = FakeSession({URL: b"data"})
    with pytest.raises(ValueError, match="does not match"):
        download_file(URL, tmp_path / "a.txt", session=session, sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_download_verifies_a_cached_file(tmp_path: Path) -> None:
    dest = tmp_path / "a.txt"
    dest.write_bytes(b"data")
    good = sha256_bytes(b"data")
    assert download_file(URL, dest, session=FakeSession({}), sha256=good) == dest
    with pytest.raises(ValueError, match="does not match"):
        download_file(URL, dest, session=FakeSession({}), sha256="0" * 64)


def test_download_http_error_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="404"):
        download_file("https://x.test/missing", tmp_path / "m", session=FakeSession({}))
    assert list(tmp_path.iterdir()) == []


def test_offline_download_uses_the_cache_and_never_the_network(tmp_path: Path) -> None:
    session = FakeSession({URL: b"hello"})
    with pytest.raises(FileNotFoundError, match="offline"):
        download_file(URL, tmp_path / "a.txt", session=session, offline=True)
    assert session.calls == []
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "a.txt").write_bytes(b"cached")
    cached = download_file(URL, tmp_path / "a.txt", session=session, offline=True)
    assert cached.read_bytes() == b"cached"
    assert session.calls == []


def test_input_record_prefers_explicit_then_recorded_release(tmp_path: Path) -> None:
    session = FakeSession({URL: b"abc"}, headers={"x-uniprot-release": "r1", "last-modified": "T"})
    path = download_file(URL, tmp_path / "u.tsv", session=session)
    assert input_record(path, URL) == {"url": URL, "sha256": sha256_bytes(b"abc"), "release": "r1"}
    assert input_record(path, URL, release="v2")["release"] == "v2"
    local = tmp_path / "local.csv"
    local.write_bytes(b"x")
    assert input_record(local, None) == {"url": None, "sha256": sha256_bytes(b"x"), "release": None}


def test_base_provenance_and_environment_record() -> None:
    provenance = base_provenance({"a": {"url": None}}, {"k": 1}, model="m")
    assert provenance == {
        "tool_version": pie.__version__,
        "inputs": {"a": {"url": None}},
        "params": {"k": 1},
        "model": "m",
    }
    env = environment_record("cpu")
    assert env["device"] == "cpu"
    assert env["torch"]
    assert "accelerator" not in env


def test_provenance_path_records_no_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_DATA_ROOT", str(tmp_path / "data"))
    assert provenance_path(tmp_path / "data" / "sources" / "x") == "${PIE_DATA_ROOT}/sources/x"
    assert provenance_path(REPO_ROOT / "data" / "sources" / "contexts") == "data/sources/contexts"
    assert provenance_path(tmp_path / "elsewhere" / "drugs.csv") == "drugs.csv"
    assert provenance_path("${PIE_DATA_ROOT}/y") == "${PIE_DATA_ROOT}/y"
