"""Package metadata and the files the wheel must carry."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from importlib.resources import files

from tests.conftest import REPO_ROOT

PROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]


def test_distribution_name_and_version() -> None:
    import pie

    assert PROJECT["name"] == "arc-pie"
    assert pie.__version__ == version("arc-pie") == PROJECT["version"]


def test_no_exact_or_direct_url_dependencies() -> None:
    deps = list(PROJECT["dependencies"])
    for extra in PROJECT["optional-dependencies"].values():
        deps += extra
    assert not [d for d in deps if "==" in d or " @ " in d]
    assert all(">=" in d for d in deps)


def test_process_extra_uses_gpudge_from_pypi() -> None:
    assert PROJECT["optional-dependencies"]["process"] == ["gpudge[fast]>=0.9.1,<0.10"]


def test_packaged_resources() -> None:
    assert (files("pie") / "configs" / "train.yaml").is_file()
    assert (files("pie") / "configs" / "experiment" / "replogle_xdataset.yaml").is_file()
    assert (files("pie.data") / "legacy_aliases.yaml").is_file()
