import hashlib
import logging
import os
import subprocess
from pathlib import Path

import pydantic
import pytest
import torch
from omegaconf.errors import MissingMandatoryValue

from pie import utils
from pie.utils import CONFIG_DIR, atomic_dir, compose_config
from tests.conftest import REPO_ROOT

SHARED_KEYS = ("WANDB_ENTITY", "WANDB_PROJECT", "PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR")


def _clear(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Unset env vars so that monkeypatch restores their original state after the test."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )


def test_load_common_env_parses_supported_forms(monkeypatch, tmp_path):
    names = ("PIE_T_A", "PIE_T_B", "PIE_T_C", "PIE_T_D", "PIE_T_E")
    _clear(monkeypatch, *names)
    path = tmp_path / "common.sh"
    path.write_text(
        "# comment\n"
        "\n"
        "PIE_T_A=plain\n"
        'export PIE_T_B="double quoted"\n'
        "  export   PIE_T_C='single'\n"
        "PIE_T_D=\n"
        "PIE_T_E=a=b\n"
    )
    assert utils.load_common_env(path) == {
        "PIE_T_A": "plain",
        "PIE_T_B": "double quoted",
        "PIE_T_C": "single",
        "PIE_T_D": "",
        "PIE_T_E": "a=b",
    }
    assert os.environ["PIE_T_B"] == "double quoted"
    assert os.environ["PIE_T_D"] == ""


def test_load_common_env_existing_environment_wins(monkeypatch, tmp_path):
    _clear(monkeypatch, "PIE_T_B")
    monkeypatch.setenv("PIE_T_A", "from-env")
    path = tmp_path / "common.sh"
    path.write_text("PIE_T_A=from-file\nPIE_T_B=b\n")
    assert utils.load_common_env(path) == {"PIE_T_B": "b"}
    assert os.environ["PIE_T_A"] == "from-env"


def test_load_common_env_missing_file_returns_empty(tmp_path):
    assert utils.load_common_env(tmp_path / "absent.sh") == {}


def test_load_common_env_rejects_malformed_line_and_applies_nothing(monkeypatch, tmp_path):
    _clear(monkeypatch, "PIE_T_A")
    path = tmp_path / "common.sh"
    path.write_text("PIE_T_A=1\nsource other.sh\n")
    with pytest.raises(ValueError, match=r"common\.sh:2"):
        utils.load_common_env(path)
    assert "PIE_T_A" not in os.environ


@pytest.mark.parametrize(
    "line",
    [
        "PIE_T_A=$HOME/runs",
        "PIE_T_A=${HOME}/runs",
        'PIE_T_A="$HOME/runs"',
        "PIE_T_A=~/runs",
        "PIE_T_A=/data  # mine",
        "PIE_T_A=/data\t#mine",
        "PIE_T_A=two words",
    ],
)
def test_load_common_env_rejects_shell_syntax_it_does_not_implement(monkeypatch, tmp_path, line):
    _clear(monkeypatch, "PIE_T_A")
    path = tmp_path / "common.sh"
    path.write_text(f"{line}\n")
    with pytest.raises(ValueError, match=r"common\.sh:1"):
        utils.load_common_env(path)
    assert "PIE_T_A" not in os.environ


def test_load_common_env_single_quoted_values_are_literal(monkeypatch, tmp_path):
    _clear(monkeypatch, "PIE_T_A", "PIE_T_B")
    path = tmp_path / "common.sh"
    path.write_text("PIE_T_A='$HOME/x # y'\nPIE_T_B='~/z'\n")
    assert utils.load_common_env(path) == {"PIE_T_A": "$HOME/x # y", "PIE_T_B": "~/z"}


def test_require_env_returns_values(monkeypatch):
    monkeypatch.setenv("PIE_T_A", "1")
    monkeypatch.setenv("PIE_T_B", "two")
    assert utils.require_env("PIE_T_A", "PIE_T_B") == {"PIE_T_A": "1", "PIE_T_B": "two"}


def test_require_env_lists_every_missing_or_empty_name(monkeypatch):
    _clear(monkeypatch, "PIE_T_A", "PIE_T_C")
    monkeypatch.setenv("PIE_T_B", "")
    monkeypatch.setenv("PIE_T_D", "ok")
    with pytest.raises(utils.MissingEnvError) as info:
        utils.require_env("PIE_T_A", "PIE_T_B", "PIE_T_D", "PIE_T_C")
    assert info.value.names == ["PIE_T_A", "PIE_T_B", "PIE_T_C"]
    assert str(info.value) == (
        "missing environment variables: PIE_T_A, PIE_T_B, PIE_T_C (export them, or set them in "
        "$PIE_ENV_FILE, ./common.sh or ~/.config/pie/common.sh)"
    )


def test_missing_api_key_is_never_sent_to_common_sh(monkeypatch):
    _clear(monkeypatch, "OPENAI_API_KEY")
    with pytest.raises(utils.MissingEnvError) as info:
        utils.require_env("OPENAI_API_KEY")
    assert str(info.value) == (
        "missing environment variables: OPENAI_API_KEY "
        "(export OPENAI_API_KEY in your environment; never put it in common.sh)"
    )


class _Point(utils.StrictModel):
    x: int


def test_strict_model_rejects_unknown_keys():
    assert _Point(x=1).x == 1
    with pytest.raises(pydantic.ValidationError):
        _Point(x=1, y=2)


def test_common_sh_example_lists_exactly_the_shared_keys(monkeypatch):
    _clear(monkeypatch, *SHARED_KEYS, "NCBI_API_KEY", "NCBI_EMAIL")
    example = REPO_ROOT / "common.sh.example"
    assert utils.load_common_env(example) == dict.fromkeys(SHARED_KEYS, "")
    text = example.read_text()
    assert "# NCBI_API_KEY=" in text
    assert "# NCBI_EMAIL=" in text
    assert "literal" in text
    assert "OPENAI_API_KEY" not in text


def test_unfilled_common_sh_fails_fast_on_every_root(monkeypatch):
    _clear(monkeypatch, *SHARED_KEYS, "NCBI_API_KEY")
    utils.load_common_env(REPO_ROOT / "common.sh.example")
    with pytest.raises(utils.MissingEnvError) as info:
        utils.require_env(*utils.ENV_ROOTS)
    assert info.value.names == list(utils.ENV_ROOTS)


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git checkout")
def test_common_sh_is_gitignored():
    assert _git("check-ignore", "-q", "common.sh").returncode == 0
    assert _git("check-ignore", "-q", "common.sh.example").returncode == 1


def test_setup_logging_sets_root_level_and_quiets_http_clients():
    root = logging.getLogger()
    previous = root.level
    try:
        utils.setup_logging("DEBUG")
        assert root.level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
    finally:
        root.setLevel(previous)


def test_configure_determinism_sets_cublas_and_seeds_torch(monkeypatch):
    _clear(monkeypatch, "CUBLAS_WORKSPACE_CONFIG", "PL_GLOBAL_SEED", "PL_SEED_WORKERS")
    utils.configure_determinism(123)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["PL_SEED_WORKERS"] == "1"
    first = torch.rand(4)
    utils.configure_determinism(123)
    assert torch.equal(first, torch.rand(4))


def test_env_global_rank(monkeypatch):
    _clear(monkeypatch, "RANK", "NODE_RANK", "LOCAL_RANK")
    assert utils.env_global_rank(4) == 0
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert utils.env_global_rank(4) == 6
    monkeypatch.setenv("RANK", "3")
    assert utils.env_global_rank(4) == 3


ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_bytes_known_vector():
    assert utils.sha256_bytes(b"abc") == ABC_SHA256


def test_sha256_file_is_independent_of_chunk_size(tmp_path):
    data = bytes(range(256)) * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert utils.sha256_file(path) == expected
    assert utils.sha256_file(path, chunk_bytes=7) == expected


def test_canonical_json_is_sorted_compact_utf8():
    expected = '{"a":"é","b":[1,2]}'.encode()
    assert utils.canonical_json({"b": [1, 2], "a": "é"}) == expected


def test_atomic_dir_publishes_on_success(tmp_path):
    final = tmp_path / "out"
    with utils.atomic_dir(final) as tmp:
        assert tmp.parent == tmp_path
        assert tmp.name.startswith(".out.tmp-")
        (tmp / "a.txt").write_text("x")
        assert not final.exists()
    assert (final / "a.txt").read_text() == "x"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_atomic_dir_removes_tmp_on_error(tmp_path):
    final = tmp_path / "out"
    with pytest.raises(RuntimeError, match="boom"), utils.atomic_dir(final) as tmp:
        (tmp / "a.txt").write_text("x")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_atomic_dir_refuses_non_empty_target(tmp_path):
    final = tmp_path / "out"
    final.mkdir()
    (final / "keep").write_text("k")
    with pytest.raises(FileExistsError), utils.atomic_dir(final):
        pass
    assert (final / "keep").read_text() == "k"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_atomic_dir_replaces_empty_target(tmp_path):
    final = tmp_path / "out"
    final.mkdir()
    with utils.atomic_dir(final) as tmp:
        (tmp / "a.txt").write_text("1")
    assert (final / "a.txt").read_text() == "1"


def test_atomic_dir_first_writer_wins(tmp_path):
    final = tmp_path / "out"
    with utils.atomic_dir(final) as tmp:
        (tmp / "who").write_text("second")
        final.mkdir()
        (final / "who").write_text("first")
    assert (final / "who").read_text() == "first"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_write_json_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "sub" / "stats.json"
    utils.write_json(path, {"b": 1, "a": [1.5, "x"]})
    assert path.read_text() == '{\n "b": 1,\n "a": [\n  1.5,\n  "x"\n ]\n}\n'
    utils.write_json(path, {"b": 2})
    assert utils.read_json(path) == {"b": 2}
    assert sorted(p.name for p in path.parent.iterdir()) == ["stats.json"]


def test_atomic_write_text_overwrites_without_leftovers(tmp_path):
    path = tmp_path / "id.txt"
    utils.atomic_write_text(path, "one")
    utils.atomic_write_text(path, "two")
    assert path.read_text() == "two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["id.txt"]


def test_to_portable_and_resolve_path_round_trip(monkeypatch, tmp_path):
    data = tmp_path / "data"
    runs = data / "runs"
    cache = tmp_path / "cache"
    monkeypatch.setenv("PIE_DATA_ROOT", str(data))
    monkeypatch.setenv("PIE_RUNS_ROOT", str(runs))
    monkeypatch.setenv("PIE_CACHE_DIR", str(cache))
    cases = {
        data / "preprocessed" / "replogle": "${PIE_DATA_ROOT}/preprocessed/replogle",
        runs / "exp" / "last.ckpt": "${PIE_RUNS_ROOT}/exp/last.ckpt",
        cache: "${PIE_CACHE_DIR}",
    }
    for path, portable in cases.items():
        assert utils.to_portable(path) == portable
        assert utils.to_portable(str(path)) == portable
        assert utils.resolve_path(portable) == path
    outside = Path("/opt/elsewhere/file.txt")
    assert utils.to_portable(outside) == str(outside)
    assert utils.resolve_path(outside) == outside
    assert utils.to_portable("${PIE_DATA_ROOT}/x") == "${PIE_DATA_ROOT}/x"


def test_to_portable_keeps_symlinks_under_the_root(monkeypatch, tmp_path):
    data = tmp_path / "data"
    target = tmp_path / "elsewhere"
    data.mkdir()
    target.mkdir()
    (data / "link").symlink_to(target)
    monkeypatch.setenv("PIE_DATA_ROOT", str(data))
    assert utils.to_portable(data / "link" / "meta.json") == "${PIE_DATA_ROOT}/link/meta.json"


def test_to_portable_does_not_match_a_sibling_with_a_shared_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("PIE_DATA_ROOT", str(tmp_path / "data"))
    sibling = tmp_path / "data2" / "meta.json"
    assert utils.to_portable(sibling) == str(sibling)


def test_resolve_path_requires_the_env_root(monkeypatch):
    _clear(monkeypatch, "PIE_CACHE_DIR")
    with pytest.raises(utils.MissingEnvError):
        utils.resolve_path("${PIE_CACHE_DIR}/evidence")


def _configs(root: Path) -> Path:
    cfg = root / "cfg"
    (cfg / "grp" / "opt").mkdir(parents=True)
    (cfg / "main.yaml").write_text(
        "defaults:\n  - _self_\n  - grp/opt: null\n\na: 1\nb: ${a}\nname: x\n"
    )
    (cfg / "grp" / "opt" / "one.yaml").write_text("# @package _global_\na: 2\n")
    (cfg / "need.yaml").write_text("x: ???\n")
    return cfg


def test_config_dir_is_the_packaged_configs_dir() -> None:
    assert Path(utils.__file__).resolve().parent / "configs" == CONFIG_DIR


def test_compose_config_resolves_and_rewrites_group_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _configs(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert compose_config("main", [], config_dir=cfg) == {"a": 1, "b": 1, "name": "x"}
    short = compose_config("main", ["opt=one"], groups={"opt": "grp/opt"}, config_dir=cfg)
    assert short == {"a": 2, "b": 2, "name": "x"}
    assert compose_config("main", ["grp/opt=one"], config_dir=cfg) == short
    # Only the exact key is rewritten; other keys pass through untouched.
    assert compose_config("main", ["name=opt"], groups={"opt": "grp/opt"}, config_dir=cfg)[
        "name"
    ] == "opt"
    assert list(work.iterdir()) == []  # no outputs/, no .hydra/


def test_compose_config_rejects_missing_values(tmp_path: Path) -> None:
    with pytest.raises(MissingMandatoryValue):
        compose_config("need", [], config_dir=_configs(tmp_path))


def test_atomic_dir_overwrite_replaces_only_after_success(tmp_path: Path) -> None:
    final = tmp_path / "out"
    final.mkdir()
    (final / "old.txt").write_text("old")
    with pytest.raises(FileExistsError), atomic_dir(final):
        pass
    with pytest.raises(RuntimeError), atomic_dir(final, overwrite=True) as tmp:
        (tmp / "new.txt").write_text("new")
        raise RuntimeError("boom")
    assert sorted(p.name for p in final.iterdir()) == ["old.txt"]
    with atomic_dir(final, overwrite=True) as tmp:
        (tmp / "new.txt").write_text("new")
    assert sorted(p.name for p in final.iterdir()) == ["new.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]  # no tmp or trash left


def test_relative_paths_resolve_against_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert utils.resolve_path("runs/x") == tmp_path / "runs" / "x"


def test_to_portable_makes_relative_paths_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in utils.ENV_ROOTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert utils.to_portable("splits/train.json") == str(tmp_path / "splits" / "train.json")


def test_common_env_search_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.delenv("PIE_ENV_FILE", raising=False)
    assert utils.common_env_candidates() == [
        work / "common.sh",
        home / ".config" / "pie" / "common.sh",
    ]
    explicit = tmp_path / "env.sh"
    monkeypatch.setenv("PIE_ENV_FILE", str(explicit))
    assert utils.common_env_candidates() == [explicit]


def test_explicit_env_file_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIE_ENV_FILE", str(tmp_path / "missing.sh"))
    with pytest.raises(FileNotFoundError, match="PIE_ENV_FILE"):
        utils.load_common_env()


def test_first_existing_common_sh_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / ".config" / "pie").mkdir(parents=True)
    (home / ".config" / "pie" / "common.sh").write_text("PIE_TEST_VALUE=home\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "common.sh").write_text("PIE_TEST_VALUE=work\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.delenv("PIE_ENV_FILE", raising=False)
    monkeypatch.delenv("PIE_TEST_VALUE", raising=False)
    assert utils.load_common_env() == {"PIE_TEST_VALUE": "work"}
