"""Remote assets use real PIE files with a synthetic Hub transport."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pie import assets
from pie.data.preprocessed import PreprocessedDir
from pie.sources.contract import read_source
from pie.utils import MissingEnvError, to_portable
from tests.conftest import REPO_ROOT
from tests.fixtures import TinyData

SHA = "a" * 40
REPO = "test/pie"
BASE = f"hf://datasets/{REPO}@{SHA}"


@pytest.fixture
def hub(tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    remote = tmp_path / "remote"
    shutil.copytree(tiny_data.preprocessed["alpha"], remote / "preprocessed")
    shutil.copytree(tiny_data.sources["ncbi_text"], remote / "ncbi_text")
    shutil.copytree(tiny_data.gene_text, remote / "gene_text")
    (remote / "de").mkdir()
    (remote / "de" / "unused.parquet").write_bytes(b"do not download")
    data_root = tmp_path / "downloaded"
    monkeypatch.setenv("PIE_DATA_ROOT", str(data_root))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls: list[dict[str, Any]] = []
    lookups: list[dict[str, Any]] = []

    def info(**kwargs: Any) -> SimpleNamespace:
        lookups.append(kwargs)
        return SimpleNamespace(sha=SHA)

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        dest = Path(kwargs["local_dir"])
        patterns = kwargs["allow_patterns"]
        if isinstance(patterns, str):
            patterns = [patterns]
        for path in remote.rglob("*"):
            rel = path.relative_to(remote)
            if path.is_file() and any(fnmatch.fnmatch(str(rel), p) for p in patterns):
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return str(dest)

    monkeypatch.setattr(assets, "snapshot_download", download)
    monkeypatch.setattr(assets, "HfApi", lambda: SimpleNamespace(repo_info=info))
    return SimpleNamespace(
        remote=remote, root=data_root, calls=calls, lookups=lookups, download=download
    )


def test_local_paths_need_no_hub_or_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIE_DATA_ROOT", raising=False)
    assert assets.resolve_asset("data/local", kind="preprocessed") == REPO_ROOT / "data/local"
    assert assets.pin_asset_reference("data/local") == "data/local"


def test_resolver_imports_on_platform_without_fcntl() -> None:
    script = '''
import builtins
original_import = builtins.__import__
def platform_import(name, *args, **kwargs):
    if name == "fcntl":
        raise ImportError("fcntl is unavailable on this platform")
    return original_import(name, *args, **kwargs)
builtins.__import__ = platform_import
import pie.assets
'''
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_downloads_selected_directory_and_reuses_it_offline(hub: Any) -> None:
    reference = f"{BASE}/preprocessed"
    path = assets.resolve_asset(reference, kind="preprocessed")
    assert path == hub.root / "hf/datasets/test/pie" / SHA / "preprocessed"
    assert PreprocessedDir.open(path).dataset == "alpha"
    assert not (path.parent / "de").exists()
    assert len(hub.calls) == 1
    assert hub.calls[0]["revision"] == SHA
    assert not hub.lookups
    assert assets.resolve_asset(reference, kind="preprocessed") == path
    assert len(hub.calls) == 1


def test_sources_share_repository_without_downloading_other_assets(hub: Any) -> None:
    token = assets.resolve_asset(f"{BASE}/ncbi_text", kind="source")
    gene = assets.resolve_asset(f"{BASE}/gene_text", kind="source")
    assert token.parent == gene.parent
    assert read_source(token).offsets is not None
    assert read_source(gene).meta.name == "gene_text"
    assert not (gene.parent / "preprocessed").exists()
    assert len(hub.calls) == 2


def test_unpinned_reference_freezes_once_per_data_root(hub: Any) -> None:
    reference = f"hf://datasets/{REPO}/preprocessed"
    pinned = assets.pin_asset_reference(reference)
    assert pinned == f"{BASE}/preprocessed"
    assert assets.pin_asset_reference(reference) == pinned
    assert assets.resolve_asset(reference, kind="preprocessed").is_dir()
    assert len(hub.lookups) == 1
    assert hub.lookups[0]["revision"] == "main"


def test_tag_is_pinned_and_portable(hub: Any) -> None:
    reference = f"hf://datasets/{REPO}@v1/preprocessed"
    assert assets.pin_asset_reference(reference) == f"{BASE}/preprocessed"
    assert hub.lookups[0]["revision"] == "v1"
    assert to_portable(reference) == reference


@pytest.mark.parametrize("content", ["", "not json", "[]", "{}", '{"commit":"bad"}'])
def test_corrupt_revision_record_requires_explicit_commit(content: str, hub: Any) -> None:
    reference = f"hf://datasets/{REPO}/preprocessed"
    assets.pin_asset_reference(reference)
    record = next((hub.root / "hf" / ".refs").glob("*.json"))
    record.write_text(content)
    with pytest.raises(ValueError, match="explicit commit"):
        assets.pin_asset_reference(reference)
    # Re-resolving main here could change the data silently. Pinned references bypass the record.
    assert len(hub.lookups) == 1
    assert assets.pin_asset_reference(f"{BASE}/preprocessed") == f"{BASE}/preprocessed"


@pytest.mark.parametrize("reference", [
    "hf://models/test/pie", "hf://datasets/test", "hf://datasets/../pie",
    "hf://datasets/test/pie/../secret", "hf://datasets/test/pie/%2e%2e/secret",
    "hf://datasets/test/pie//secret", "hf://datasets/test/pie@/preprocessed",
    "hf://datasets/test/pie/preprocessed?x=1", "hf://datasets/test/pie/a\\b",
])
def test_invalid_references_fail_before_network(reference: str, hub: Any) -> None:
    with pytest.raises(ValueError, match="HF reference"):
        assets.resolve_asset(reference, kind="preprocessed")
    assert not hub.calls and not hub.lookups


def test_missing_data_root_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIE_DATA_ROOT", raising=False)
    with pytest.raises(MissingEnvError, match="PIE_DATA_ROOT"):
        assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed")


def test_incomplete_download_does_not_get_completion_manifest(hub: Any) -> None:
    (hub.remote / "preprocessed" / "tested.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"tested\.npy"):
        assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed")
    shutil.copyfile(
        hub.remote / "preprocessed" / "fdr.npy", hub.remote / "preprocessed" / "tested.npy"
    )
    with pytest.raises(ValueError, match="dtype"):
        assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed")
    assert len(hub.calls) == 2


def test_truncated_file_is_detected_and_repaired(hub: Any) -> None:
    reference = f"{BASE}/preprocessed"
    path = assets.resolve_asset(reference, kind="preprocessed")
    (path / "tested.npy").write_bytes(b"interrupted")
    assert assets.resolve_asset(reference, kind="preprocessed") == path
    assert PreprocessedDir.open(path).tested.dtype.name == "bool"
    assert len(hub.calls) == 2
    assert hub.calls[-1].get("force_download") is True


def test_cached_assets_work_with_offline_mode(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = f"{BASE}/preprocessed"
    path = assets.resolve_asset(reference, kind="preprocessed")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert assets.resolve_asset(reference, kind="preprocessed") == path
    with pytest.raises(FileNotFoundError, match="offline"):
        assets.resolve_asset(f"{BASE}/gene_text", kind="source")
    with pytest.raises(FileNotFoundError, match="offline"):
        assets.pin_asset_reference(f"hf://datasets/{REPO}@new/preprocessed")
    assert len(hub.calls) == 1


def test_interrupted_download_can_be_retried(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(**kwargs: Any) -> str:
        hub.download(**kwargs)
        raise ConnectionError("connection interrupted")

    monkeypatch.setattr(assets, "snapshot_download", interrupted)
    with pytest.raises(RuntimeError, match="connection interrupted"):
        assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed")
    monkeypatch.setattr(assets, "snapshot_download", hub.download)
    assert assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed").is_dir()
    assert len(hub.calls) == 2


def test_concurrent_callers_download_once(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(**kwargs: Any) -> str:
        time.sleep(0.05)
        return hub.download(**kwargs)

    monkeypatch.setattr(assets, "snapshot_download", slow)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(
            lambda _: assets.resolve_asset(f"{BASE}/preprocessed", kind="preprocessed"), range(4)
        ))
    assert len(set(paths)) == 1
    assert len(hub.calls) == 1


def test_remote_config_round_trip_does_not_download(hub: Any, tiny_data: TinyData) -> None:
    from pie.config import portable_train_config, resolved_train_config
    from tests.pipeline import tiny_train_config

    cfg = tiny_train_config(tiny_data, hub.root / "run")
    cfg.data.preprocessed_dirs = [f"{BASE}/preprocessed"]
    cfg.data.source_dirs["ncbi_text"] = f"{BASE}/ncbi_text"
    cfg.data.gene_text_dir = f"{BASE}/gene_text"
    restored = resolved_train_config(portable_train_config(cfg))
    assert restored.data.preprocessed_dirs == cfg.data.preprocessed_dirs
    assert restored.data.source_dirs == cfg.data.source_dirs
    assert restored.data.gene_text_dir == cfg.data.gene_text_dir
    assert not hub.calls and not hub.lookups


@pytest.mark.parametrize("revision", ["v1", "b" * 40])
def test_resume_replays_prior_commit_without_resolving_branch(
    revision: str, hub: Any, tiny_data: TinyData
) -> None:
    from pie.config import pinned_train_config
    from tests.pipeline import tiny_train_config

    cfg = tiny_train_config(tiny_data, hub.root / "run")
    cfg.data.preprocessed_dirs = [f"hf://datasets/{REPO}@{revision}/preprocessed"]
    previous = cfg.model_copy(deep=True)
    previous.data.preprocessed_dirs = [f"{BASE}/preprocessed"]
    pinned = pinned_train_config(cfg, previous=previous)
    assert pinned.data.preprocessed_dirs == previous.data.preprocessed_dirs
    assert not hub.lookups


def test_processes_sharing_data_root_download_once(
    hub: Any, tmp_path: Path
) -> None:
    count = tmp_path / "download_calls.txt"
    # Fresh interpreters exercise process locking without inheriting mocks or using Unix fork.
    script = f'''
import shutil
import time
from pathlib import Path
from pie import assets

def download(**kwargs):
    with Path({str(count)!r}).open("a") as handle:
        handle.write("download\\n")
    time.sleep(0.05)
    out = Path(kwargs["local_dir"]) / "preprocessed"
    shutil.copytree(Path({str(hub.remote)!r}) / "preprocessed", out, dirs_exist_ok=True)
    return str(out.parent)

assets.snapshot_download = download
print(assets.resolve_asset({f"{BASE}/preprocessed"!r}, kind="preprocessed"))
'''
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", script], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(3)
    ]
    try:
        outputs = []
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr
            outputs.append(stdout.strip())
        assert len(set(outputs)) == 1
        assert count.read_text() == "download\n"
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.communicate(timeout=5)


def test_source_build_and_verify_accept_remote_datasets(
    hub: Any, tiny_data: TinyData, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pie.sources import registry
    from pie.sources.config import compose_sources_config

    def build(ctx: Any) -> Path:
        assert [d.dataset for d in ctx.datasets] == ["alpha"]
        out = ctx.out_root / "esm2"
        shutil.copytree(tiny_data.sources["esm2"], out)
        return out

    monkeypatch.setitem(registry.TOOLS, "esm2", registry.SourceTool("esm2", (), False, build))
    cfg = compose_sources_config([
        "tools=[esm2]", "with_deps=false", f"preprocessed_dirs=[{BASE}/preprocessed]",
        f"output_root={tmp_path / 'built_sources'}",
    ])
    out = registry.build_sources(cfg)
    assert read_source(out["esm2"]).meta.name == "esm2"
    report = registry.verify_command(cfg.model_copy(update={"mode": "verify"}))
    assert "alpha" in report["coverage"]["esm2"]
    assert len(hub.calls) == 1


def test_training_checkpoint_and_prediction_relocate_remote_assets(
    hub: Any, tiny_data: TinyData, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pie.config import EvalConfig, InferConfig
    from pie.evaluate import run_eval
    from pie.infer import run_infer
    from pie.predict import load_checkpoint
    from pie.train import run_train
    from tests.pipeline import tiny_train_config

    cfg = tiny_train_config(tiny_data, tmp_path / "runs" / "remote")
    cfg.data.preprocessed_dirs[0] = f"hf://datasets/{REPO}@v1/preprocessed"
    cfg.data.source_dirs["ncbi_text"] = f"hf://datasets/{REPO}@v1/ncbi_text"
    cfg.data.gene_text_dir = f"hf://datasets/{REPO}@v1/gene_text"
    run = run_train(cfg)
    loaded = load_checkpoint(run / "last.ckpt")
    assert loaded.config.data.preprocessed_dirs[0] == f"{BASE}/preprocessed"
    assert loaded.config.data.source_dirs["ncbi_text"] == f"{BASE}/ncbi_text"
    assert loaded.config.data.gene_text_dir == f"{BASE}/gene_text"
    assert len(hub.calls) == 3
    assert f"{BASE}/preprocessed" in (run / "config.yaml").read_text()
    monkeypatch.setenv("PIE_DATA_ROOT", str(tmp_path / "other_machine"))
    # Loading a checkpoint is read-only; runtime fetches its pinned assets at the new root.
    load_checkpoint(run / "last.ckpt")
    assert len(hub.calls) == 3
    override = [f"{BASE}/preprocessed", str(tiny_data.preprocessed["beta"])]
    out = run_eval(EvalConfig(
        experiment_name="remote", run_dir=str(run), ckpt="last",
        split_path=str(tiny_data.split_dir / "test.json"), row_set="remote_test",
        preprocessed_dirs=None, save_predictions=False, overwrite=False,
        device="cpu", batch_size=2,
    ))
    assert (out / "metrics_last.csv").is_file()
    predictions = run_infer(InferConfig(
        experiment_name="remote", run_dir=str(run), ckpt="last", rows_kind="split",
        rows_path=str(tiny_data.split_dir / "test.json"), preprocessed_dirs=override,
        output_path=str(tmp_path / "predictions.parquet"), overwrite=False,
        device="cpu", batch_size=2,
    ))
    assert predictions.is_file()
    assert len(hub.calls) == 6


def _splits_uri(rest: str = "exp/fold") -> str:
    return f"hf://datasets/arcinstitute/PIE_splits@{SHA}/{rest}"


def _remote_split_dir(hub: Any, rest: str = "exp/fold") -> None:
    dest = hub.remote / rest
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.json").write_text('{"replogle.k562": ["AAAS"]}')
    (dest / "test.json").write_text('{"replogle.k562": ["AAMP"]}')


def test_resolve_split_dir_downloads_once(hub: Any) -> None:
    _remote_split_dir(hub)
    path = assets.resolve_asset(_splits_uri(), kind="splits")
    assert sorted(p.name for p in path.glob("*.json")) == ["test.json", "train.json"]
    assert len(hub.calls) == 1
    assert assets.resolve_asset(_splits_uri(), kind="splits") == path
    assert len(hub.calls) == 1  # the completion manifest is reused


def test_resolve_split_file(hub: Any) -> None:
    _remote_split_dir(hub)
    path = assets.resolve_split_file(f"{_splits_uri()}/test.json")
    assert path.name == "test.json" and path.is_file()


def test_split_file_reference_must_name_a_json_file(hub: Any) -> None:
    with pytest.raises(ValueError, match=r"must name a \.json split file"):
        assets.resolve_split_file(_splits_uri())


def test_missing_split_file_in_downloaded_dir(hub: Any) -> None:
    _remote_split_dir(hub)
    with pytest.raises(FileNotFoundError, match=r"val\.json"):
        assets.resolve_split_file(f"{_splits_uri()}/val.json")


def test_invalid_split_json_is_rejected(hub: Any) -> None:
    dest = hub.remote / "exp" / "bad"
    dest.mkdir(parents=True)
    (dest / "train.json").write_text('["not", "a", "mapping"]')
    with pytest.raises(ValueError):
        assets.resolve_asset(_splits_uri("exp/bad"), kind="splits")


def test_offline_split_dir(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _remote_split_dir(hub)
    path = assets.resolve_asset(_splits_uri(), kind="splits")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert assets.resolve_asset(_splits_uri(), kind="splits") == path
    with pytest.raises(FileNotFoundError, match="offline mode"):
        assets.resolve_asset(_splits_uri("exp/other"), kind="splits")


def test_local_split_file_passes_through(tmp_path: Path) -> None:
    f = tmp_path / "test.json"
    f.write_text("{}")
    assert assets.resolve_split_file(f) == f
