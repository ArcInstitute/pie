"""The gene and chemical source tools, run end to end through the registry (encoders faked)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.sources.helpers import make_options
from tests.sources.inputs import (
    DEPMAP_KEYS,
    DEPMAP_VALUES,
    STRING_EMBEDDINGS,
    FakeChemModel,
    FakeChemTokenizer,
    FakeEsmModel,
    FakeEsmTokenizer,
    FakeQwenModel,
    FakeQwenTokenizer,
    chem_metadata,
    fake_inchi_keys,
    release_table,
    write_depmap_csv,
    write_jump_files,
    write_ncbi_ftp,
    write_string_files,
    write_uniprot_tsv,
)

import pie
from pie.sources import chem_profiles, uniprot
from pie.sources.contract import SOURCE_NAMES, read_descriptions, read_source
from pie.sources.embed import chemberta, esm2, qwen
from pie.sources.registry import TOOLS, RunContext, resolve_order
from pie.sources.text.ncbi import NCBI_FTP_FILES, build_genes_table, describe_ncbi_genes
from pie.utils import sha256_file

NEW_TOOLS = (
    "ncbi_text",
    "esm2",
    "string_space",
    "depmap_gene_effect",
    "smiles",
    "l1000_tas",
    "prism_secondary",
    "jump_morphology",
)


def _ctx(tmp_path: Path, datasets: Any = (), **options: Any) -> RunContext:
    return RunContext(
        datasets=list(datasets),
        prior_root=None,
        out_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=make_options(**options),
    )


def _drug_dataset(*perts: str) -> Any:
    return SimpleNamespace(dataset="toy", perts=list(perts), meta=SimpleNamespace(pert_kind="drug"))


def _gene_dataset() -> Any:
    return SimpleNamespace(dataset="genes", perts=["AAA"], meta=SimpleNamespace(pert_kind="gene"))


def test_gene_and_chemical_tools_are_registered() -> None:
    assert list(TOOLS) == list(SOURCE_NAMES)
    for name in NEW_TOOLS:
        tool = TOOLS[name]
        assert (tool.name, tool.deps, tool.uses_openai) == (name, (), False)
    assert resolve_order(["jump_morphology", "string_space", "ncbi_text"]) == [
        "ncbi_text",
        "string_space",
        "jump_morphology",
    ]


def test_string_space_tool(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    files = write_string_files(ctx.cache_dir / "string_space", "v12.0")
    out = TOOLS["string_space"].run(ctx)
    assert out == ctx.out_root / "string_space"
    source = read_source(out)
    assert (source.meta.layout, source.meta.index, source.meta.dtype) == (
        "dense",
        "pert",
        "float16",
    )
    assert source.meta.keys == ["AAA", "BBB", "CCC"]
    np.testing.assert_array_equal(source.embeddings, STRING_EMBEDDINGS[[2, 0, 3]])
    provenance = source.meta.provenance
    assert provenance["tool_version"] == pie.__version__
    assert provenance["inputs"]["space_h5"]["sha256"] == sha256_file(files["space_h5"])
    assert provenance["inputs"]["protein_info"]["release"] == "v12.0"


def test_depmap_tool_requires_the_csv(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="depmap_csv"):
        TOOLS["depmap_gene_effect"].run(_ctx(tmp_path))
    csv = write_depmap_csv(tmp_path / "CRISPRGeneEffect.csv")
    source = read_source(TOOLS["depmap_gene_effect"].run(_ctx(tmp_path, depmap_csv=csv)))
    assert source.meta.keys == DEPMAP_KEYS
    assert source.meta.dtype == "float32"
    np.testing.assert_array_equal(source.embeddings, DEPMAP_VALUES)
    assert source.meta.provenance["params"]["model_ids"] == ["ACH-1", "ACH-2"]
    assert source.meta.provenance["inputs"]["crispr_gene_effect"] == {
        "url": None,
        "sha256": sha256_file(csv),
        "release": None,
    }


def test_ncbi_text_tool_writes_tokens_and_texts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path, device="cpu")
    files = write_ncbi_ftp(ctx.cache_dir / "ncbi_text")
    monkeypatch.setenv("PIE_CACHE_DIR", str(tmp_path / "pie_cache"))

    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        return FakeQwenTokenizer(), FakeQwenModel()

    monkeypatch.setattr(qwen, "_load_model", load)
    out = TOOLS["ncbi_text"].run(ctx)
    source = read_source(out)
    assert (source.meta.layout, source.meta.dtype, source.meta.dim) == ("token", "float16", 3)
    texts = read_descriptions(out)
    assert texts == describe_ncbi_genes(build_genes_table(files))
    assert source.meta.keys == list(texts)
    assert source.offsets is not None
    lengths = np.diff(source.offsets)
    assert lengths.tolist() == [len(texts[key]) for key in source.meta.keys]
    expected = np.array([ord(c) for c in texts["BBB"]], dtype=np.float16)
    np.testing.assert_array_equal(source.tokens("BBB")[:, 0], expected)
    provenance = source.meta.provenance
    assert (provenance["model"], provenance["revision"]) == (qwen.MODEL, qwen.REVISION)
    assert set(provenance["inputs"]) == set(NCBI_FTP_FILES)
    assert provenance["params"]["exclude_sections"] == ["gene_info", "summary"]
    assert list((tmp_path / "pie_cache" / "embed" / "ncbi_text").iterdir()) == []


def test_esm2_tool_embeds_the_matched_proteins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path, device="cpu")
    stream = write_uniprot_tsv(ctx.cache_dir / "esm2" / uniprot.STREAM_FILE)
    gene_info = write_ncbi_ftp(ctx.cache_dir / "esm2")["gene_info"]

    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        return FakeEsmTokenizer(), FakeEsmModel()

    monkeypatch.setattr(esm2, "_load_model", load)
    source = read_source(TOOLS["esm2"].run(ctx))
    assert source.meta.keys == ["AAA", "CCC", "DDD"]
    # MAAA: ids 13, 1, 1, 1 -> 4.0; MCCC -> 5.5; MDDD -> 6.25; mean residue position 2.5
    np.testing.assert_allclose(source.embeddings, [[4.0, 2.5], [5.5, 2.5], [6.25, 2.5]])
    table = uniprot.build_proteins_table(stream, gene_info)
    assert source.meta.provenance["proteins_sha256"] == uniprot.proteins_sha256(table)
    assert source.meta.provenance["revision"] == esm2.REVISION


def test_smiles_tool_needs_metadata_and_a_drug_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    datasets = [_gene_dataset(), _drug_dataset("drugA_1.0uM", "drugB_1.0uM", "drugC_1.0uM")]
    with pytest.raises(ValueError, match="drug_metadata"):
        TOOLS["smiles"].run(_ctx(tmp_path, datasets))
    metadata = tmp_path / "drugs.parquet"
    pd.DataFrame(
        {"drug": ["drugA", "drugB ", "drugC"], "canonical_smiles": ["CC", "N", None]}
    ).to_parquet(metadata)
    with pytest.raises(ValueError, match="no drug dataset"):
        TOOLS["smiles"].run(_ctx(tmp_path, [_gene_dataset()], drug_metadata=metadata))

    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        return FakeChemTokenizer(), FakeChemModel()

    monkeypatch.setattr(chemberta, "_load_model", load)
    ctx = _ctx(tmp_path, datasets, drug_metadata=metadata, device="cpu")
    source = read_source(TOOLS["smiles"].run(ctx))
    assert source.meta.keys == ["drugA_1.0uM", "drugB_1.0uM"]
    np.testing.assert_allclose(source.embeddings, [[34.25, 1.0], [27.0, 1.0]])
    assert source.meta.provenance["params"]["omitted_drugs"] == ["drugC"]


def test_chemical_profile_tool_runs_through_the_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = tmp_path / "drugs.csv"
    chem_metadata().to_csv(metadata, index=False)
    ctx = _ctx(tmp_path, [_drug_dataset("drugA_1.0uM", "drugC_1.0uM")], drug_metadata=metadata)
    files = write_jump_files(ctx.cache_dir / "jump_morphology")
    monkeypatch.setattr(chem_profiles, "RELEASES", release_table(files, "jump_morphology"))
    monkeypatch.setattr(chem_profiles, "_inchi_keys", fake_inchi_keys)
    monkeypatch.setattr(chem_profiles, "_JUMP_COORDS", 2)
    source = read_source(TOOLS["jump_morphology"].run(ctx))
    assert source.meta.keys == ["drugA_1.0uM"]
    np.testing.assert_allclose(source.embeddings, [[2.0, 20.0]])
    assert source.meta.provenance["params"]["axes"] == ["f1", "f2"]
    assert set(source.meta.provenance["inputs"]) == {
        "jump_compound.csv.gz",
        "jump_profiles.parquet",
        "drug_metadata",
    }
