"""Registry order, the OpenAI key gate, and the three text tools end to end (clients faked)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.sources.helpers import make_options

from pie.data import preprocessed as pp
from pie.data.preprocessed import PreprocessedDir
from pie.sources import registry
from pie.sources.contract import DescriptionConflictError, read_descriptions, read_source
from pie.sources.embed import openai as openai_embed
from pie.sources.registry import TOOLS, RunContext, SourceTool, resolve_order
from pie.sources.text import contexts, drugs, genes, tools
from pie.utils import MissingEnvError, sha256_file


def dataset(name: str, kind: str, perts: list[str], genes_: list[str], ctxs: list[str]) -> Any:
    meta = SimpleNamespace(pert_kind=kind, control_label="ctrl")
    return SimpleNamespace(
        dataset=name, perts=perts, genes=genes_, contexts=ctxs, meta=meta, pert_ensembl={}
    )


GENE_SET = dataset("g", "gene", ["TP53", "ctrl", "AAK1"], ["TP53", "GAPDH"], ["k562"])
DRUG_SET = dataset("t", "drug", ["Zeta_5.0uM", "DMSO_TF_0.0uM"], ["GAPDH", "ACTB"], ["hepg2"])


def ctx(
    tmp_path: Path, datasets: list[Any], prior: Path | None = None, **options: Any
) -> RunContext:
    with_paths = []
    for d in datasets:  # fakes get a dir holding the context map that context_text records
        if isinstance(d, SimpleNamespace):
            path = tmp_path / "pre" / d.dataset
            path.mkdir(parents=True, exist_ok=True)
            (path / "contexts.yaml").write_text(f"# {d.dataset}\n")
            d = SimpleNamespace(**{**vars(d), "path": path})
        with_paths.append(d)
    return RunContext(
        datasets=with_paths,
        prior_root=prior,
        out_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=make_options(**options),
    )


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fake every client and the encoder; returns the list of embed calls."""
    calls: list[list[str]] = []

    def embed(texts: list[str], **_kw: Any) -> np.ndarray:
        calls.append(list(texts))
        return np.stack([np.full(openai_embed.DIM, len(t), np.float32) for t in texts])

    def cellosaurus(cache_dir: Path, *_a: Any, **_k: Any) -> Any:
        record = {"release": "56.0", "url": "https://api.cellosaurus.org", "records": {}}
        return SimpleNamespace(cache_dir=cache_dir, provenance=lambda: record)

    def esummary(cache_dir: Path, *_a: Any, **_k: Any) -> Any:
        entry = {"url": str(cache_dir), "sha256": None, "release": None}
        return SimpleNamespace(cache_dir=cache_dir, provenance=lambda: {"ncbi_esummary": entry})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_embed, "embed_texts", embed)
    monkeypatch.setattr(genes.GeneInfoIndex, "load", classmethod(lambda cls, _p: "INDEX"))
    monkeypatch.setattr(genes, "EsummaryClient", esummary)
    monkeypatch.setattr(
        genes, "describe_genetic_perts", lambda keys, *_a: {k: f"gene {k}" for k in keys}
    )
    monkeypatch.setattr(
        drugs, "describe_drug_perts", lambda keys, *_a: {k: f"drug {k}" for k in keys}
    )
    monkeypatch.setattr(contexts, "CellosaurusClient", cellosaurus)
    monkeypatch.setattr(
        contexts,
        "describe_contexts",
        lambda ds, client: {
            c: f"ctx {c} {client.cache_dir.name}" for d in ds for c in d.contexts
        },
    )
    monkeypatch.setattr(
        genes,
        "describe_gene_queries",
        lambda vocab, prior, pert_output, pert_keys, *_a: {
            g: (prior or {}).get(g) or (pert_output[g] if g in pert_keys else f"query {g}")
            for g in vocab
        },
    )
    return calls


def test_text_tools_are_registered_in_contract_order() -> None:
    assert list(TOOLS)[:3] == ["context_text", "perturbation_text", "gene_text"]
    assert all(TOOLS[name].uses_openai for name in list(TOOLS)[:3])
    assert TOOLS["gene_text"].deps == ("perturbation_text",)
    assert resolve_order(["gene_text"]) == ["perturbation_text", "gene_text"]
    assert resolve_order(["gene_text", "context_text"]) == [
        "context_text",
        "perturbation_text",
        "gene_text",
    ]
    with pytest.raises(KeyError, match="nope"):
        resolve_order(["nope"])


def test_registration_api_and_openai_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    def runner(name: str) -> Any:
        return lambda c: ran.append(name) or c.out_root / name

    monkeypatch.setitem(TOOLS, "fake_b", SourceTool("fake_b", ("fake_a",), False, runner("fake_b")))
    monkeypatch.setitem(TOOLS, "fake_a", SourceTool("fake_a", (), False, runner("fake_a")))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_order(["fake_b"]) == ["fake_a", "fake_b"]
    out = registry.run_sources(["fake_b"], ctx(tmp_path, []))
    assert ran == ["fake_a", "fake_b"] and out["fake_b"] == tmp_path / "out" / "fake_b"
    with pytest.raises(MissingEnvError):
        registry.run_sources(["fake_a", "context_text"], ctx(tmp_path, []))
    assert ran == ["fake_a", "fake_b"]


def test_ordered_keys_sorts_within_groups_and_keeps_first() -> None:
    assert tools.ordered_keys([["b", "a"], ["c", "a"]]) == ["a", "b", "c"]
    # codepoint order within a group, as in the canonical sources (upper case before lower)
    assert tools.ordered_keys([["ctrl", "TP53", "AAK1"], ["Zeta", "DMSO"]]) == [
        "AAK1",
        "TP53",
        "ctrl",
        "DMSO",
        "Zeta",
    ]


def test_perturbation_text_orders_extends_and_replaces(
    tmp_path: Path, fakes: list[list[str]]
) -> None:
    metadata = tmp_path / "drugs.csv"
    pd.DataFrame({"drug": ["Zeta"], "pubchem_cid": [7]}).to_csv(metadata, index=False)
    options = {"gene_info": tmp_path / "gene_info.gz", "drug_metadata": metadata}
    first = TOOLS["perturbation_text"].run(ctx(tmp_path / "a", [GENE_SET, DRUG_SET], **options))
    source = read_source(first)
    keys = ["AAK1", "TP53", "ctrl", "DMSO_TF_0.0uM", "Zeta_5.0uM"]
    assert source.meta.keys == keys and source.meta.index == "pert"
    assert list(read_descriptions(first)) == keys
    assert source.meta.dim == openai_embed.DIM and source.meta.dtype == "float32"
    assert read_descriptions(first)["Zeta_5.0uM"] == "drug Zeta_5.0uM"
    assert fakes == [[f"gene {k}" for k in keys[:3]] + [f"drug {k}" for k in keys[3:]]]
    provenance = source.meta.provenance
    expected = {"ncbi_gene_info", "ncbi_esummary", "drug_metadata", "pubchem"}
    assert set(provenance["inputs"]) == expected
    assert provenance["inputs"]["ncbi_esummary"]["url"] == str(
        tmp_path / "a" / "cache" / "perturbation_text"
    )
    assert provenance["model"] == openai_embed.MODEL and "revision" in provenance

    # A genetic dataset after the first one takes the gene-query path: no control key, and
    # symbols that are perturbations of the first genetic dataset copy its text.
    extra = dataset("x", "gene", ["TP53", "BRCA1", "ctrl"], ["TP53"], ["rpe1"])
    prior_root = tmp_path / "a" / "out"
    second = TOOLS["perturbation_text"].run(
        ctx(tmp_path / "b", [GENE_SET, DRUG_SET, extra], prior_root, **options)
    )
    grown = read_source(second)
    assert grown.meta.keys == [*keys, "BRCA1"]
    assert read_descriptions(second)["BRCA1"] == "query BRCA1"
    assert fakes[1:] == [["query BRCA1"]]
    assert np.asarray(grown.embeddings[:5]).tobytes() == np.asarray(source.embeddings).tobytes()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(genes, "describe_genetic_perts", lambda k, *_a: {x: f"new {x}" for x in k})
    try:
        with pytest.raises(DescriptionConflictError):
            TOOLS["perturbation_text"].run(ctx(tmp_path / "c", [GENE_SET], prior_root, **options))
        replaced = TOOLS["perturbation_text"].run(
            ctx(tmp_path / "d", [GENE_SET], prior_root, on_conflict="replace", **options)
        )
    finally:
        monkeypatch.undo()
    texts = read_descriptions(replaced)
    assert list(texts) == keys and texts["TP53"] == "new TP53"
    assert texts["Zeta_5.0uM"] == "drug Zeta_5.0uM"
    assert fakes[2:] == [["new AAK1", "new TP53", "new ctrl"]]
    assert read_source(replaced).tokens("TP53")[0, 0] == len("new TP53")


def test_gene_query_path_uses_prior_gene_text(
    tmp_path: Path, fakes: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"gene_info": tmp_path / "gene_info.gz"}
    base = tmp_path / "a"
    TOOLS["perturbation_text"].run(ctx(base, [GENE_SET], **options))
    TOOLS["gene_text"].run(ctx(base, [GENE_SET], **options))
    seen: list[tuple[Any, ...]] = []
    query = genes.describe_gene_queries

    def spy(vocab: Any, prior: Any, pert_output: Any, pert_keys: Any, *rest: Any) -> Any:
        seen.append((list(vocab), prior, dict(pert_output), list(pert_keys)))
        return query(vocab, prior, pert_output, pert_keys, *rest)

    monkeypatch.setattr(genes, "describe_gene_queries", spy)
    later = dataset("x", "gene", ["GAPDH", "NEW1", "ctrl"], ["GAPDH"], ["rpe1"])
    out = TOOLS["perturbation_text"].run(
        ctx(tmp_path / "b", [GENE_SET, later], base / "out", **options)
    )
    texts = read_descriptions(out)
    assert list(texts) == ["AAK1", "TP53", "ctrl", "GAPDH", "NEW1"]
    assert texts["GAPDH"] == "query GAPDH" and texts["NEW1"] == "query NEW1"
    [(vocab, prior, pert_output, pert_keys)] = seen
    assert vocab == ["GAPDH", "NEW1"] and pert_keys == ["AAK1", "TP53"]
    assert prior == read_descriptions(base / "out" / "gene_text")
    assert pert_output["TP53"] == "gene TP53"


def test_drug_dataset_needs_drug_metadata(tmp_path: Path, fakes: list[list[str]]) -> None:
    with pytest.raises(ValueError, match=r"options\.drug_metadata"):
        TOOLS["perturbation_text"].run(ctx(tmp_path, [DRUG_SET]))
    assert fakes == []


def test_gene_text_copies_from_perturbation_text(tmp_path: Path, fakes: list[list[str]]) -> None:
    run_ctx = ctx(tmp_path, [GENE_SET, DRUG_SET], gene_info=tmp_path / "gene_info.gz")
    with pytest.raises(FileNotFoundError, match="perturbation_text"):
        TOOLS["gene_text"].run(run_ctx)
    (tmp_path / "pt").mkdir()
    TOOLS["perturbation_text"].run(ctx(tmp_path / "pt", [GENE_SET], gene_info=tmp_path / "g.gz"))
    run_ctx.options.pert_output = tmp_path / "pt" / "out" / "perturbation_text"
    out = TOOLS["gene_text"].run(run_ctx)
    texts = read_descriptions(out)
    # one codepoint-sorted gene axis over all datasets, as in the canonical gene_text
    assert list(texts) == ["ACTB", "GAPDH", "TP53"]
    assert texts == {"GAPDH": "query GAPDH", "TP53": "gene TP53", "ACTB": "query ACTB"}
    source = read_source(out)
    assert source.meta.index == "gene" and source.meta.keys == list(texts)
    assert source.meta.provenance["inputs"]["ncbi_esummary"]["url"] == str(
        tmp_path / "cache" / "gene_text"
    )


def test_context_text_orders_contexts_by_dataset(tmp_path: Path, fakes: list[list[str]]) -> None:
    out = TOOLS["context_text"].run(ctx(tmp_path, [DRUG_SET, GENE_SET]))
    texts = read_descriptions(out)
    assert list(texts) == ["hepg2", "k562"]
    # the Cellosaurus client gets the per-tool cache subdir
    assert texts == {"hepg2": "ctx hepg2 context_text", "k562": "ctx k562 context_text"}
    source = read_source(out)
    assert source.meta.index == "context"
    assert source.meta.provenance["inputs"]["cellosaurus"]["release"] == "56.0"


def test_keep_prior_keeps_old_text_and_rows_without_embedding(
    tmp_path: Path, fakes: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"gene_info": tmp_path / "gene_info.gz"}
    first = TOOLS["perturbation_text"].run(ctx(tmp_path / "a", [GENE_SET], **options))
    monkeypatch.setattr(genes, "describe_genetic_perts", lambda k, *_a: {x: f"new {x}" for x in k})
    kept = TOOLS["perturbation_text"].run(
        ctx(tmp_path / "b", [GENE_SET], tmp_path / "a" / "out", on_conflict="keep-prior", **options)
    )
    assert read_descriptions(kept) == read_descriptions(first)
    assert np.asarray(read_source(kept).embeddings).tobytes() == np.asarray(
        read_source(first).embeddings
    ).tobytes()
    assert fakes[1:] == []


def _genetic_dir(tmp_path: Path, pert_ensembl: dict[str, str]) -> PreprocessedDir:
    """A real one-context genetic preprocessed dir with perts GA and ctrl over genes G1, G2."""
    shape = (2, 2)
    arrays = {
        pp.FOLD_CHANGES: np.ones(shape, np.float32),
        pp.FDR: np.full(shape, 0.5, np.float32),
        pp.TESTED: np.ones(shape, bool),
        pp.LFC_TRUE: np.zeros(shape),
        pp.DELTA_P: np.zeros(shape, np.float32),
        pp.CTRL_MEANS: np.ones((1, 2), np.float32),
        pp.CTX_IDS: np.zeros(2, np.int32),
        pp.PERT_IDS: np.arange(2, dtype=np.int32),
    }
    meta = pp.PreprocessedMeta(
        format_version=pp.FORMAT_VERSION,
        dataset="g",
        genes=["G1", "G2"],
        context_to_id={"k562": 0},
        pert_to_id={"GA": 0, "ctrl": 1},
        pert_kind="gene",
        control_label="ctrl",
        num_rows=2,
        num_genes=2,
        num_contexts=1,
        num_perts=2,
        controls_only=False,
        tool_version="0.0.0",
        array_sha256={},
        pert_ensembl=pert_ensembl,
    )
    return PreprocessedDir.open(pp.write_preprocessed(tmp_path / "pre" / "g", meta, arrays))


def test_perturbation_text_passes_the_preprocessed_ensembl_ids(
    tmp_path: Path, fakes: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[dict[str, str]] = []

    def fake_describe(
        keys: list[str], index: Any, esummary: Any, ensembl_ids: dict[str, str], control: str
    ) -> dict[str, str]:
        recorded.append(dict(ensembl_ids))
        return {k: f"text {k}" for k in keys}

    monkeypatch.setattr(genes, "describe_genetic_perts", fake_describe)
    data = _genetic_dir(tmp_path, pert_ensembl={"GA": "ENSG00000000001"})
    out = TOOLS["perturbation_text"].run(ctx(tmp_path, [data], gene_info=tmp_path / "g.gz"))
    assert recorded == [{"GA": "ENSG00000000001"}]
    assert read_descriptions(out) == {"GA": "text GA", "ctrl": "text ctrl"}
    assert "h5ad" not in read_source(out).meta.provenance["params"]


def test_extend_in_place_keeps_prior_rows_and_replaces_the_dir(
    tmp_path: Path, fakes: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "out"
    first = TOOLS["context_text"].run(ctx(tmp_path, [GENE_SET]))
    prior_rows = np.asarray(read_source(first).embeddings).tobytes()
    prior_texts = read_descriptions(first)
    embed = openai_embed.embed_texts

    def only_new(texts: list[str], **kw: Any) -> np.ndarray:
        assert not set(texts) & set(prior_texts.values()), "a prior key was re-embedded"
        return embed(texts, **kw)

    monkeypatch.setattr(openai_embed, "embed_texts", only_new)
    grown = dataset("x", "gene", ["TP53"], ["TP53"], ["rpe1"])
    run_ctx = ctx(tmp_path, [GENE_SET, grown], root)
    run_ctx.overwrite = True
    out = TOOLS["context_text"].run(run_ctx)
    assert out == first == root / "context_text"
    source = read_source(out)
    assert source.meta.keys == ["k562", "rpe1"]
    assert np.asarray(source.embeddings[:1]).tobytes() == prior_rows
    assert read_descriptions(out)["k562"] == prior_texts["k562"]
    assert fakes[1:] == [["ctx rpe1 context_text"]]
    assert sorted(p.name for p in root.iterdir()) == ["context_text"]


def test_context_text_records_each_context_map_digest(
    tmp_path: Path, fakes: list[list[str]]
) -> None:
    out = TOOLS["context_text"].run(ctx(tmp_path, [GENE_SET]))
    params = read_source(out).meta.provenance["params"]
    expected = sha256_file(tmp_path / "pre" / "g" / "contexts.yaml")
    assert params == {"cellosaurus_release": "56.0", "contexts_sha256": {"g": expected}}
