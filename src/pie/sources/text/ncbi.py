"""NCBI Gene FTP parser and gene-text renderer for the ``ncbi_text`` source.

The genes table has one row per human protein-coding gene, in ``gene_info`` order, with its GO
terms (``gene2go``), MedGen-named phenotype associations (``mim2gene_medgen`` and the MedGen
``NAMES`` table) and gene-group memberships (``gene_group``). ``render_gene_text`` turns one row
into the sectioned text that the token-level encoder embeds.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pie.sources.common import base_provenance, download_file, environment_record, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source
from pie.sources.embed import qwen
from pie.utils import require_env

if TYPE_CHECKING:
    import requests

    from pie.sources.registry import RunContext

NCBI_FTP_FILES: tuple[str, ...] = (
    "gene_info",
    "gene2go",
    "mim2gene_medgen",
    "medgen_names",
    "gene_group",
)
FTP_URLS: dict[str, str] = {
    "gene_info": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz",
    "gene2go": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz",
    "mim2gene_medgen": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/mim2gene_medgen",
    "medgen_names": "https://ftp.ncbi.nlm.nih.gov/pub/medgen/csv/NAMES.csv.gz",
    "gene_group": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_group.gz",
}
TAX_ID = "9606"
GENE_TYPES: tuple[str, ...] = ("protein-coding",)
SECTIONS: tuple[str, ...] = (
    "gene_info",
    "summary",
    "go_biological_process",
    "go_molecular_function",
    "go_cellular_component",
    "disease_associations",
    "gene_groups",
)
EXCLUDED_SECTIONS: tuple[str, ...] = ("gene_info", "summary")
TABLE_COLUMNS: tuple[str, ...] = (
    "gene_id",
    "tax_id",
    "symbol",
    "name",
    "gene_type",
    "chromosome",
    "map_location",
    "synonyms",
    "db_xrefs",
    "go_biological_process",
    "go_molecular_function",
    "go_cellular_component",
    "disease_associations",
    "gene_groups",
)
READ_CHUNK_ROWS = 2_000_000
# (table column, gene2go Category, header title, one-line description)
_GO_SECTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "go_biological_process",
        "Process",
        "BIOLOGICAL PROCESS",
        "Gene Ontology biological process annotations, sorted by evidence count.",
    ),
    (
        "go_molecular_function",
        "Function",
        "MOLECULAR FUNCTION",
        "Gene Ontology molecular function annotations, sorted by evidence count.",
    ),
    (
        "go_cellular_component",
        "Component",
        "CELLULAR COMPONENT",
        "Gene Ontology cellular component annotations, sorted by evidence count.",
    ),
)
_GENE_INFO_DESCRIPTION = (
    "Basic gene identifiers, genomic location, synonyms, and database cross-references."
)
_DISEASE_DESCRIPTION = (
    "Known disease associations with confidence scores and OMIM "
    "(Online Mendelian Inheritance in Man) identifiers."
)
_GROUPS_DESCRIPTION = "Gene family and group memberships by relationship type."


# --- fetch and read ---------------------------------------------------------------------------


def ftp_file_name(name: str) -> str:
    """Upstream file name of one NCBI FTP input (the last URL segment)."""
    return FTP_URLS[name].rsplit("/", 1)[1]


def fetch_ncbi_ftp(
    cache_dir: Path,
    session: requests.Session | None = None,
    names: Sequence[str] = NCBI_FTP_FILES,
    offline: bool = False,
) -> dict[str, Path]:
    """Download the named NCBI FTP files into `cache_dir` (cached files are reused)."""
    unknown = sorted(set(names) - set(FTP_URLS))
    if unknown:
        raise KeyError(f"unknown NCBI FTP file(s) {unknown}; expected some of {NCBI_FTP_FILES}")
    return {
        name: download_file(
            FTP_URLS[name], Path(cache_dir) / ftp_file_name(name), session=session, offline=offline
        )
        for name in names
    }


def _strip_header(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = pd.Index([str(column).lstrip("#").strip() for column in frame.columns])
    return frame


def _compression(path: Path) -> str | None:
    return "gzip" if Path(path).suffix == ".gz" else None


def read_ncbi_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    """A whole NCBI table, every column as str, leading '#' stripped from the header."""
    frame = pd.read_csv(path, sep=sep, compression=_compression(path), dtype=str)
    return _strip_header(frame)


def read_human_rows(
    path: Path, tax_id: str = TAX_ID, chunk_rows: int = READ_CHUNK_ROWS
) -> pd.DataFrame:
    """The rows of a large all-organism NCBI table whose tax_id is `tax_id`, read in chunks."""
    parts = []
    with pd.read_csv(
        path, sep="\t", compression=_compression(path), dtype=str, chunksize=chunk_rows
    ) as reader:
        for chunk in reader:
            frame = _strip_header(chunk)
            parts.append(frame[frame["tax_id"] == tax_id])
    return pd.concat(parts, ignore_index=True)


def protein_coding_gene_ids(gene_info: pd.DataFrame) -> list[str]:
    """GeneIDs of the human protein-coding genes, in gene_info row order."""
    rows = gene_info[
        (gene_info["tax_id"] == TAX_ID) & gene_info["type_of_gene"].isin(list(GENE_TYPES))
    ]
    return rows["GeneID"].tolist()


def protein_coding_symbols(gene_info_gz: Path) -> list[str]:
    """Symbols of the human protein-coding genes of a gene_info file, in row order."""
    info = read_human_rows(gene_info_gz)
    rows = info[info["type_of_gene"].isin(list(GENE_TYPES))]
    return [str(symbol) for symbol in rows["Symbol"].dropna()]


# --- parse ------------------------------------------------------------------------------------


def _parse_gene_info(frame: pd.DataFrame, gene_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = frame[(frame["tax_id"] == TAX_ID) & (frame["GeneID"].isin(gene_ids))]
    out: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        synonyms = row.get("Synonyms", "-")
        xrefs = row.get("dbXrefs", "-")
        out[row["GeneID"]] = {
            "gene_id": int(row["GeneID"]),
            "tax_id": int(row["tax_id"]),
            "symbol": row["Symbol"],
            "name": row.get("description", ""),
            "gene_type": row.get("type_of_gene", ""),
            "chromosome": row.get("chromosome", ""),
            "map_location": row.get("map_location", ""),
            "synonyms": synonyms.split("|") if synonyms != "-" else [],
            "db_xrefs": xrefs.split("|") if xrefs != "-" else [],
        }
    return out


def _parse_gene2go(
    frame: pd.DataFrame, gene_ids: set[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = frame[(frame["tax_id"] == TAX_ID) & (frame["GeneID"].isin(gene_ids))]
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for gene_id, gene_rows in rows.groupby("GeneID"):
        categories: dict[str, list[dict[str, Any]]] = {}
        for category, category_rows in gene_rows.groupby("Category"):
            terms = [
                {
                    "go_id": go_id,
                    "term": go_rows.iloc[0]["GO_term"],
                    "evidence_count": len(go_rows),
                }
                for go_id, go_rows in category_rows.groupby("GO_ID")
            ]
            terms.sort(key=itemgetter("evidence_count"), reverse=True)
            categories[str(category)] = terms
        out[str(gene_id)] = categories
    return out


def _comment_confidence(comment: str) -> int:
    if not comment or comment == "-":
        return 4
    lowered = comment.lower()
    if "question" in lowered:
        return 1
    if "nondisease" in lowered:
        return 2
    if any(word in lowered for word in ("susceptibility", "somatic", "modifier")):
        return 3
    return 1


def _disease_rank(disease: Mapping[str, Any]) -> tuple[Any, Any]:
    return (disease.get("comment_confidence", 0), disease.get("n_sources", 0))


def _parse_mim2gene(
    mim: pd.DataFrame, names: pd.DataFrame, gene_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    phenotypes = mim[(mim["type"] == "phenotype") & (mim["GeneID"].isin(gene_ids))]
    cui_to_name: dict[str, str] = {}
    for cui, name in zip(names["CUI"], names["name"], strict=True):
        if cui and name:
            cui_to_name[cui] = name
    out: dict[str, list[dict[str, Any]]] = {}
    for gene_id, gene_rows in phenotypes.groupby("GeneID"):
        diseases: list[dict[str, Any]] = []
        for _, row in gene_rows.iterrows():
            comment = row.get("Comment", "-")
            source = row.get("Source", "")
            n_sources = (
                len([part for part in source.split(";") if part.strip()])
                if source and source != "-"
                else 0
            )
            cui = row.get("MedGenCUI", "")
            diseases.append(
                {
                    "disease_name": cui_to_name.get(cui, "") if cui and cui != "-" else "",
                    "mim_number": row.get("MIM number", ""),
                    "comment_confidence": _comment_confidence(comment),
                    "n_sources": n_sources,
                }
            )
        diseases.sort(key=_disease_rank, reverse=True)
        out[str(gene_id)] = diseases
    return out


def _parse_gene_groups(
    frame: pd.DataFrame, gene_ids: set[str], id_to_symbol: Mapping[str, str]
) -> dict[str, dict[str, list[str]]]:
    rows = frame[frame["GeneID"].isin(gene_ids) | frame["Other_GeneID"].isin(gene_ids)]
    groups: dict[str, dict[str, set[str]]] = {}
    for _, row in rows.iterrows():
        gene_id = row["GeneID"]
        other_id = row["Other_GeneID"]
        relation = row["relationship"]
        targets = [g for g in (gene_id, other_id) if g in gene_ids]
        for target in targets:
            other = other_id if target == gene_id else gene_id
            groups.setdefault(target, {}).setdefault(relation, set()).add(
                id_to_symbol.get(other, other)
            )
    return {
        gene: {relation: sorted(members) for relation, members in relations.items()}
        for gene, relations in groups.items()
    }


def _gene_record(
    gene_id: str,
    info: Mapping[str, Mapping[str, Any]],
    go: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    diseases: Mapping[str, list[dict[str, Any]]],
    groups: Mapping[str, dict[str, list[str]]],
) -> dict[str, Any]:
    meta = info.get(gene_id, {})
    terms = go.get(gene_id, {})
    return {
        "gene_id": meta.get("gene_id", int(gene_id)),
        "tax_id": meta.get("tax_id", 0),
        "symbol": meta.get("symbol", ""),
        "name": meta.get("name", ""),
        "gene_type": meta.get("gene_type", ""),
        "chromosome": meta.get("chromosome", ""),
        "map_location": meta.get("map_location", ""),
        "synonyms": meta.get("synonyms", []),
        "db_xrefs": meta.get("db_xrefs", []),
        "go_biological_process": terms.get("Process", []),
        "go_molecular_function": terms.get("Function", []),
        "go_cellular_component": terms.get("Component", []),
        "disease_associations": diseases.get(gene_id, []),
        "gene_groups": groups.get(gene_id, {}),
    }


def build_genes_table(files: Mapping[str, Path]) -> pd.DataFrame:
    """One row per human protein-coding gene (gene_info order), columns TABLE_COLUMNS."""
    gene_info = read_human_rows(files["gene_info"])
    gene_ids = protein_coding_gene_ids(gene_info)
    id_set = set(gene_ids)
    id_to_symbol = dict(zip(gene_info["GeneID"], gene_info["Symbol"], strict=True))
    info = _parse_gene_info(gene_info, id_set)
    go = _parse_gene2go(read_human_rows(files["gene2go"]), id_set)
    diseases = _parse_mim2gene(
        read_ncbi_table(files["mim2gene_medgen"]),
        read_ncbi_table(files["medgen_names"], sep=","),
        id_set,
    )
    groups = _parse_gene_groups(read_ncbi_table(files["gene_group"]), id_set, id_to_symbol)
    records = [_gene_record(g, info, go, diseases, groups) for g in gene_ids]
    return pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS))


# --- render -----------------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    """A list/dict column value, decoding a JSON string; anything unreadable becomes []."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return []
    if isinstance(value, list | dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _fmt_list(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "None"


def _evidence_count(term: Mapping[str, Any]) -> Any:
    return term.get("evidence_count", 0)


def _go_line(term: Mapping[str, Any]) -> str:
    return (
        f"- {term.get('go_id', 'N/A')}: {term.get('term', 'N/A')} "
        f"(evidence: {term.get('evidence_count', 0)})"
    )


def _disease_line(disease: Mapping[str, Any]) -> str:
    confidence = disease.get("comment_confidence", "N/A")
    return (
        f"- {disease.get('disease_name', 'N/A')} (confidence: {confidence}, "
        f"MIM: {disease.get('mim_number', 'N/A')})"
    )


def _gene_groups(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, dict):
        return raw
    groups: dict[str, list[str]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                for relation, members in item.items():
                    groups.setdefault(relation, []).extend(
                        members if isinstance(members, list) else [members]
                    )
    return groups


def render_gene_text(
    row: Mapping[str, Any] | pd.Series,
    exclude: Sequence[str] = EXCLUDED_SECTIONS,
    keep_symbol_line: bool = True,
) -> str:
    """Render one genes-table row as sectioned text, leaving out the sections in `exclude`.

    With ``gene_info`` excluded and `keep_symbol_line`, a single ``Gene: <symbol>`` line leads.
    GO terms are listed by evidence count and diseases by (confidence, n_sources), both
    descending; an empty section prints ``None``.
    """
    excluded = set(exclude)
    unknown = sorted(excluded - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown section(s) in exclude: {unknown}")
    parts: list[str] = []
    if "gene_info" not in excluded:
        parts += [
            "=== GENE INFO ===",
            _GENE_INFO_DESCRIPTION,
            f"Symbol: {row['symbol']}",
            f"Full Name: {row['name']}",
            f"Gene Type: {row['gene_type']}",
            f"Chromosome: {row['chromosome']}",
            f"Map Location: {row['map_location']}",
            f"Synonyms: {_fmt_list(_json_value(row.get('synonyms')))}",
            f"Database Cross-References: {_fmt_list(_json_value(row.get('db_xrefs')))}",
        ]
    elif keep_symbol_line:
        parts.append(f"Gene: {row['symbol']}")
    if "summary" not in excluded:
        summary = row.get("summary", "") or ""
        parts += ["", "=== SUMMARY ===", "Free-text summary of gene function from NCBI."]
        parts.append(summary or "No summary available.")
    for column, _category, title, description in _GO_SECTIONS:
        if column in excluded:
            continue
        terms = _json_value(row.get(column))
        if isinstance(terms, list) and terms:
            terms = sorted(terms, key=_evidence_count, reverse=True)
        parts += ["", f"=== GO {title} ===", description]
        parts += [_go_line(term) for term in terms] if terms else ["None"]
    if "disease_associations" not in excluded:
        diseases = _json_value(row.get("disease_associations"))
        if isinstance(diseases, list) and diseases:
            diseases = sorted(diseases, key=_disease_rank, reverse=True)
        parts += ["", "=== DISEASE ASSOCIATIONS ===", _DISEASE_DESCRIPTION]
        parts += [_disease_line(d) for d in diseases] if diseases else ["None"]
    if "gene_groups" not in excluded:
        groups = _gene_groups(_json_value(row.get("gene_groups")))
        parts += ["", "=== GENE GROUPS ===", _GROUPS_DESCRIPTION]
        if groups:
            parts += [f"- {relation}: {', '.join(members)}" for relation, members in groups.items()]
        else:
            parts.append("None")
    # Excluding gene_info leaves the next section's blank separator first: strip it.
    return "\n".join(parts).strip()


def describe_ncbi_genes(
    table: pd.DataFrame,
    exclude: Sequence[str] = EXCLUDED_SECTIONS,
    keep_symbol_line: bool = True,
) -> dict[str, str]:
    """{symbol: rendered text} in table order; a repeated symbol is an error."""
    texts: dict[str, str] = {}
    for _, row in table.iterrows():
        symbol = str(row["symbol"])
        if symbol in texts:
            raise ValueError(f"duplicate gene symbol {symbol!r} in the genes table")
        texts[symbol] = render_gene_text(row, exclude, keep_symbol_line)
    return texts


# --- tool -------------------------------------------------------------------------------------


def run_ncbi_text(ctx: RunContext) -> Path:
    """ncbi_text: NCBI FTP files -> genes table -> rendered texts -> token-level embeddings."""
    pinned = ctx.options.gene_info
    names = tuple(n for n in NCBI_FTP_FILES if pinned is None or n != "gene_info")
    fetched = fetch_ncbi_ftp(ctx.cache_dir / "ncbi_text", names=names, offline=ctx.options.offline)
    files = {n: Path(pinned) if n == "gene_info" and pinned else fetched[n] for n in NCBI_FTP_FILES}
    texts = describe_ncbi_genes(build_genes_table(files))
    keys = list(texts)
    ordered = [texts[key] for key in keys]
    cache_root = Path(require_env("PIE_CACHE_DIR")["PIE_CACHE_DIR"])
    work_dir = cache_root / "embed" / "ncbi_text" / qwen.cache_key(ordered)
    tokens, offsets = qwen.embed_tokens(ordered, device=ctx.options.device, work_dir=work_dir)
    params = {
        "tax_id": TAX_ID,
        "gene_types": list(GENE_TYPES),
        "exclude_sections": list(EXCLUDED_SECTIONS),
        "keep_symbol_line": True,
        **qwen.PARAMS,
    }
    provenance = base_provenance(
        {name: input_record(path, FTP_URLS[name]) for name, path in files.items()},
        params,
        model=qwen.MODEL,
        revision=qwen.REVISION,
        environment=environment_record(ctx.options.device),
    )
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="ncbi_text",
        layout="token",
        index="pert",
        keys=keys,
        dim=int(tokens.shape[1]),
        dtype="float16",
        provenance=provenance,
    )
    out = write_source(
        ctx.out_root / "ncbi_text",
        meta,
        tokens,
        offsets=offsets,
        descriptions=texts,
        overwrite=ctx.overwrite,
    )
    del tokens
    shutil.rmtree(work_dir, ignore_errors=True)
    return out
