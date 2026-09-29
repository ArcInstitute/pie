"""Gene descriptions from NCBI Gene (gene_info + E-utilities), rendered to text."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import requests

from pie.sources.text import _http
from pie.sources.text.contexts import _clean as clean
from pie.sources.text.contexts import _normalize_list as normalize_list
from pie.sources.text.contexts import _normalize_value, sort_key
from pie.utils import atomic_write_text, sha256_file

CONTROL_TEXT = "Non-targeting control"

SOURCE_FIELDS = (
    "hgnc_gene_symbol",
    "full_name",
    "synonyms",
    "gene_type",
    "organism",
    "lineage",
    "map_location",
    "alt_designations",
    "ncbi_gene_id",
    "ensembl_gene_id",
    "gene_summary",
)
LIST_FIELDS = {"synonyms", "alt_designations"}
EMBEDDING_FIELDS = (
    ("hgnc_gene_symbol", "HGNC Gene Symbol"),
    ("full_name", "Full Name"),
    ("synonyms", "Synonyms"),
    ("gene_type", "Gene Type"),
    ("organism", "Organism"),
    ("lineage", "Lineage"),
    ("map_location", "Map Location"),
    ("alt_designations", "Alt Designations"),
    ("gene_summary", "Gene Summary"),
)


def _normalize_gene_type(raw: object) -> str:
    value = clean(raw)
    return "pseudogene" if value == "pseudo" else value


def _parse_pipe_list(field: object) -> list[str]:
    value = clean(field)
    return normalize_list(value.split("|")) if value else []


def _parse_ensembl_id(dbxrefs: object) -> str:
    for token in _parse_pipe_list(dbxrefs):
        if token.startswith("Ensembl:"):
            return token.removeprefix("Ensembl:")
    return ""


HUMAN_TAX_ID = "9606"

_GI = {
    "tax_id": 0,
    "geneid": 1,
    "symbol": 2,
    "synonyms": 4,
    "dbxrefs": 5,
    "map_location": 7,
    "description": 8,
    "type_of_gene": 9,
    "nomenclature_symbol": 10,
    "nomenclature_name": 11,
    "other_designations": 13,
    "modification_date": 14,
}


def _load_gene_info(path: str | os.PathLike[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with gzip.open(path, "rt") as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) <= _GI["other_designations"]:
                continue
            if parts[_GI["tax_id"]].strip() != HUMAN_TAX_ID:
                continue
            gene_id = parts[_GI["geneid"]]
            rows[gene_id] = {
                "tax_id": clean(parts[_GI["tax_id"]]),
                "geneid": gene_id,
                "symbol": clean(parts[_GI["symbol"]]),
                "synonyms": _parse_pipe_list(parts[_GI["synonyms"]]),
                "dbxrefs": clean(parts[_GI["dbxrefs"]]),
                "map_location": clean(parts[_GI["map_location"]]),
                "description": clean(parts[_GI["description"]]),
                "type_of_gene": clean(parts[_GI["type_of_gene"]]),
                "nomenclature_symbol": clean(parts[_GI["nomenclature_symbol"]]),
                "nomenclature_name": clean(parts[_GI["nomenclature_name"]]),
                "other_designations": _parse_pipe_list(parts[_GI["other_designations"]]),
                "modification_date": clean(parts[_GI["modification_date"]]),
            }
    return rows


def _build_indexes(
    rows_by_id: dict[str, dict],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    symbol_index: dict[str, set[str]] = {}
    synonym_index: dict[str, set[str]] = {}
    for gene_id, row in rows_by_id.items():
        for symbol in {row["symbol"], row["nomenclature_symbol"]}:
            if symbol:
                symbol_index.setdefault(symbol.casefold(), set()).add(gene_id)
        for synonym in row["synonyms"]:
            if synonym:
                synonym_index.setdefault(synonym.casefold(), set()).add(gene_id)
    return symbol_index, synonym_index


def _build_ensembl_index(rows_by_id: dict[str, dict]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for gene_id, row in rows_by_id.items():
        ensembl_id = _parse_ensembl_id(row["dbxrefs"])
        if ensembl_id:
            index.setdefault(ensembl_id, set()).add(gene_id)
    return index


def _resolve(
    runtime_key: str,
    ensembl_id: str,
    symbol_index: dict[str, set[str]],
    synonym_index: dict[str, set[str]],
    ensembl_index: dict[str, set[str]],
    rows_by_id: dict[str, dict],
) -> tuple[str | None, str]:
    """Propose a GeneID and registry-compatible resolution method, or a status."""
    if ensembl_id.startswith("ENSG") and ensembl_id in ensembl_index:
        candidates = ensembl_index[ensembl_id]
        if len(candidates) == 1:
            return next(iter(candidates)), "ensembl_id"
        folded_key = runtime_key.casefold()
        matches = [
            gene_id
            for gene_id in candidates
            if rows_by_id[gene_id]["symbol"].casefold() == folded_key
            or rows_by_id[gene_id]["nomenclature_symbol"].casefold() == folded_key
        ]
        if len(matches) == 1:
            return matches[0], "ensembl_id"

    folded_key = runtime_key.casefold()
    for index, method in (
        (symbol_index, "official_symbol"),
        (synonym_index, "synonym"),
    ):
        hits = index.get(folded_key)
        if hits:
            if len(hits) == 1:
                return next(iter(hits)), method
            return None, "ambiguous"
    return None, "unresolved"


def _extract_fields(row: dict) -> dict:
    """Extract the eight gene_info-backed fields in schema order."""
    return {
        "hgnc_gene_symbol": row["nomenclature_symbol"] or row["symbol"],
        "full_name": row["nomenclature_name"] or row["description"],
        "synonyms": normalize_list(row["synonyms"]),
        "gene_type": _normalize_gene_type(row["type_of_gene"]),
        "map_location": clean(row["map_location"]),
        "alt_designations": normalize_list(row["other_designations"]),
        "ncbi_gene_id": row["geneid"],
        "ensembl_gene_id": _parse_ensembl_id(row["dbxrefs"]),
    }


def _render_embedding_text(runtime_key: str, entry: dict) -> str:
    segments = [f"Gene Name: {_normalize_value(runtime_key)}"]
    for field, label in EMBEDDING_FIELDS:
        value = entry[field]
        if isinstance(value, list):
            value = ", ".join(normalize_list(_normalize_value(item) for item in value))
        else:
            value = _normalize_value(value)
        if value:
            segments.append(f"{label}: {value}")
    return " ;\n".join(segments)


def _build_entry(runtime_key: str, fields: dict) -> dict:
    entry = {
        field: normalize_list(fields.get(field, []))
        if field in LIST_FIELDS
        else clean(fields.get(field, ""))
        for field in SOURCE_FIELDS
    }
    entry["embedding_text"] = _render_embedding_text(runtime_key, entry)
    return entry


class GeneInfoIndex:
    """Human rows (tax_id 9606) of an NCBI gene_info.gz (Homo_sapiens or all-species) with symbol,
    synonym and Ensembl indexes."""

    def __init__(self, path: Path, rows: dict[str, dict]) -> None:
        self.path = Path(path)
        self.sha256 = sha256_file(self.path)
        self.rows = rows
        self._symbols, self._synonyms = _build_indexes(rows)
        self._ensembl = _build_ensembl_index(rows)

    @classmethod
    def load(cls, gene_info_gz: Path) -> GeneInfoIndex:
        return cls(gene_info_gz, _load_gene_info(gene_info_gz))

    def resolve_with_method(self, key: str, ensembl_id: str = "") -> tuple[str | None, str]:
        """(GeneID, ensembl_id|official_symbol|synonym) or (None, ambiguous|unresolved)."""
        return _resolve(
            key, ensembl_id or "", self._symbols, self._synonyms, self._ensembl, self.rows
        )

    def resolve(self, key: str, ensembl_id: str | None = None) -> dict[str, object] | None:
        gene_id, _ = self.resolve_with_method(key, ensembl_id or "")
        return None if gene_id is None else self.rows[gene_id]

    def fields(self, gene_id: str) -> dict[str, object]:
        return _extract_fields(self.rows[gene_id])


def render_gene(key: str, record: Mapping[str, object] | None) -> str:
    """Canonical gene text; `record` = fields + organism/lineage/gene_summary, None = minimal."""
    return _build_entry(key, dict(record or {}))["embedding_text"]


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)
GENE_INFO_NAME = "Homo_sapiens.gene_info.gz"
EUTILS_TOOL = "pie-sources"
HUMAN = "Homo sapiens"
NO_KEY_PACING_SECONDS = 0.34
MAX_ESUMMARY_BATCH_SIZE = 200
DIGEST_PREFIX_LENGTH = 12

_ENSEMBL_BARE = re.compile(r"ENSG\d+")
_ENSEMBL_SUFFIX = re.compile(r"(?P<stem>.+)_(?P<ensembl>ENSG\d+)")


def _gene_info_path(cache_dir: Path, digest: str) -> Path:
    """Content-addressed location of one gene_info snapshot (a refresh never overwrites)."""
    return Path(cache_dir) / "gene_info" / digest[:DIGEST_PREFIX_LENGTH] / GENE_INFO_NAME


def _cached_gene_info(cache_dir: Path, digest: str | None) -> Path | None:
    if digest:
        pinned = _gene_info_path(cache_dir, digest)
        return pinned if pinned.is_file() else None
    found = sorted(Path(cache_dir).glob(f"gene_info/*/{GENE_INFO_NAME}"))
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} cached gene_info snapshots under {Path(cache_dir) / 'gene_info'}; "
            "pass --gene-info to choose one"
        )
    return found[0] if found else None


def fetch_gene_info(
    cache_dir: Path,
    session: requests.Session | None = None,
    *,
    offline: bool = False,
    digest: str | None = None,
) -> Path:
    """Homo_sapiens.gene_info.gz cached at <cache_dir>/gene_info/<sha256[:12]>/, downloaded once.

    With `digest`, only that snapshot is accepted and a download must hash to it. Without it, a
    lone cached snapshot is used and several are refused rather than silently picking one.
    """
    cache_dir = Path(cache_dir)
    warm = _cached_gene_info(cache_dir, digest)
    if warm is not None:
        return warm
    if offline:
        subject = f"snapshot {digest[:DIGEST_PREFIX_LENGTH]}" if digest else GENE_INFO_NAME
        target = _gene_info_path(cache_dir, digest) if digest else cache_dir / "gene_info"
        raise _http.cache_miss("NCBI gene_info", subject, target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    response = _http.get(GENE_INFO_URL, session=session, timeout=300.0, stream=True)
    temporary: str | None = None
    try:
        hasher = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_dir, prefix=f".{GENE_INFO_NAME}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            for block in response.iter_content(chunk_size=1 << 20):
                if block:
                    hasher.update(block)
                    handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        actual = hasher.hexdigest()
        if digest and actual != digest:
            raise ValueError(
                f"downloaded gene_info sha256 {actual} does not match the expected digest "
                f"{digest}; refusing to cache it"
            )
        destination = _gene_info_path(cache_dir, actual)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
    return destination


def _read_json_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _merge_json_cache(path: Path, fetched: Mapping[str, object]) -> dict:
    """Merge `fetched` into the cache file under an exclusive lock and return the merged cache.

    Re-reading inside the lock keeps a concurrent writer's entries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            cache = _read_json_cache(path)
            cache.update(fetched)
            text = json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            atomic_write_text(path, text)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return cache


class EsummaryClient:
    """NCBI E-utilities gene summaries and organism lineage, cache-first.

    <cache_dir>/esummary_cache.json holds {gene_id: summary}; lineage_cache.json holds
    {gene_id: {organism, lineage}}. Only ids missing from a cache are fetched, so a warm cache
    issues no requests; offline, a miss raises CacheMissError. The caches are the
    reproducibility boundary: organism and lineage land in every gene text.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None,
        session: requests.Session | None = None,
        *,
        offline: bool = False,
        email: str | None = None,
        batch_size: int = MAX_ESUMMARY_BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key or None
        self.session = session
        self.offline = offline
        self.email = clean(email or os.environ.get("NCBI_EMAIL"))
        self.batch_size = min(max(1, batch_size), MAX_ESUMMARY_BATCH_SIZE)
        self.sleep = sleep

    @property
    def summary_path(self) -> Path:
        return self.cache_dir / "esummary_cache.json"

    @property
    def lineage_path(self) -> Path:
        return self.cache_dir / "lineage_cache.json"

    def _params(self, params: Mapping[str, str]) -> dict[str, str]:
        # `tool` and `email` are NCBI's recommended contact fields, not required ones, and
        # neither changes the response; an unset email is omitted rather than failing the run.
        result = {**params, "tool": EUTILS_TOOL}
        if self.email:
            result["email"] = self.email
        if self.api_key:
            result["api_key"] = self.api_key
        return result

    def _get(self, endpoint: str, params: Mapping[str, str]) -> requests.Response:
        response = _http.get(
            f"{EUTILS}/{endpoint}",
            params=self._params(params),
            session=self.session,
            sleep=self.sleep,
        )
        if not self.api_key:
            self.sleep(NO_KEY_PACING_SECONDS)
        return response

    def summary_texts(self, gene_ids: Sequence[str]) -> dict[str, str]:
        """{gene_id: summary} for `gene_ids` ('' when NCBI has none)."""
        cache = _read_json_cache(self.summary_path)
        missing = list(dict.fromkeys(gid for gid in gene_ids if gid not in cache))
        if missing and self.offline:
            subject = f"gene ids {', '.join(missing)}"
            raise _http.cache_miss("gene summary", subject, self.summary_path)
        fetched: dict[str, str] = {}
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            params = {"db": "gene", "id": ",".join(chunk), "retmode": "json"}
            payload = self._get("esummary.fcgi", params).json()["result"]
            for gene_id in payload.get("uids", []):
                fetched[gene_id] = payload[gene_id].get("summary", "") or ""
        if missing:
            cache = _merge_json_cache(self.summary_path, fetched)
        return {gid: cache.get(gid, "") for gid in gene_ids}

    def lineage(self, gene_ids: Sequence[str]) -> tuple[str, str]:
        """(organism, lineage) of the lowest cached GeneID in `gene_ids`, else of the lowest.

        Lineage is per organism, so any human GeneID gives the same value; accepting any cached
        id keeps an offline run working when its scope differs from the run that filled the cache.
        """
        ordered = sorted(set(gene_ids), key=int)
        if not ordered:
            raise ValueError("lineage needs at least one gene id")
        cache = _read_json_cache(self.lineage_path)
        for gene_id in ordered:
            entry = cache.get(gene_id)
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("organism"), str)
                and isinstance(entry.get("lineage"), str)
            ):
                return entry["organism"], entry["lineage"]
        first = ordered[0]
        if self.offline:
            # Never fall back to a default organism: offline output comes from the cache only.
            raise _http.cache_miss("lineage", f"gene {first}", self.lineage_path)
        params = {"db": "gene", "id": first, "retmode": "xml"}
        root = ET.fromstring(self._get("efetch.fcgi", params).content)
        organism = root.findtext(".//Org-ref/Org-ref_taxname") or HUMAN
        lineage = root.findtext(".//OrgName/OrgName_lineage") or ""
        _merge_json_cache(self.lineage_path, {first: {"organism": organism, "lineage": lineage}})
        return organism, lineage

    def summaries(self, gene_ids: Sequence[str]) -> dict[str, dict[str, object]]:
        """{gene_id: {organism, lineage, gene_summary}}, the E-utilities part of a gene record."""
        ids = list(dict.fromkeys(gene_ids))
        if not ids:
            return {}
        texts = self.summary_texts(ids)
        organism, lineage = self.lineage(ids)
        if organism != HUMAN:
            raise ValueError(f"NCBI lineage returned non-human organism {organism!r}")
        return {
            gid: {"organism": organism, "lineage": lineage, "gene_summary": texts[gid]}
            for gid in ids
        }

    def provenance(self) -> dict[str, dict[str, object]]:
        """{name: {url, sha256, release}} of both caches, for provenance['inputs']."""
        return {
            name: {
                "url": EUTILS,
                "sha256": sha256_file(path) if path.is_file() else None,
                "release": None,
            }
            for name, path in (
                ("ncbi_esummary", self.summary_path),
                ("ncbi_lineage", self.lineage_path),
            )
        }


def _resolution_inputs(symbol: str) -> tuple[str, str]:
    """(key, Ensembl id) to resolve: `ENSG...` by id; `<stem>_ENSG...` by id, then by stem."""
    if _ENSEMBL_BARE.fullmatch(symbol):
        return symbol, symbol
    match = _ENSEMBL_SUFFIX.fullmatch(symbol)
    if match:
        return match["stem"], match["ensembl"]
    return symbol, ""


def _render_fresh(
    key: str,
    gene_id: str,
    index: GeneInfoIndex,
    records: Mapping[str, Mapping[str, object]],
) -> str:
    fields = index.fields(gene_id)
    fields.update(records[gene_id])
    return render_gene(key, fields)


def describe_genetic_perts(
    keys: Sequence[str],
    index: GeneInfoIndex,
    esummary: EsummaryClient,
    ensembl_ids: Mapping[str, str],
    control_label: str,
) -> dict[str, str]:
    """Perturbation text for one genetic dataset, in sort_key order.

    Every non-control key must resolve (the preprocessed dir's Ensembl id, official symbol,
    unambiguous synonym); the control key, when present, gets CONTROL_TEXT.
    """
    ordered = sorted(set(keys), key=sort_key)
    resolved: dict[str, str] = {}
    failures: list[str] = []
    for key in ordered:
        if key == control_label:
            continue
        gene_id, method = index.resolve_with_method(key, ensembl_ids.get(key, ""))
        if gene_id is None:
            failures.append(f"{key} ({method})")
        else:
            resolved[key] = gene_id
    if failures:
        raise ValueError(f"unresolved production records: {', '.join(failures)}")
    records = esummary.summaries(sorted(set(resolved.values()), key=int))
    return {
        key: CONTROL_TEXT
        if key == control_label
        else _render_fresh(key, resolved[key], index, records)
        for key in ordered
    }


def describe_gene_queries(
    genes: Sequence[str],
    prior: Mapping[str, str] | None,
    pert_output: Mapping[str, str],
    pert_keys: Sequence[str],
    index: GeneInfoIndex,
    esummary: EsummaryClient,
) -> dict[str, str]:
    """Gene text for arbitrary symbols, in sort_key order, by tier.

    1. `prior[g]` verbatim; 2. `pert_output[g]` verbatim when g is a genetic perturbation key;
    3. a fresh render from `index` + E-utilities; 4. the minimal "Gene Name: <g>" render.
    A copied symbol never reaches `resolve`, so it cannot be re-derived from another vintage.
    """
    prior = prior or {}
    copyable = set(pert_keys)
    ordered = sorted(set(genes), key=sort_key)
    copied: dict[str, str] = {}
    fresh: dict[str, str] = {}
    for symbol in ordered:
        if symbol in prior:
            copied[symbol] = prior[symbol]
        elif symbol in copyable and symbol in pert_output:
            copied[symbol] = pert_output[symbol]
        else:
            key, ensembl_id = _resolution_inputs(symbol)
            gene_id, _ = index.resolve_with_method(key, ensembl_id)
            if gene_id is not None:
                fresh[symbol] = gene_id
    records = esummary.summaries(sorted(set(fresh.values()), key=int))
    result: dict[str, str] = {}
    for symbol in ordered:
        if symbol in copied:
            result[symbol] = copied[symbol]
        elif symbol in fresh:
            result[symbol] = _render_fresh(symbol, fresh[symbol], index, records)
        else:
            result[symbol] = render_gene(symbol, None)
    return result
