"""Context descriptions: Cellosaurus records (release-pinned, cached) rendered to text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from pie.data.preprocessed import CONTEXTS_FILE
from pie.sources.text import _http
from pie.sources.text.context_file import (
    _ACCESSION,
    StimulationEntry,
    load_context_file,
)
from pie.utils import sha256_file, write_json

if TYPE_CHECKING:
    from pie.data.preprocessed import PreprocessedDir

RELEASE_URL = "https://api.cellosaurus.org/release-info"
CELL_LINE_URL = "https://api.cellosaurus.org/cell-line/{accession}?format=json"
_RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*", re.ASCII)


def _clean(value: object) -> str:
    """Stripped string; None and '-' are blank."""
    if value is None:
        return ""
    result = str(value).strip()
    return "" if result == "-" else result


def _normalize_list(values: Iterable[object]) -> list[str]:
    """Clean, case-insensitively deduplicate (first spelling wins), and sort."""
    unique: dict[str, str] = {}
    for value in values:
        normalized = _clean(value)
        if normalized:
            unique.setdefault(normalized.casefold(), normalized)
    return sorted(unique.values(), key=lambda value: (value.casefold(), value))


def _normalize_value(value: object) -> str:
    """Collapse control and repeated whitespace of one rendered value."""
    return " ".join(_clean(value).split())


def sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)


ENTRY_FIELDS = (
    "context_name",
    "recommended_name",
    "synonyms",
    "cellosaurus_id",
    "category",
    "organism",
    "ncbi_taxonomy_id",
    "sex",
    "age",
    "populations",
    "diseases",
    "derived_from_sites",
    "cell_type",
    "parent_cell_lines",
    "genetic_integrations",
    "sequence_variations",
    "cross_references",
    "embedding_text",
)


def _only_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        records = payload["Cellosaurus"]["cell-line-list"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Cellosaurus response must contain exactly one cell-line-list record"
        ) from error
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("Cellosaurus response must contain exactly one cell-line-list record")
    return records[0]


def _primary_accession(record: dict) -> str:
    primary = [
        _clean(item.get("value"))
        for item in record.get("accession-list", [])
        if isinstance(item, dict) and _clean(item.get("type")).casefold() == "primary"
    ]
    primary = [value for value in primary if value]
    if len(primary) != 1:
        raise ValueError("Cellosaurus record must contain exactly one primary accession")
    return primary[0]


def _verify_requested_accession(payload: dict, accession: str) -> None:
    primary = _primary_accession(_only_record(payload))
    if primary != accession:
        raise ValueError(
            f"requested accession {accession!r} does not match primary accession {primary!r}"
        )


def _xref_fields(item: object) -> tuple[str, str]:
    if not isinstance(item, dict):
        return "", ""
    return _clean(item.get("database")), _clean(item.get("accession"))


def _first_xref(container: object) -> tuple[str, str]:
    if not isinstance(container, dict):
        return "", ""
    xrefs = container.get("xref-list", [])
    if not isinstance(xrefs, list) or not xrefs:
        return "", ""
    return _xref_fields(xrefs[0])


def _normalize_records(records: list[dict], fields: tuple[str, ...]) -> list[dict]:
    unique: dict[tuple[str, ...], dict] = {}
    for source in records:
        record = {field: _clean(source.get(field)) for field in fields}
        key = tuple(record[field].casefold() for field in fields)
        if any(key):
            unique.setdefault(key, record)
    return sorted(
        unique.values(),
        key=lambda record: tuple((record[field].casefold(), record[field]) for field in fields),
    )


_AGE = re.compile(r"^(?:([0-9]+)Y)?(?:([0-9]+)M)?(?:([0-9]+)D)?$", re.ASCII)
_AGE_UNITS = ("year", "month", "day")

# Cellosaurus controlled developmental-stage terms, carried through verbatim.
_DESCRIPTIVE_AGES = frozenset(
    {
        "Adolescent",
        "Adult",
        "Blastocyst stage",
        "Children",
        "Embryo",
        "Fetus",
        "Juvenile",
        "Newborn",
    }
)


def _humanize_age(raw: object) -> str:
    if raw == "Age unspecified":
        return ""
    value = _clean(raw)
    if not value:
        return ""
    if not isinstance(raw, str) or raw != value:
        raise ValueError(f"unrecognized Cellosaurus age {value!r}")
    if value in _DESCRIPTIVE_AGES:
        return value
    match = _AGE.fullmatch(value)
    if match is None or not any(match.groups()):
        raise ValueError(
            f"unrecognized Cellosaurus age {value!r}; if this is a controlled "
            "developmental stage, add it to _DESCRIPTIVE_AGES"
        )
    parts = []
    for amount, unit in zip(match.groups(), _AGE_UNITS, strict=True):
        if amount is None:
            continue
        count = int(amount)
        parts.append(f"{count} {unit if count == 1 else unit + 's'}")
    return ", ".join(parts)


def _species(record: dict) -> tuple[str, str]:
    species = record.get("species-list", [])
    if not isinstance(species, list) or len(species) != 1:
        raise ValueError("Cellosaurus record must contain exactly one human species")
    item = species[0]
    if (
        not isinstance(item, dict)
        or _clean(item.get("database")) != "NCBI_TaxID"
        or _clean(item.get("accession")) != "9606"
    ):
        raise ValueError("Cellosaurus record must contain exactly one human species")
    organism = _clean(item.get("label"))
    if organism == "Homo sapiens (Human)":
        organism = "Homo sapiens"
    return organism, "9606"


def _names(record: dict) -> tuple[str, list[str]]:
    identifiers: list[str] = []
    synonyms: list[str] = []
    for item in record.get("name-list", []):
        if not isinstance(item, dict):
            continue
        value = _clean(item.get("value"))
        kind = _clean(item.get("type")).casefold()
        if kind == "identifier" and value:
            identifiers.append(value)
        elif kind == "synonym":
            synonyms.append(value)
    return (identifiers[0] if identifiers else ""), _normalize_list(synonyms)


def _diseases(record: dict) -> list[dict]:
    return _normalize_records(
        [item for item in record.get("disease-list", []) if isinstance(item, dict)],
        ("label", "database", "accession"),
    )


def _derived_sites(record: dict) -> list[dict]:
    sites: list[dict] = []
    for wrapper in record.get("derived-from-site-list", []):
        site = wrapper.get("site", {}) if isinstance(wrapper, dict) else {}
        if not isinstance(site, dict):
            continue
        database, accession = _first_xref(site)
        sites.append(
            {
                "name": _clean(site.get("value")),
                "database": database,
                "accession": accession,
            }
        )
    return _normalize_records(sites, ("name", "database", "accession"))


def _cell_type(record: dict) -> dict:
    result = {"name": "", "database": "", "accession": ""}
    values = record.get("cell-type-list", [])
    item = values[0] if isinstance(values, list) and values else record.get("cell-type", {})
    if not isinstance(item, dict):
        return result
    database, accession = _first_xref(item)
    if not database and not accession:
        database, accession = _xref_fields(item.get("xref"))
    return {
        "name": _clean(item.get("label") or item.get("value")),
        "database": database,
        "accession": accession,
    }


def _parents(record: dict) -> list[dict]:
    parents: list[dict] = []
    for item in record.get("parent-list", []):
        if not isinstance(item, dict) or _clean(item.get("type")).casefold() != "derived-from":
            continue
        name = item.get("name", {})
        accession = item.get("accession", {})
        parents.append(
            {
                "name": _clean(name.get("value")) if isinstance(name, dict) else _clean(name),
                "cellosaurus_id": _clean(accession.get("value"))
                if isinstance(accession, dict)
                else _clean(accession),
            }
        )
    for item in record.get("derived-from", []):
        if not isinstance(item, dict):
            continue
        parents.append(
            {
                "name": _clean(item.get("label")),
                "cellosaurus_id": _clean(item.get("accession")),
            }
        )
    return _normalize_records(parents, ("name", "cellosaurus_id"))


def _nested_value(value: object) -> str:
    if isinstance(value, dict):
        return _clean(value.get("label") or value.get("value"))
    return _clean(value)


def _genetic_integrations(record: dict) -> list[dict]:
    integrations: list[dict] = []
    for item in record.get("genetic-integration-list", []):
        if not isinstance(item, dict):
            continue
        target = item.get("target", {})
        xref = item.get("xref", {})
        database, accession = _xref_fields(xref)
        if not database and not accession:
            database, accession = _first_xref(target)
        if not database and not accession:
            database, accession = _xref_fields(item)
        integrations.append(
            {
                "method": _nested_value(item.get("method")),
                "label": (
                    _clean(item.get("label")) or _nested_value(target) or _nested_value(xref)
                ),
                "database": database,
                "accession": accession,
            }
        )
    for comment in record.get("comment-list", []):
        if (
            isinstance(comment, dict)
            and _clean(comment.get("category")).casefold() == "genetic integration"
        ):
            integrations.append(
                {
                    "method": "",
                    "label": _clean(comment.get("value")),
                    "database": "",
                    "accession": "",
                }
            )
    return _normalize_records(integrations, ("method", "label", "database", "accession"))


def _sequence_variations(record: dict) -> list[dict]:
    variations: list[dict] = []
    for item in record.get("sequence-variation-list", []):
        if not isinstance(item, dict):
            continue
        genes = _normalize_list(
            xref.get("label")
            for xref in item.get("xref-list", [])
            if isinstance(xref, dict) and _clean(xref.get("database")) == "HGNC"
        )
        variations.append(
            {
                "genes": genes,
                "variation_type": _clean(item.get("variation-type")),
                "mutation_description": _clean(item.get("mutation-description")),
                "zygosity": _clean(item.get("zygosity-type")),
                "note": _clean(item.get("variation-note")),
            }
        )
    unique: dict[tuple, dict] = {}
    for variation in variations:
        key = (
            tuple(gene.casefold() for gene in variation["genes"]),
            *(
                variation[field].casefold()
                for field in (
                    "variation_type",
                    "mutation_description",
                    "zygosity",
                    "note",
                )
            ),
        )
        if any(key):
            unique.setdefault(key, variation)
    return sorted(
        unique.values(),
        key=lambda item: (
            tuple((gene.casefold(), gene) for gene in item["genes"]),
            *(
                (item[field].casefold(), item[field])
                for field in (
                    "variation_type",
                    "mutation_description",
                    "zygosity",
                    "note",
                )
            ),
        ),
    )


def _cross_references(record: dict) -> dict[str, list[str]]:
    databases = {
        "ATCC": "atcc",
        "DepMap": "depmap",
        "Cell_Model_Passport": "cell_model_passport",
    }
    values: dict[str, list[str]] = {"atcc": [], "depmap": [], "cell_model_passport": []}
    for item in record.get("xref-list", []):
        if not isinstance(item, dict):
            continue
        target = databases.get(_clean(item.get("database")))
        if target is not None:
            values[target].append(_clean(item.get("accession")))
    return {key: _normalize_list(items) for key, items in values.items()}


def _embedding_list(values) -> str:
    return ", ".join(_normalize_list(_normalize_value(value) for value in values))


def _render_integration(item: dict) -> str:
    label = _clean(item.get("label"))
    method = _clean(item.get("method"))
    if label and method:
        return f"{label} via {method}"
    return label or method


def _render_variation(item: dict) -> str:
    genes = "/".join(_normalize_list(item.get("genes", [])))
    description = ", ".join(
        value
        for value in (
            _clean(item.get("variation_type")),
            _clean(item.get("mutation_description")),
        )
        if value
    )
    result = f"{genes}: {description}" if genes and description else genes or description
    details = "; ".join(
        value for value in (_clean(item.get("zygosity")), _clean(item.get("note"))) if value
    )
    if details:
        result = f"{result} ({details})" if result else details
    return result


def render_context_embedding(entry: Mapping[str, Any]) -> str:
    """Render descriptive context metadata without structured identifiers."""
    scalar = _normalize_value
    values = [
        ("Context Name", scalar(entry.get("context_name"))),
        ("Recommended Name", scalar(entry.get("recommended_name"))),
        ("Synonyms", _embedding_list(entry.get("synonyms", []))),
        ("Cell Line Category", scalar(entry.get("category"))),
        ("Organism", scalar(entry.get("organism"))),
        ("Sex", scalar(entry.get("sex"))),
        ("Age", scalar(entry.get("age"))),
        ("Population", _embedding_list(entry.get("populations", []))),
        (
            "Disease",
            _embedding_list(item.get("label") for item in entry.get("diseases", [])),
        ),
        (
            "Derived From Site",
            _embedding_list(item.get("name") for item in entry.get("derived_from_sites", [])),
        ),
        ("Cell Type", scalar(entry.get("cell_type", {}).get("name"))),
        (
            "Parent Cell Line",
            _embedding_list(item.get("name") for item in entry.get("parent_cell_lines", [])),
        ),
        (
            "Genetic Integration",
            _embedding_list(
                _render_integration(item) for item in entry.get("genetic_integrations", [])
            ),
        ),
        (
            "Sequence Variation",
            _embedding_list(
                _render_variation(item) for item in entry.get("sequence_variations", [])
            ),
        ),
    ]
    return " ;\n".join(f"{label}: {value}" for label, value in values if value)


def build_context_entry(runtime_key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Parse one official Cellosaurus response into the exact context schema."""
    record = _only_record(payload)
    primary = _primary_accession(record)
    recommended_name, synonyms = _names(record)
    organism, taxonomy_id = _species(record)
    populations = _normalize_list(
        item.get("value")
        for item in record.get("comment-list", [])
        if isinstance(item, dict) and _clean(item.get("category")) == "Population"
    )
    entry: dict[str, Any] = {
        "context_name": _clean(runtime_key),
        "recommended_name": recommended_name,
        "synonyms": synonyms,
        "cellosaurus_id": primary,
        "category": _clean(record.get("category")),
        "organism": organism,
        "ncbi_taxonomy_id": taxonomy_id,
        "sex": _clean(record.get("sex")),
        "age": _humanize_age(record.get("age")),
        "populations": populations,
        "diseases": _diseases(record),
        "derived_from_sites": _derived_sites(record),
        "cell_type": _cell_type(record),
        "parent_cell_lines": _parents(record),
        "genetic_integrations": _genetic_integrations(record),
        "sequence_variations": _sequence_variations(record),
        "cross_references": _cross_references(record),
        "embedding_text": "",
    }
    entry["embedding_text"] = render_context_embedding(entry)
    return entry


class CellosaurusReleaseError(RuntimeError):
    """The Cellosaurus release differs from the pinned one."""


def release_version(payload: Mapping[str, object]) -> str:
    try:
        header = payload["Cellosaurus"]["header"]  # type: ignore[index]
        version = _clean(header["release"]["version"])
    except (KeyError, TypeError) as error:
        raise ValueError("Cellosaurus release metadata has no release version") from error
    if not version:
        raise ValueError("Cellosaurus release metadata has no release version")
    return version


class CellosaurusClient:
    """Cellosaurus records for one pinned release, cached as <cache_dir>/<release>/<accession>.json.

    The API serves only the current release, so the first `get` checks the release (live, or the
    cached release-info.json when offline) and refuses to continue on a mismatch.
    """

    def __init__(
        self,
        cache_dir: Path,
        release: str,
        offline: bool,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(release, str) or _RELEASE.fullmatch(release) is None:
            raise ValueError(f"invalid Cellosaurus release: {release!r}")
        self.cache_dir = Path(cache_dir)
        self.release = release
        self.offline = offline
        self.session = session
        self._release_checked = False
        self.used: set[str] = set()

    @property
    def release_info_path(self) -> Path:
        return self.cache_dir / "release-info.json"

    def record_path(self, accession: str) -> Path:
        return self.cache_dir / self.release / f"{accession}.json"

    def check_release(self) -> None:
        if self._release_checked:
            return
        path = self.release_info_path
        if self.offline:
            if not path.is_file():
                raise _http.cache_miss("Cellosaurus release", path.name, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _http.get(RELEASE_URL, session=self.session).json()
        version = release_version(payload)
        if version != self.release:
            raise CellosaurusReleaseError(
                f"Cellosaurus release {version} differs from the pinned {self.release}; "
                f"pass --cellosaurus-release {version} to accept the new release"
            )
        if not self.offline:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, payload)
        self._release_checked = True

    def get(self, accession: str) -> dict[str, object]:
        if not isinstance(accession, str) or _ACCESSION.fullmatch(accession) is None:
            raise ValueError(f"invalid Cellosaurus accession: {accession!r}")
        self.check_release()
        path = self.record_path(accession)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            _verify_requested_accession(payload, accession)
            self.used.add(accession)
            return payload
        if self.offline:
            raise _http.cache_miss("Cellosaurus", f"accession {accession}", path)
        url = CELL_LINE_URL.format(accession=accession)
        payload = _http.get(url, session=self.session).json()
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object for {accession}")
        _verify_requested_accession(payload, accession)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        self.used.add(accession)
        return payload

    def provenance(self, accessions: Iterable[str] | None = None) -> dict[str, object]:
        """Release, API root and sha256 of each record (default: every record `get` returned)."""
        chosen = self.used if accessions is None else set(accessions)
        return {
            "release": self.release,
            "url": "https://api.cellosaurus.org",
            "records": {
                acc: sha256_file(self.record_path(acc)) for acc in sorted(chosen)
            },
        }


def _stimulation_segments(stimulation: StimulationEntry) -> list[str]:
    scalar = _normalize_value
    name = scalar(stimulation.name)
    abbreviation = scalar(stimulation.abbreviation)
    return [
        f"Stimulation: {name} ({abbreviation})" if abbreviation else f"Stimulation: {name}",
        f"Stimulation Family: {scalar(stimulation.family)}",
        f"Stimulation Receptors: {', '.join(scalar(r) for r in stimulation.receptors)}",
        f"Stimulation Signaling: {scalar(stimulation.signaling)}",
        f"Stimulation Description: {scalar(stimulation.description)}",
    ]


def render_context(
    key: str, record: Mapping[str, Any], stimulation: StimulationEntry | None
) -> str:
    """Cellosaurus context render for `key`, plus the stimulation segments when given."""
    base = str(build_context_entry(key, record)["embedding_text"])
    if stimulation is None:
        return base
    segments = base.split(" ;\n") if base else []
    return " ;\n".join([*segments, *_stimulation_segments(stimulation)])


def describe_contexts(
    datasets: Sequence[PreprocessedDir], client: CellosaurusClient
) -> dict[str, str]:
    """{context: text} for every context of every dataset: datasets in order, keys by sort_key.

    Each dataset's context map is <preprocessed dir>/contexts.yaml.
    """
    texts: dict[str, str] = {}
    for ds in datasets:
        path = ds.path / CONTEXTS_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{ds.path}: no {CONTEXTS_FILE} for dataset {ds.dataset!r}; rebuild the dir with "
                f"pie prep contexts=<file>, or copy a context map to {path}"
            )
        mapping = load_context_file(path)
        missing = [key for key in ds.contexts if key not in mapping.contexts]
        if missing:
            raise KeyError(f"{path.name} has no entry for contexts: {', '.join(missing)}")
        for key in sorted(ds.contexts, key=sort_key):
            entry = mapping.contexts[key]
            stimulation = mapping.stimulations[entry.stimulation] if entry.stimulation else None
            text = render_context(key, client.get(entry.cellosaurus), stimulation)
            if texts.get(key, text) != text:
                raise ValueError(f"context {key!r} renders differently in dataset {ds.dataset!r}")
            texts[key] = text
    return texts
