"""Drug-dose perturbation text: PubChem PUG-REST records rendered as embedding text."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import requests

from pie.sources.text import _http
from pie.utils import atomic_write_text

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_TIMEOUT_SECONDS = 60
MIN_INTERVAL_SECONDS = 0.25
RETRY_BACKOFF_BASE_SECONDS = 1.0
DOSE_UNIT = "uM"
MAX_SYNONYMS = 20
RENDER_SYNONYMS = 10
CONTROL_TEXT = "Vehicle Control: DMSO ;\nDose: 0 micromolar"
PROPERTY_NAMES = (
    "Title",
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
    "XLogP",
    "TPSA",
    "Complexity",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "Charge",
)
RELATIONSHIPS = frozenset({"primary", "component", "isomer"})
ENDPOINTS: dict[str, str] = {
    "properties": f"compound/cid/{{cid}}/property/{','.join(PROPERTY_NAMES)}/JSON",
    "synonyms": "compound/cid/{cid}/synonyms/JSON",
    "descriptions": "compound/cid/{cid}/description/JSON",
}
ENDPOINT_MARKERS: dict[str, str] = {
    "properties": "/property/",
    "synonyms": "/synonyms/",
    "descriptions": "/description/",
}
# Reviewed identities for drugs whose metadata CID is absent or covers only part of the compound.
IDENTITY_OVERRIDES: dict[str, tuple[tuple[int, str], ...]] = {
    "Sacubitril/Valsartan": ((24755620, "primary"),),
    "Verteporfin": ((5362420, "isomer"), (9940086, "isomer")),
}


def _embedding_value(value: object) -> str:
    """Strip, map None and '-' to '', and collapse inner whitespace."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text == "-" else " ".join(text.split())


def _numeric_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _embedding_value(value)


def parse_drug_dose(key: str) -> tuple[str, float]:
    """Split a label key '<raw name>_<dose>uM' into (raw name with its whitespace, dose)."""
    name, sep, dose = key.rpartition("_")
    if not sep or not name.strip() or not dose.endswith(DOSE_UNIT):
        raise ValueError(f"not a drug-dose key: {key!r}")
    try:
        value = float(dose[: -len(DOSE_UNIT)])
    except ValueError:
        raise ValueError(f"not a drug-dose key: {key!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"not a drug-dose key: {key!r}")
    return name, value


def is_control(key: str, control_label: str) -> bool:
    """True when the key, or its parsed compound name, equals the control label."""
    if key == control_label:
        return True
    try:
        name, _dose = parse_drug_dose(key)
    except ValueError:
        return False
    return name.strip() == control_label


# ---- payload validation ----

_STRING_PROPERTY_FIELDS = frozenset(
    {
        "Title",
        "MolecularFormula",
        "CanonicalSMILES",
        "IsomericSMILES",
        "SMILES",
        "ConnectivitySMILES",
        "InChI",
        "InChIKey",
    }
)
_NUMERIC_PROPERTY_FIELDS = frozenset(
    {
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "Complexity",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Charge",
    }
)
_DESCRIPTION_RESPONSE_FIELDS = (
    "Description",
    "DescriptionSourceName",
    "DescriptionURL",
)


def _invalid_field(endpoint: str, field: str, cid: int) -> ValueError:
    return ValueError(f"malformed PubChem {endpoint} field {field} for CID {cid}")


_NONNEGATIVE_NUMERIC_FIELDS = frozenset({"TPSA", "Complexity"})
_NONNEGATIVE_INTEGRAL_FIELDS = frozenset(
    {"HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount"}
)


def _validate_numeric_property(field: str, value: object, cid: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise _invalid_field("properties", field, cid)
    if isinstance(value, str) and not value.strip():
        raise _invalid_field("properties", field, cid)
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise _invalid_field("properties", field, cid) from None
    if not number.is_finite():
        raise _invalid_field("properties", field, cid)
    if field == "MolecularWeight" and number <= 0:
        raise _invalid_field("properties", field, cid)
    if field in _NONNEGATIVE_NUMERIC_FIELDS and number < 0:
        raise _invalid_field("properties", field, cid)
    if field in _NONNEGATIVE_INTEGRAL_FIELDS and (
        number < 0 or number != number.to_integral_value()
    ):
        raise _invalid_field("properties", field, cid)
    if field == "Charge" and number != number.to_integral_value():
        raise _invalid_field("properties", field, cid)


def _validate_endpoint_fields(endpoint: str, rows: list[dict], cid: int) -> None:
    for row in rows:
        if endpoint == "synonyms":
            if "Synonym" not in row:
                continue
            values = row["Synonym"]
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise _invalid_field(endpoint, "Synonym", cid)
        elif endpoint == "descriptions":
            for field in _DESCRIPTION_RESPONSE_FIELDS:
                if field in row and not isinstance(row[field], str):
                    raise _invalid_field(endpoint, field, cid)
        else:
            for field in _STRING_PROPERTY_FIELDS:
                if field in row and not isinstance(row[field], str):
                    raise _invalid_field(endpoint, field, cid)
            for field in _NUMERIC_PROPERTY_FIELDS:
                if field not in row:
                    continue
                _validate_numeric_property(field, row[field], cid)


def _validate_payload(endpoint: str, payload: object, cid: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"malformed PubChem {endpoint} response for CID {cid}")
    try:
        rows = (
            payload["PropertyTable"]["Properties"]
            if endpoint == "properties"
            else payload["InformationList"]["Information"]
        )
    except (KeyError, TypeError):
        raise ValueError(f"malformed PubChem {endpoint} response for CID {cid}") from None
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) or row.get("CID") != cid for row in rows)
    ):
        raise ValueError(f"malformed or mismatched PubChem {endpoint} response for CID {cid}")
    _validate_endpoint_fields(endpoint, rows, cid)
    return payload


class PubChemClient:
    """PUG-REST client with a per-CID JSON cache; a cached endpoint is never fetched again."""

    def __init__(
        self,
        cache_dir: Path,
        session: requests.Session | None = None,
        *,
        offline: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "pubchem"
        self.offline = offline
        self._session = session
        self._sleep = sleep
        self._last_request = -math.inf

    def _request_json(self, url: str) -> object:
        wait = self._last_request + MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()
        response = _http.get(
            url,
            session=self._session,
            timeout=REQUEST_TIMEOUT_SECONDS,
            sleep=self._sleep,
            backoff_base=RETRY_BACKOFF_BASE_SECONDS,
        )
        return response.json()

    def record(self, cid: int) -> dict[str, dict]:
        """Validated {'properties', 'synonyms', 'descriptions'} payloads of one CID."""
        if isinstance(cid, bool) or not isinstance(cid, int) or cid <= 0:
            raise ValueError(f"invalid PubChem CID: {cid!r}")
        record: dict[str, dict] = {}
        for endpoint, template in ENDPOINTS.items():
            path = self.cache_dir / str(cid) / f"{endpoint}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
            elif self.offline:
                raise _http.cache_miss(f"PubChem {endpoint}", f"CID {cid}", path)
            else:
                payload = self._request_json(f"{BASE_URL}/{template.format(cid=cid)}")
                _validate_payload(endpoint, payload, cid)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            record[endpoint] = _validate_payload(endpoint, payload, cid)
        return record


# ---- synonym filter patterns ----

_CAS_PATTERN = re.compile(r"^[1-9][0-9]{1,6}-[0-9]{2}-[0-9]$")
_PURE_CID_TITLE = re.compile(r"^(?:PubChem\s+)?CID\s*[:#-]?\s*[1-9]\d*$", re.IGNORECASE)
_INCHI_KEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", re.IGNORECASE)
_EXCLUDED_PREFIX_PATTERN = re.compile(
    r"^(?:(?:CID|CHEBI|CHEMBL|DTXSID)(?=[\s_:.-]*\d)|UNII(?=[\s_:.-]))",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"^(?:(?:https?|ftp)://|www\.)", re.IGNORECASE)


def normalize_synonyms(compound_name: str, title: str, values: Iterable[object]) -> list[str]:
    """Return ordered, human-readable PubChem synonyms suitable for embedding."""
    excluded_names = {
        value.casefold()
        for value in (_embedding_value(compound_name), _embedding_value(title))
        if value
    }
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        synonym = _embedding_value(value)
        folded = synonym.casefold()
        if (
            not synonym
            or folded in excluded_names
            or folded in seen
            or synonym.isdigit()
            or _CAS_PATTERN.fullmatch(synonym)
            or _URL_PATTERN.match(synonym)
            or synonym.casefold().startswith("inchi")
            or _INCHI_KEY_PATTERN.fullmatch(synonym)
            or _EXCLUDED_PREFIX_PATTERN.match(synonym)
        ):
            continue
        seen.add(folded)
        result.append(synonym)
        if len(result) == MAX_SYNONYMS:
            break
    return result


def normalize_descriptions(payload: dict) -> list[dict[str, str]]:
    """Normalize all source-attributed PubChem descriptions in response order."""
    rows = payload.get("InformationList", {}).get("Information", [])
    result = []
    seen_text: set[str] = set()
    for row in rows:
        text = _embedding_value(row.get("Description"))
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        result.append(
            {
                "text": text,
                "source_name": _embedding_value(row.get("DescriptionSourceName")),
                "source_url": _embedding_value(row.get("DescriptionURL")),
            }
        )
    return result


def select_primary_description(descriptions: list[dict[str, str]]) -> str:
    """Select ChEBI, then DrugBank, then first response-order description."""
    for preferred_source in ("chebi", "drugbank"):
        for description in descriptions:
            if description.get("source_name", "").casefold() == preferred_source:
                return _embedding_value(description.get("text"))
    if descriptions:
        return _embedding_value(descriptions[0].get("text"))
    return ""


def build_component(
    cid: int, relationship: str, compound_name: str, record: Mapping[str, dict]
) -> dict:
    """Normalize one semantic component from an already validated raw record."""
    if relationship not in RELATIONSHIPS:
        raise ValueError(f"invalid PubChem relationship: {relationship!r}")
    properties = record["properties"]["PropertyTable"]["Properties"][0]
    synonym_rows = record["synonyms"]["InformationList"]["Information"]
    synonym_values = [synonym for row in synonym_rows for synonym in row.get("Synonym", [])]
    title = _embedding_value(properties.get("Title"))
    entry = {
        "pubchem_cid": str(cid),
        "title": title,
        "relationship": relationship,
        "synonyms": normalize_synonyms(compound_name, title, synonym_values),
        "molecular_formula": _embedding_value(properties.get("MolecularFormula")),
        "molecular_weight": _numeric_value(properties.get("MolecularWeight")),
        "smiles": _embedding_value(properties.get("SMILES", properties.get("IsomericSMILES"))),
        "connectivity_smiles": _embedding_value(
            properties.get("ConnectivitySMILES", properties.get("CanonicalSMILES"))
        ),
        "inchi": _embedding_value(properties.get("InChI")),
        "inchi_key": _embedding_value(properties.get("InChIKey")),
        "xlogp": _numeric_value(properties.get("XLogP")),
        "topological_polar_surface_area": _numeric_value(properties.get("TPSA")),
        "complexity": _numeric_value(properties.get("Complexity")),
        "formal_charge": _numeric_value(properties.get("Charge")),
        "hydrogen_bond_donor_count": _numeric_value(properties.get("HBondDonorCount")),
        "hydrogen_bond_acceptor_count": _numeric_value(properties.get("HBondAcceptorCount")),
        "rotatable_bond_count": _numeric_value(properties.get("RotatableBondCount")),
        "descriptions": normalize_descriptions(record["descriptions"]),
    }
    return entry


_RENDER_FIELDS = (
    ("molecular_formula", "Molecular Formula", ""),
    ("molecular_weight", "Molecular Weight", " g/mol"),
    ("smiles", "SMILES", ""),
    ("connectivity_smiles", "Connectivity SMILES", ""),
    ("xlogp", "XLogP", ""),
    (
        "topological_polar_surface_area",
        "Topological Polar Surface Area",
        " square angstroms",
    ),
    ("complexity", "Complexity", ""),
    ("formal_charge", "Formal Charge", ""),
    ("hydrogen_bond_donor_count", "Hydrogen Bond Donors", ""),
    ("hydrogen_bond_acceptor_count", "Hydrogen Bond Acceptors", ""),
    ("rotatable_bond_count", "Rotatable Bonds", ""),
)


def _append_segment(segments: list[str], label: str, value: object) -> None:
    normalized = _embedding_value(value)
    if normalized:
        segments.append(f"{label}: {normalized}")


def render_embedding(compound_name: str, dose: dict, components: list[dict]) -> str:
    """Render deterministic, identifier-free chemical embedding text."""
    normalized_name = _embedding_value(compound_name)
    segments = [f"Compound Name: {normalized_name}"]
    multiple = len(components) > 1
    selected_descriptions: list[tuple[str, str]] = []

    for component in components:
        relationship = _embedding_value(component.get("relationship"))
        if relationship not in RELATIONSHIPS:
            raise ValueError(f"invalid PubChem relationship: {relationship!r}")
        title = _embedding_value(component.get("title"))
        embedding_title = "" if _PURE_CID_TITLE.fullmatch(title) else title
        if multiple and embedding_title:
            rendered_title = (
                embedding_title
                if relationship == "primary"
                else f"{embedding_title} ({relationship})"
            )
            _append_segment(segments, "PubChem Compound", rendered_title)
        elif not multiple and embedding_title.casefold() != normalized_name.casefold():
            _append_segment(segments, "PubChem Compound", embedding_title)

        synonyms = []
        seen_synonyms: set[str] = set()
        for synonym in component.get("synonyms", []):
            normalized = _embedding_value(synonym)
            folded = normalized.casefold()
            if normalized and folded not in seen_synonyms:
                seen_synonyms.add(folded)
                synonyms.append(normalized)
            if len(synonyms) == RENDER_SYNONYMS:
                break
        if synonyms:
            segments.append(f"Synonyms: {', '.join(synonyms)}")

        smiles = _embedding_value(component.get("smiles"))
        for field, label, suffix in _RENDER_FIELDS:
            value = _embedding_value(component.get(field))
            if field == "connectivity_smiles" and value == smiles:
                continue
            if value:
                segments.append(f"{label}: {value}{suffix}")

        description = select_primary_description(component.get("descriptions", []))
        if description:
            selected_descriptions.append((embedding_title, description))

    if selected_descriptions:
        if multiple:
            description_text = " | ".join(
                f"{title}: {description}" if title else description
                for title, description in selected_descriptions
            )
        else:
            description_text = selected_descriptions[0][1]
        _append_segment(segments, "Compound Description", description_text)

    dose_value = _numeric_value(dose.get("value"))
    dose_unit = _embedding_value(dose.get("unit"))
    if dose_unit.casefold() == "um":
        dose_unit = "micromolar"
    dose_text = " ".join(value for value in (dose_value, dose_unit) if value)
    _append_segment(segments, "Dose", dose_text)
    return " ;\n".join(segments)


def render_drug(key: str, record: Mapping[str, object] | None) -> str:
    """Text of one label key; `record` is {"components": [...]}, or None for the control."""
    if record is None:
        return CONTROL_TEXT
    name, dose = parse_drug_dose(key)
    components = list(record["components"])  # type: ignore[call-overload]
    return render_embedding(name.strip(), {"value": dose, "unit": DOSE_UNIT}, components)


def _metadata_cid(value: object, drug: str) -> int | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"drug {drug!r} has an invalid PubChem CID {value!r}") from None
    if not number.is_integer() or number <= 0:
        raise ValueError(f"drug {drug!r} has an invalid PubChem CID {value!r}")
    return int(number)


def resolve_identities(
    names: Iterable[str], drug_metadata: pd.DataFrame
) -> dict[str, tuple[tuple[int, str], ...]]:
    """Trimmed compound name -> ((cid, relationship), ...), sorted case-insensitively by name."""
    if not {"drug", "pubchem_cid"} <= set(drug_metadata.columns):
        raise ValueError("drug metadata needs columns drug and pubchem_cid")
    table: dict[str, int | None] = {}
    for drug, value in zip(drug_metadata["drug"], drug_metadata["pubchem_cid"], strict=True):
        name = str(drug).strip()
        if name in table:
            raise ValueError(f"duplicate drug {name!r} in the drug metadata")
        table[name] = _metadata_cid(value, name)
    resolved: dict[str, tuple[tuple[int, str], ...]] = {}
    for name in sorted(set(names), key=lambda value: (value.casefold(), value)):
        if name not in table:
            raise ValueError(f"drug {name!r} is missing from the drug metadata")
        cid = table[name]
        override = IDENTITY_OVERRIDES.get(name)
        if override is not None:
            if cid is not None and cid not in {c for c, _rel in override}:
                raise ValueError(
                    f"metadata CID {cid} of {name!r} disagrees with its reviewed override"
                )
            resolved[name] = override
        elif cid is None:
            raise ValueError(f"drug {name!r} has no PubChem CID and no reviewed override")
        else:
            resolved[name] = ((cid, "primary"),)
    return resolved


def describe_drug_perts(
    keys: Sequence[str], drug_metadata: pd.DataFrame, client: PubChemClient, control_label: str
) -> dict[str, str]:
    """{label key: embedding text} in `keys` order; each CID is fetched once."""
    parsed = {key: parse_drug_dose(key) for key in keys if not is_control(key, control_label)}
    identities = resolve_identities((name.strip() for name, _ in parsed.values()), drug_metadata)
    specs_by_cid: dict[int, list[tuple[str, str]]] = {}
    for name, pairs in identities.items():
        for cid, relationship in pairs:
            specs_by_cid.setdefault(cid, []).append((relationship, name))
    components: dict[tuple[str, int, str], dict] = {}
    for cid in sorted(specs_by_cid):
        record = client.record(cid)
        for relationship, name in specs_by_cid[cid]:
            components[(name, cid, relationship)] = build_component(cid, relationship, name, record)
    texts: dict[str, str] = {}
    for key in keys:
        if key not in parsed:
            texts[key] = render_drug(key, None)
            continue
        name = parsed[key][0].strip()
        chosen = [components[(name, cid, rel)] for cid, rel in identities[name]]
        texts[key] = render_drug(key, {"components": chosen})
    return texts
