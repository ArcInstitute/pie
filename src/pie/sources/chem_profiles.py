"""Measured drug-dose profiles from public screens, exported unscaled with missing values as 0.

- ``l1000_tas``: LINCS L1000 transcriptional activity score per cell line (nearest dose and time).
- ``prism_secondary``: PRISM secondary-screen log2 viability per QC-passing cell line.
- ``jump_morphology``: median JUMP Cell Painting well profile per drug.

Drugs are matched to external compounds by exact InChIKey, then by the standardized parent
InChIKey of the SMILES, then by normalized exact name. Every input file is pinned by sha256.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pie.sources.common import base_provenance, download_file, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source

if TYPE_CHECKING:
    import requests

    from pie.data.preprocessed import PreprocessedDir
    from pie.sources.registry import RunContext

PROFILES: tuple[str, ...] = ("l1000_tas", "prism_secondary", "jump_morphology")
L1000_TIME_HOURS = 24.0
GSE70138_RELEASE = "2017-03-06"
_L1000_SUFFIX: dict[str, str] = {"GSE92742": "", "GSE70138": f"_{GSE70138_RELEASE}"}
_L1000_SHA256: dict[str, dict[str, str]] = {
    "GSE92742": {
        "pert_info": "b1945b3fde51021865b12269929cc78ccf2665be27f53a4da9336e4ddcf4b42f",
        "sig_info": "19da29c0ee12ddf27f9698cd0da40beaff58657dcde9d382aae068737e831299",
        "sig_metrics": "54f19003e5e3445bc293347cca004eeb066e22e97f5eec50f1605b393e90f466",
    },
    "GSE70138": {
        "pert_info": "000171e8ce17cb00a3e80a907d97f0bc4218077eb4cb2ac03f4e0f9f4f2e5493",
        "sig_info": "b78284478d62029f600d4397334c538d363123110affd2454bf0055bd0678e01",
        "sig_metrics": "05dd9139f7db1caef03b534611c0bcf97dd071a764445688eaa7e6c47e50b04c",
    },
}
_JUMP_COMPOUND_URL = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/"
    "e5e20e73df0067bff48076d1c437259503f6aa60/metadata/compound.csv.gz"
)
_JUMP_PROFILES_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/"
    "workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet"
)
_L1000_CELLS = 83
_PRISM_CELLS = 480
_JUMP_COORDS = 737
_DOSE_KEY = re.compile("(.+)_([0-9.eE+-]+)(?:uM|\u00b5M|\u03bcM)")


def l1000_file(series: str, kind: str) -> str:
    """Pinned L1000 file name; series GSE92742|GSE70138, kind pert_info|sig_info|sig_metrics."""
    return f"{series}_Broad_LINCS_{kind}{_L1000_SUFFIX[series]}.txt.gz"


def _geo_url(series: str, name: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series[:5]}nnn/{series}/suppl/{name}"


def _l1000_releases() -> dict[str, dict[str, str]]:
    return {
        l1000_file(series, kind): {
            "url": _geo_url(series, l1000_file(series, kind)),
            "sha256": sha256,
            "source": "l1000_tas",
        }
        for series, files in _L1000_SHA256.items()
        for kind, sha256 in files.items()
    }


RELEASES: dict[str, dict[str, str]] = {
    **_l1000_releases(),
    "secondary-screen-cell-line-info.csv": {
        "url": "https://ndownloader.figshare.com/files/20237769",
        "sha256": "b93436b5f4bcf5fd14589697be4ca8f99ffd99d639223080380bc514c3718d27",
        "source": "prism_secondary",
    },
    "secondary-screen-replicate-collapsed-treatment-info.csv": {
        "url": "https://ndownloader.figshare.com/files/20237763",
        "sha256": "9d0d1fb4faa87a63cd84965ec5e2b55a9df5680520a41b029abb679cfd4384f7",
        "source": "prism_secondary",
    },
    "secondary-screen-replicate-collapsed-logfold-change.csv": {
        "url": "https://ndownloader.figshare.com/files/20237757",
        "sha256": "a358beb9efbc96b3d777cbb0212e4cd724080f17970d241627ccd17855d21939",
        "source": "prism_secondary",
    },
    "jump_compound.csv.gz": {
        "url": _JUMP_COMPOUND_URL,
        "sha256": "8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a",
        "source": "jump_morphology",
    },
    "jump_profiles.parquet": {
        "url": _JUMP_PROFILES_URL,
        "sha256": "1dd9b76ce9635cc98ea2c6a58f4c1d6ed6aafc1a3990ddcb997162d16582c00f",
        "source": "jump_morphology",
    },
}


# --- drugs ------------------------------------------------------------------------------------


def parse_perturbation(key: str) -> dict[str, Any]:
    """{"pert", "drug", "dose_uM"} of a ``<drug>_<dose>uM`` key (micro-sign spellings accepted)."""
    match = _DOSE_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"expected a <drug>_<dose>uM key, got {key!r}")
    drug, number = match.groups()
    dose = float(number)
    if not 0 < dose < float("inf"):
        raise ValueError(f"invalid dose in {key!r}")
    return {"pert": key, "drug": drug, "dose_uM": dose}


def namekey(value: object) -> str:
    """The lower-case alphanumeric characters of a name."""
    return re.sub("[^a-z0-9]", "", str(value).lower())


def _inchi_keys(smiles: str) -> tuple[str | None, str | None]:
    """(InChIKey, charge-parent InChIKey) of a SMILES string; (None, None) if RDKit rejects it."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None, None
    return Chem.MolToInchiKey(mol), Chem.MolToInchiKey(rdMolStandardize.ChargeParent(mol))


class CompoundMatcher:
    """Exact matching of external compounds to drugs, with ambiguity narrowed by exact name."""

    def __init__(self, identities: Sequence[Mapping[str, Any]]) -> None:
        self.keys: dict[str, set[str]] = {}
        self.names: dict[str, set[str]] = {}
        for row in identities:
            for key in (row.get("inchi_key"), row.get("parent_inchi_key")):
                if key:
                    self.keys.setdefault(key, set()).add(row["drug"])
            self.names.setdefault(namekey(row["drug"]), set()).add(row["drug"])

    def match(self, name: str = "", key: str = "", smiles: str = "") -> list[str]:
        """Drugs for one external compound: full key, then SMILES-derived keys, then name."""
        drugs: set[str] = set()
        if key and key in self.keys:
            drugs = self.keys[key]
        if not drugs and smiles:
            for candidate in _inchi_keys(smiles):
                if candidate is not None and candidate in self.keys:
                    drugs = self.keys[candidate]
                    break
        if not drugs and namekey(name) in self.names:
            drugs = self.names[namekey(name)]
        if len(drugs) > 1:
            drugs = {drug for drug in drugs if namekey(drug) == namekey(name)}
        return sorted(drugs)


def read_drug_metadata(path: Path) -> pd.DataFrame:
    """The drug table (``drug``, ``canonical_smiles``, ...) from a .csv or a parquet file."""
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def drug_keys(datasets: Sequence[PreprocessedDir]) -> list[str]:
    """Sorted union of the perturbation keys of every drug dataset."""
    keys = sorted({key for d in datasets if d.meta.pert_kind == "drug" for key in d.perts})
    if not keys:
        raise ValueError("no drug dataset among --datasets (pert_kind 'drug' is required)")
    return keys


def drug_identities(drugs: Sequence[str], drug_metadata: pd.DataFrame) -> list[dict[str, Any]]:
    """Per drug: its canonical SMILES and the InChIKey pair derived from it (None without one)."""
    metadata = drug_metadata.fillna("")
    if not {"drug", "canonical_smiles"} <= set(metadata.columns):
        raise ValueError("drug metadata needs drug and canonical_smiles columns")
    metadata = metadata.assign(drug_key=metadata["drug"].astype(str).str.strip())
    if metadata["drug_key"].duplicated().any():
        raise ValueError("duplicate drug names in the drug metadata")
    metadata = metadata.set_index("drug_key")
    identities: list[dict[str, Any]] = []
    for drug in drugs:
        if drug.strip() not in metadata.index:
            raise ValueError(f"missing drug metadata for {drug!r}")
        smiles = metadata.loc[drug.strip(), "canonical_smiles"]
        inchi_key, parent_key = _inchi_keys(smiles) if smiles else (None, None)
        identities.append(
            {
                "drug": drug,
                "canonical_smiles": smiles,
                "inchi_key": inchi_key,
                "parent_inchi_key": parent_key,
            }
        )
    return identities


def fetch_profile_inputs(
    name: str, cache_dir: Path, session: requests.Session | None = None, offline: bool = False
) -> dict[str, Path]:
    """Download (or verify in the cache) the pinned input files of one profile."""
    return {
        file_name: download_file(
            entry["url"],
            Path(cache_dir) / file_name,
            session=session,
            sha256=entry["sha256"],
            offline=offline,
        )
        for file_name, entry in RELEASES.items()
        if entry["source"] == name
    }


# --- selection rules --------------------------------------------------------------------------


def _nearest_dose_rows(doses: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
    if not np.isfinite(query) or query <= 0:
        raise ValueError("query dose must be finite and positive")
    valid = np.isfinite(values) & np.isfinite(doses) & (doses > 0)
    if not valid.any():
        return np.array([], dtype=int)
    distance = np.full(len(doses), np.inf)
    distance[valid] = np.abs(np.log10(doses[valid] / query))
    tied = valid & np.isclose(distance, distance.min(), rtol=0, atol=1e-10)
    lower = np.min(doses[tied])
    return np.flatnonzero(tied & np.isclose(doses, lower, rtol=1e-8, atol=0))


def _prefer_redo(values: np.ndarray, screens: np.ndarray, cell_index: int) -> np.ndarray:
    valid = np.isfinite(values[:, cell_index])
    redo = valid & (screens == "MTS010")
    return redo if redo.any() else valid


def _build_l1000(
    raw: Mapping[str, Path], perturbations: Sequence[Mapping[str, Any]], matcher: CompoundMatcher
) -> tuple[np.ndarray, list[str]]:
    parts = []
    cells: set[str] = set()
    for series in _L1000_SUFFIX:
        pert_info = pd.read_csv(raw[l1000_file(series, "pert_info")], sep="\t").fillna("")
        mapping: dict[str, list[str]] = {}
        for _, row in pert_info[pert_info.pert_type == "trt_cp"].iterrows():
            drugs = matcher.match(
                row.pert_iname, row.get("inchi_key", ""), row.get("canonical_smiles", "")
            )
            if drugs:
                mapping[row.pert_id] = drugs
        info = pd.read_csv(raw[l1000_file(series, "sig_info")], sep="\t", low_memory=False)
        info = info[info.pert_type == "trt_cp"]
        cells.update(info.cell_id.dropna().unique())
        info = info[info.pert_id.isin(list(mapping))].copy()
        info["drug"] = info.pert_id.map(mapping)
        info = info.explode("drug")
        metrics = pd.read_csv(
            raw[l1000_file(series, "sig_metrics")], sep="\t", usecols=["sig_id", "tas"]
        )
        info = info.merge(metrics, on="sig_id", validate="many_to_one")
        info["series"] = series
        parts.append(info)
    cell_list = sorted(cells)
    if len(cell_list) != _L1000_CELLS:
        raise ValueError(
            f"pinned L1000 releases must have {_L1000_CELLS} chemical cell contexts, "
            f"found {len(cell_list)}"
        )
    signatures = pd.concat(parts, ignore_index=True).drop_duplicates(["sig_id", "drug"])
    number = pd.to_numeric(
        signatures.pert_idose.str.extract(r"^([0-9.eE+-]+)")[0], errors="coerce"
    )
    units = (
        signatures.pert_idose.str.extract(r"^[0-9.eE+-]+\s*(.*)$")[0]
        .str.lower()
        .str.replace("\u00b5", "u")
        .str.replace("\u03bc", "u")
    )
    signatures["dose"] = number * units.map({"um": 1.0, "nm": 0.001, "mm": 1000.0, "m": 1e6})
    signatures["hours"] = pd.to_numeric(
        signatures.pert_itime.str.extract(r"^([0-9.eE+-]+)")[0], errors="coerce"
    )
    signatures["tas"] = signatures.tas.replace(-666, np.nan)
    signatures = signatures[
        (signatures.dose > 0) & (signatures.hours > 0) & np.isfinite(signatures.tas)
    ].copy()
    groups = {key: rows for key, rows in signatures.groupby(["drug", "cell_id"])}
    x = np.full((len(perturbations), len(cell_list)), np.nan)
    for i, pert in enumerate(perturbations):
        for j, cell in enumerate(cell_list):
            rows = groups.get((pert["drug"], cell))
            if rows is None:
                continue
            distance = np.abs(np.log10(rows.dose / pert["dose_uM"])) + np.abs(
                np.log2(rows.hours / L1000_TIME_HOURS)
            )
            hit = rows[np.isclose(distance, distance.min(), atol=1e-10, rtol=0)]
            x[i, j] = hit.tas.median()
    return x, ["TAS:" + cell for cell in cell_list]


def _build_prism(
    raw: Mapping[str, Path], perturbations: Sequence[Mapping[str, Any]], matcher: CompoundMatcher
) -> tuple[np.ndarray, list[str]]:
    info = pd.read_csv(raw["secondary-screen-replicate-collapsed-treatment-info.csv"]).fillna("")
    mapping: dict[str, list[str]] = {}
    for (name, smiles), rows in info.groupby(["name", "smiles"]):
        for drug in matcher.match(name, smiles=smiles):
            mapping.setdefault(drug, []).extend(rows.column_name.tolist())
    matrix = pd.read_csv(
        raw["secondary-screen-replicate-collapsed-logfold-change.csv"], index_col=0
    )
    qc = pd.read_csv(raw["secondary-screen-cell-line-info.csv"])
    qc = qc[qc.passed_str_profiling.astype(str).str.upper() == "TRUE"]
    matrix = matrix.loc[matrix.index.isin(set(qc.row_name))]
    row_to_model = dict(zip(qc.row_name, qc.depmap_id, strict=True))
    matrix.index = pd.Index([row_to_model[row] for row in matrix.index])
    if not matrix.index.is_unique or matrix.index.isna().any():
        raise ValueError("PRISM cell identifiers must be unique and non-missing")
    matrix = matrix.loc[np.isfinite(matrix.to_numpy(float)).any(axis=1)].sort_index()
    cells = list(matrix.index)
    if len(cells) != _PRISM_CELLS:
        raise ValueError(
            f"pinned PRISM secondary release must have {_PRISM_CELLS} measured QC-passing lines, "
            f"found {len(cells)}"
        )
    info = info.set_index("column_name")
    x = np.full((len(perturbations), len(cells)), np.nan)
    for i, pert in enumerate(perturbations):
        columns = sorted(set(mapping.get(pert["drug"], [])) & set(matrix.columns))
        if not columns:
            continue
        meta = info.loc[columns]
        doses = meta.dose.to_numpy(float)
        screens = meta.screen_id.to_numpy(str)
        values = matrix[columns].to_numpy(float).T
        for j in range(len(cells)):
            allowed = _prefer_redo(values, screens, j) & np.isfinite(doses) & (doses > 0)
            candidate = np.flatnonzero(allowed)
            if not len(candidate):
                continue
            hits = candidate[
                _nearest_dose_rows(doses[candidate], values[candidate, j], pert["dose_uM"])
            ]
            if not len(hits):
                continue
            x[i, j] = np.median(values[hits, j])
    return x, ["log2_viability:" + str(cell) for cell in cells]


def _build_jump(
    raw: Mapping[str, Path],
    perturbations: Sequence[Mapping[str, Any]],
    drugs: Sequence[str],
    matcher: CompoundMatcher,
) -> tuple[np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    compounds = pd.read_csv(raw["jump_compound.csv.gz"])
    mapping: dict[str, list[str]] = {}
    for _, row in compounds.iterrows():
        found = matcher.match(key=row.Metadata_InChIKey)
        if found:
            mapping[row.Metadata_JCP2022] = found
    parts = []
    for batch in pq.ParquetFile(raw["jump_profiles.parquet"]).iter_batches(batch_size=16384):
        frame = batch.to_pandas()
        parts.append(frame[frame.Metadata_JCP2022.isin(list(mapping))])
    wells = pd.concat(parts, ignore_index=True)
    columns = [column for column in wells if not str(column).startswith("Metadata")]
    if len(columns) != _JUMP_COORDS:
        raise ValueError(
            f"pinned JUMP release must have {_JUMP_COORDS} morphology coordinates, "
            f"found {len(columns)}"
        )
    profiles: dict[str, np.ndarray] = {}
    for drug in drugs:
        compounds_of_drug = [jcp for jcp, matched in mapping.items() if drug in matched]
        rows = wells[wells.Metadata_JCP2022.isin(compounds_of_drug)]
        profiles[drug] = (
            rows[columns].median().to_numpy(float) if len(rows) else np.full(len(columns), np.nan)
        )
    x = np.stack([profiles[pert["drug"]] for pert in perturbations])
    return x, [str(column) for column in columns]


def _export(
    name: str, x: np.ndarray, keys: Sequence[str], n_axes: int
) -> tuple[list[str], np.ndarray]:
    values = np.asarray(x, dtype=np.float32).astype(np.float64)
    if values.shape != (len(keys), n_axes) or np.isinf(values).any():
        raise ValueError(f"{name}: invalid profile matrix {values.shape} or infinite values")
    covered = np.isfinite(values).any(axis=1)
    if name == "jump_morphology" and not np.isfinite(values[covered]).all():
        raise ValueError("partially missing JUMP profiles require an explicit policy")
    filled = np.where(np.isfinite(values), values, 0).astype(np.float32)
    return [keys[int(i)] for i in np.flatnonzero(covered)], np.ascontiguousarray(filled[covered])


def _build(
    name: str,
    drug_keys: Sequence[str],
    drug_metadata: pd.DataFrame,
    cache_dir: Path,
    session: requests.Session | None,
    offline: bool = False,
) -> tuple[list[str], np.ndarray, list[str], dict[str, Path]]:
    if name not in PROFILES:
        raise KeyError(f"unknown chemical profile {name!r}; expected one of {PROFILES}")
    perturbations = [parse_perturbation(key) for key in sorted(drug_keys)]
    if not perturbations:
        raise ValueError("no drug-dose keys to build a profile for")
    drugs = sorted({pert["drug"] for pert in perturbations})
    matcher = CompoundMatcher(drug_identities(drugs, drug_metadata))
    raw = fetch_profile_inputs(name, cache_dir, session, offline=offline)
    if name == "l1000_tas":
        x, axes = _build_l1000(raw, perturbations, matcher)
    elif name == "prism_secondary":
        x, axes = _build_prism(raw, perturbations, matcher)
    else:
        x, axes = _build_jump(raw, perturbations, drugs, matcher)
    keys, values = _export(name, x, [pert["pert"] for pert in perturbations], len(axes))
    return keys, values, axes, raw


def build_profile(
    name: str,
    drug_keys: Sequence[str],
    drug_metadata: pd.DataFrame,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> tuple[list[str], np.ndarray]:
    """(covered drug-dose keys sorted, float32 rows) of one profile; unmeasured values are 0."""
    keys, values, _, _ = _build(name, drug_keys, drug_metadata, cache_dir, session)
    return keys, values


def run_profile(name: str, ctx: RunContext) -> Path:
    """One of l1000_tas, prism_secondary, jump_morphology over the drug datasets of `ctx`."""
    metadata_path = ctx.options.drug_metadata
    if metadata_path is None:
        raise ValueError(
            f"{name} needs options.drug_metadata (a table with drug and canonical_smiles columns)"
        )
    keys, values, axes, raw = _build(
        name,
        drug_keys(ctx.datasets),
        read_drug_metadata(metadata_path),
        ctx.cache_dir / name,
        None,
        offline=ctx.options.offline,
    )
    inputs = {
        file_name: input_record(path, RELEASES[file_name]["url"]) for file_name, path in raw.items()
    }
    inputs["drug_metadata"] = input_record(metadata_path, None)
    params: dict[str, Any] = {"normalization": "none", "fill_value": 0.0, "axes": axes}
    if name == "l1000_tas":
        params["l1000_time_hours"] = L1000_TIME_HOURS
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name=name,
        layout="dense",
        index="pert",
        keys=keys,
        dim=int(values.shape[1]),
        dtype="float32",
        provenance=base_provenance(inputs, params),
    )
    return write_source(ctx.out_root / name, meta, values, overwrite=ctx.overwrite)
