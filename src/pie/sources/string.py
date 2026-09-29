"""STRING SPACE protein-network embeddings, keyed by the STRING preferred gene name."""

from __future__ import annotations

from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from pie.sources.common import base_provenance, download_file, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source

if TYPE_CHECKING:
    import requests

    from pie.sources.registry import RunContext

TAXON = 9606
_BASE_URL = "https://stringdb-downloads.org/download"


def string_urls(release: str) -> dict[str, str]:
    """Download URLs of the SPACE embeddings h5 and the protein.info table of a release."""
    return {
        "space_h5": (
            f"{_BASE_URL}/protein.network.embeddings.{release}/"
            f"{TAXON}.protein.network.embeddings.{release}.h5"
        ),
        "protein_info": (
            f"{_BASE_URL}/protein.info.{release}/{TAXON}.protein.info.{release}.txt.gz"
        ),
    }


def fetch_string(
    release: str, cache_dir: Path, session: requests.Session | None = None, offline: bool = False
) -> dict[str, Path]:
    """Download both files of a release into `cache_dir` under their upstream names."""
    return {
        name: download_file(
            url, Path(cache_dir) / url.rsplit("/", 1)[1], session=session, offline=offline
        )
        for name, url in string_urls(release).items()
    }


def build_string_space(space_h5: Path, protein_info: Path) -> tuple[list[str], np.ndarray]:
    """(keys, float16 rows): the first protein per preferred name (h5 order), sorted by name."""
    with h5py.File(space_h5, "r") as handle:
        embeddings = handle["embeddings"][:]
        proteins = [protein.decode() for protein in handle["proteins"][:]]
    info = pd.read_csv(protein_info, sep="\t", compression="gzip")
    names = dict(zip(info["#string_protein_id"], info["preferred_name"], strict=True))
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    for position, protein in enumerate(proteins):
        name = names.get(protein)
        if name is None:
            continue
        name = str(name)
        if name in seen:
            continue
        seen.add(name)
        rows.append((name, position))
    rows.sort(key=itemgetter(0))
    keys = [name for name, _ in rows]
    selected = embeddings[[position for _, position in rows]]
    return keys, selected.astype(np.float32).astype(np.float16)


def run_string_space(ctx: RunContext) -> Path:
    """string_space: STRING SPACE h5 + protein.info -> preferred-name keyed float16 rows."""
    release = ctx.options.string_release
    files = fetch_string(release, ctx.cache_dir / "string_space", offline=ctx.options.offline)
    keys, embeddings = build_string_space(files["space_h5"], files["protein_info"])
    urls = string_urls(release)
    provenance = base_provenance(
        {name: input_record(path, urls[name], release=release) for name, path in files.items()},
        {
            "taxon": TAXON,
            "key": "preferred_name",
            "repeated_names": "first protein in h5 order",
            "row_order": "sorted by key",
        },
    )
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="string_space",
        layout="dense",
        index="pert",
        keys=keys,
        dim=int(embeddings.shape[1]),
        dtype="float16",
        provenance=provenance,
    )
    return write_source(
        ctx.out_root / "string_space", meta, embeddings, overwrite=ctx.overwrite
    )
