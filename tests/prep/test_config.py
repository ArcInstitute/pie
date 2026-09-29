from __future__ import annotations

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from pie.prep.config import DATASETS, LABEL_FORMATS, compose_prep_config
from pie.utils import CONFIG_DIR

IO = ["labels=/l/*.csv", "h5ad=/h/*.h5ad", "output_dir=/o/x"]
TABLE = {
    "context_from": "column", "context_col": "context", "pert_col": "pert",
    "gene_col": "gene_symbol", "fc_col": "fold_change", "fc_space": "linear", "fdr_col": "fdr",
}  # fmt: skip
# The G1 invocations (context col, pert col, control label, context map, pert format), plus the
# keys the amendment adds. Label rewrites that are no-ops on PIE label tables are listed too.
G1 = {
    "replogle": ("cell_line", "gene", "non-targeting",
                 {"K562": "k562", "RPE1": "rpe1", "hepg2": "hepg2", "jurkat": "jurkat"}, "gene"),
    "tahoe": ("context", "perturbation", "DMSO_TF_0.0uM", None, "drug_dose"),
    "jiang": ("context", "gene", "non-targeting", None, "gene"),
    "orion": ("context", "gene", "non-targeting", None, "gene"),
    "arc_vcc_25": ("context", "target_gene", "non-targeting", None, "gene"),
}  # fmt: skip


def test_config_files_exist() -> None:
    assert (CONFIG_DIR / "prep.yaml").is_file()
    assert sorted(p.stem for p in (CONFIG_DIR / "prep" / "dataset").glob("*.yaml")) == list(
        DATASETS
    )
    formats = sorted(p.stem for p in (CONFIG_DIR / "prep" / "label_format").glob("*.yaml"))
    assert formats == list(LABEL_FORMATS)


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_overlays_reproduce_the_g1_invocations(name: str) -> None:
    cfg = compose_prep_config([f"dataset={name}", *IO, "genes=/g.txt"])
    context_col, pert_col, control, context_map, pert_format = G1[name]
    assert cfg.name == name
    assert cfg.label.model_dump(include=set(TABLE)) == TABLE
    assert (cfg.obs.context_col, cfg.obs.pert_col, cfg.obs.control_label) == (
        context_col,
        pert_col,
        control,
    )
    assert cfg.obs.context_map == context_map
    assert cfg.obs.pert_format == pert_format
    assert cfg.obs.pert_id_col == ("gene_id" if name == "replogle" else None)
    assert (cfg.labels, cfg.h5ad, cfg.output_dir, cfg.genes) == (
        "/l/*.csv",
        "/h/*.h5ad",
        "/o/x",
        "/g.txt",
    )
    assert (cfg.controls_only, cfg.overwrite) == (False, False)


def test_label_rewrites_per_dataset() -> None:
    rewrites = {
        name: compose_prep_config([f"dataset={name}", *IO]).label.model_dump(
            include={"context_case", "context_map", "pert_format"}
        )
        for name in DATASETS
    }
    assert rewrites["replogle"] == {
        "context_case": "lower",
        "context_map": None,
        "pert_format": "plain",
    }
    assert rewrites["tahoe"]["pert_format"] == "drug_dose"
    assert rewrites["arc_vcc_25"]["context_map"] == {
        "competition_train": "ARC_H1", "competition_val": "ARC_H1_VAL",
        "ARC_H1": "ARC_H1", "ARC_H1_VAL": "ARC_H1_VAL",
    }  # fmt: skip
    for name in ("jiang", "orion"):
        assert rewrites[name] == {
            "context_case": "asis",
            "context_map": None,
            "pert_format": "plain",
        }


def test_replogle_context_map_maps_h5ad_lines_onto_the_split_folds() -> None:
    cfg = compose_prep_config(["dataset=replogle", *IO])
    assert sorted((cfg.obs.context_map or {}).values()) == ["hepg2", "jurkat", "k562", "rpe1"]


def test_pie_process_label_format_reads_de_parquets() -> None:
    label = compose_prep_config(["dataset=replogle", "label_format=pie_process", *IO]).label
    assert (label.context_from, label.context_col, label.pert_col, label.gene_col) == (
        "stem",
        None,
        "target",
        "feature",
    )
    assert (label.fc_col, label.fc_space, label.fdr_col) == ("log2_fold_change", "log2", "p_adj")
    assert label.context_case == "lower"  # the dataset rewrite survives the format switch


@pytest.mark.parametrize("name", DATASETS)
def test_overlays_set_only_values_that_differ_from_the_base(name: str) -> None:
    def leaves(node: object, prefix: str = "") -> dict[str, object]:
        if not isinstance(node, dict):
            return {prefix: node}
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out

    base = leaves(OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "prep.yaml")))
    overlay = leaves(
        OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "prep" / "dataset" / f"{name}.yaml"))
    )
    assert (
        sorted(k for k, v in overlay.items() if base.get(k) == v and not isinstance(v, dict)) == []
    )


def test_required_keys_and_strictness() -> None:
    with pytest.raises(MissingMandatoryValue):
        compose_prep_config(["dataset=jiang", "labels=/l", "h5ad=/h"])  # no output_dir
    with pytest.raises(ValidationError, match="labels"):
        compose_prep_config(["dataset=jiang", "h5ad=/h", "output_dir=/o"])
    assert compose_prep_config(["dataset=jiang", "h5ad=/h", "output_dir=/o", "controls_only=true"])
    with pytest.raises(ConfigCompositionException):
        compose_prep_config(["dataset=jiang", *IO, "label.nope=1"])
    with pytest.raises(ValidationError):
        compose_prep_config(["dataset=jiang", *IO, "+obs.nope=1"])
    with pytest.raises(ConfigCompositionException):
        compose_prep_config(["dataset=nope", *IO])
    with pytest.raises(ValidationError, match="pert_id_col"):
        compose_prep_config(["dataset=tahoe", *IO, "obs.pert_id_col=gene_id"])
    with pytest.raises(ValidationError, match="context_col"):
        compose_prep_config(["dataset=jiang", *IO, "label.context_col=null"])


def test_a_non_string_context_map_key_is_rejected() -> None:
    # A context such as `1` (or a YAML-boolean-looking `on` in a YAML file) must not silently
    # become a non-string key: the strict schema refuses it.
    with pytest.raises(ValidationError):
        compose_prep_config(["dataset=jiang", *IO, "obs.context_map={1: k562}"])
