"""pie-sources config: configs/sources.yaml defaults and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError

from pie.sources.config import compose_sources_config
from pie.utils import REPO_ROOT

REQUIRED = ["tools=[context_text]", "preprocessed_dirs=[/p/a]", "output_root=/s"]


def test_defaults_are_the_documented_builder_settings() -> None:
    cfg = compose_sources_config(REQUIRED)
    assert (cfg.tools, cfg.with_deps, cfg.prior_root, cfg.overwrite) == (
        ["context_text"],
        True,
        None,
        False,
    )
    opts = cfg.options
    assert opts.contexts_dir == Path("data/sources/contexts")
    assert opts.resolved().contexts_dir == REPO_ROOT / "data" / "sources" / "contexts"
    assert (opts.on_conflict, opts.cellosaurus_release, opts.offline) == ("error", "56.0", False)
    assert (opts.string_release, opts.device) == ("v12.0", "cuda")
    assert (opts.gene_info, opts.drug_metadata, opts.depmap_csv, opts.pert_output) == (None,) * 4


def test_unknown_tools_keys_and_the_removed_h5ad_option_fail() -> None:
    with pytest.raises(ValidationError, match="nope"):
        compose_sources_config(["tools=[nope]", *REQUIRED[1:]])
    with pytest.raises(ValidationError):
        compose_sources_config(["tools=[]", *REQUIRED[1:]])
    with pytest.raises(ConfigCompositionException):
        compose_sources_config([*REQUIRED, "options.h5ad={replogle: x}"])
    with pytest.raises(ValidationError):
        compose_sources_config([*REQUIRED, "+options.h5ad={replogle: x}"])
