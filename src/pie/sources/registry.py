"""The pie-sources tool registry: tool entries, dependency order and the run driver."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pie.assets import resolve_asset
from pie.data.preprocessed import PreprocessedDir
from pie.sources.config import SourceOptions
from pie.utils import resolve_path

if TYPE_CHECKING:
    from pie.sources.config import SourcesConfig

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    datasets: list[PreprocessedDir]
    prior_root: Path | None
    out_root: Path
    cache_dir: Path
    options: SourceOptions
    overwrite: bool = False


@dataclass(frozen=True)
class SourceTool:
    name: str
    deps: tuple[str, ...]
    uses_openai: bool
    run: Callable[[RunContext], Path]


def _run_context_text(ctx: RunContext) -> Path:
    from pie.sources.text.tools import run_context_text

    return run_context_text(ctx)


def _run_perturbation_text(ctx: RunContext) -> Path:
    from pie.sources.text.tools import run_perturbation_text

    return run_perturbation_text(ctx)


def _run_gene_text(ctx: RunContext) -> Path:
    from pie.sources.text.tools import run_gene_text

    return run_gene_text(ctx)


TOOLS: dict[str, SourceTool] = {
    "context_text": SourceTool("context_text", (), True, _run_context_text),
    "perturbation_text": SourceTool("perturbation_text", (), True, _run_perturbation_text),
    "gene_text": SourceTool("gene_text", ("perturbation_text",), True, _run_gene_text),
}


def resolve_order(names: list[str]) -> list[str]:
    """Add missing deps, then order topologically, taking ready tools in TOOLS order."""
    unknown = [name for name in names if name not in TOOLS]
    if unknown:
        raise KeyError(f"unknown source tool(s) {unknown}; known: {list(TOOLS)}")
    selected: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in selected:
            selected.add(name)
            pending.extend(TOOLS[name].deps)
    order: list[str] = []
    while len(order) < len(selected):
        ready = [
            n for n in TOOLS if n in selected and n not in order
            and all(dep in order for dep in TOOLS[n].deps)
        ]
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(selected - set(order))}")
        order.append(ready[0])
    return order


def select_tools(names: list[str], with_deps: bool) -> list[str]:
    """Dependency-ordered names (with_deps) or exactly the named tools, in TOOLS order."""
    if with_deps:
        return resolve_order(names)
    unknown = [name for name in names if name not in TOOLS]
    if unknown:
        raise KeyError(f"unknown source tool(s) {unknown}; known: {list(TOOLS)}")
    return [name for name in TOOLS if name in names]


def run_sources(names: list[str], ctx: RunContext) -> dict[str, Path]:
    """Check OPENAI_API_KEY first if any selected tool embeds with OpenAI, then run in order."""
    order = resolve_order(names)
    if any(TOOLS[name].uses_openai for name in order):
        from pie.sources.embed import openai as openai_embed

        openai_embed.require_openai_key()
    outputs: dict[str, Path] = {}
    for name in order:
        log.info("building source %s", name)
        outputs[name] = TOOLS[name].run(ctx)
    return outputs


def _occupied(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def build_sources(cfg: SourcesConfig) -> dict[str, Path]:
    """Check the cache root, every target and OPENAI_API_KEY first, then build in order."""
    names = select_tools(cfg.tools, cfg.with_deps)
    cache_dir = resolve_path("${PIE_CACHE_DIR}") / "http"
    out_root = resolve_path(cfg.output_root)
    busy = [str(out_root / name) for name in names if _occupied(out_root / name)]
    if busy and not cfg.overwrite:
        raise FileExistsError(f"{', '.join(busy)} exist; set overwrite=true to replace them")
    if any(TOOLS[name].uses_openai for name in names):
        from pie.sources.embed import openai as openai_embed

        openai_embed.require_openai_key()
    ctx = RunContext(
        datasets=[
            PreprocessedDir.open(resolve_asset(p, kind="preprocessed"))
            for p in cfg.preprocessed_dirs
        ],
        prior_root=resolve_path(cfg.prior_root) if cfg.prior_root else None,
        out_root=out_root,
        cache_dir=cache_dir,
        options=cfg.options.resolved(),
        overwrite=cfg.overwrite,
    )
    outputs: dict[str, Path] = {}
    for name in names:
        log.info("building source %s", name)
        outputs[name] = TOOLS[name].run(ctx)
    return outputs


# --- gene and chemical source tools (imports are lazy: the registry pulls no optional deps) ---


def _run_ncbi_text(ctx: RunContext) -> Path:
    from pie.sources.text.ncbi import run_ncbi_text

    return run_ncbi_text(ctx)


def _run_esm2(ctx: RunContext) -> Path:
    from pie.sources.embed.esm2 import run_esm2

    return run_esm2(ctx)


def _run_string_space(ctx: RunContext) -> Path:
    from pie.sources.string import run_string_space

    return run_string_space(ctx)


def _run_depmap_gene_effect(ctx: RunContext) -> Path:
    from pie.sources.depmap import run_depmap

    return run_depmap(ctx)


def _run_smiles(ctx: RunContext) -> Path:
    from pie.sources.embed.chemberta import run_smiles

    return run_smiles(ctx)


def _run_l1000_tas(ctx: RunContext) -> Path:
    from pie.sources.chem_profiles import run_profile

    return run_profile("l1000_tas", ctx)


def _run_prism_secondary(ctx: RunContext) -> Path:
    from pie.sources.chem_profiles import run_profile

    return run_profile("prism_secondary", ctx)


def _run_jump_morphology(ctx: RunContext) -> Path:
    from pie.sources.chem_profiles import run_profile

    return run_profile("jump_morphology", ctx)


TOOLS.update(
    {
        "ncbi_text": SourceTool(
            name="ncbi_text", deps=(), uses_openai=False, run=_run_ncbi_text
        ),
        "esm2": SourceTool(name="esm2", deps=(), uses_openai=False, run=_run_esm2),
        "string_space": SourceTool(
            name="string_space", deps=(), uses_openai=False, run=_run_string_space
        ),
        "depmap_gene_effect": SourceTool(
            name="depmap_gene_effect", deps=(), uses_openai=False, run=_run_depmap_gene_effect
        ),
        "smiles": SourceTool(name="smiles", deps=(), uses_openai=False, run=_run_smiles),
        "l1000_tas": SourceTool(
            name="l1000_tas", deps=(), uses_openai=False, run=_run_l1000_tas
        ),
        "prism_secondary": SourceTool(
            name="prism_secondary", deps=(), uses_openai=False, run=_run_prism_secondary
        ),
        "jump_morphology": SourceTool(
            name="jump_morphology", deps=(), uses_openai=False, run=_run_jump_morphology
        ),
    }
)


def verify_sources(
    sources: dict[str, Path],
    preprocessed: list[PreprocessedDir],
    aliases_path: Path | None,
    reference_root: Path | None,
) -> dict[str, object]:
    """Coverage of every dataset key per source (after aliases) and, with a reference, diffs."""
    from pie.sources.verify import verify_sources as verify

    return verify(sources, preprocessed, aliases_path, reference_root)


def verify_command(cfg: SourcesConfig) -> dict[str, object]:
    """mode=verify: verify_sources over <output_root>/<name> for exactly the named tools."""
    out_root = resolve_path(cfg.output_root)
    aliases, reference = cfg.verify.aliases, cfg.verify.reference
    return verify_sources(
        {name: out_root / name for name in cfg.tools},
        [
            PreprocessedDir.open(resolve_asset(p, kind="preprocessed"))
            for p in cfg.preprocessed_dirs
        ],
        resolve_path(aliases) if aliases else None,
        resolve_path(reference) if reference else None,
    )
