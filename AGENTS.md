# PIE

PIE predicts how a perturbation changes gene expression in a cell context. For each (context,
perturbation) pair and every measured gene it outputs the probability of differential expression
(`p_de`), the log2 fold change (`lfc_pred`) and the expression-level shift (`delta_p_pred`), from
knowledge-source embeddings of the perturbation, context and genes plus pooled training evidence.

## Repository map

- `src/pie/configs/`: Hydra configs shipped in the package, one per command (`process.yaml`, `prep.yaml`, `sources.yaml`, `train.yaml`,
  `eval.yaml`, `infer.yaml`), overlays in `process/dataset/`, `prep/dataset/`, `prep/label_format/`
  and `experiment/` (the two canonical experiments).
- `src/pie/cli.py`: the `pie <command>` entry point (`process`, `prep`, `sources`, `train`, `eval`,
  `infer`); the old `pie-<command>` scripts remain as deprecated aliases.
- On HF, not in the repo: split files in `arcinstitute/PIE_splits`
  (`{"<dataset>.<context>": [perturbation, ...]}`), each dataset's context map in
  `preprocessed/contexts.yaml` (context to Cellosaurus accession, Jiang stimulations) and
  per-source perturbation aliases in `PIE_sources/<name>/aliases.yaml` (used only when a key is
  missing). `src/pie/sources/curated_aliases.yaml` is the reviewed table: `pie sources` writes
  each source's entries into its output dir, and runs saved before aliases moved read it.
- `src/pie/config.py`: strict schema of train/eval/infer (unknown keys are errors); `utils.py`:
  `common.sh` loading, env checks, config composition, logging, determinism, hashing, atomic writes.
- `src/pie/assets.py`: local/HF asset resolver, revision pinning, selective downloads,
  repository locks and completion manifests under `PIE_DATA_ROOT/hf/`.
- `src/pie/process/`: raw count h5ads to log1p expression h5ads and per-context DE parquets.
- `src/pie/prep/`: labels and expression h5ads to a preprocessed dir (`config.py` is its schema).
- `src/pie/sources/`: source format (`contract.py`), tool registry (`registry.py`), config schema
  (`config.py`), text renderers (`text/`), embedders (`embed/`) and the UniProt, STRING, DepMap and
  chemical-profile builders.
- `src/pie/data/`: preprocessed reader, splits, delta-p grid, evidence cache, samplers, datamodule.
- `src/pie/model/`: `PieModel` (Perceiver trunk, evidence module, heads) and the losses.
- `src/pie/metrics.py` (the scorer for validation and evaluation), `train.py` (Lightning module,
  trainer, checkpoints), `predict.py` (shared by `evaluate.py` and `infer.py`).
- `tests/`: mirrors `src/pie/`.

## Setup

```bash
uv sync --frozen                    # runtime and dev tools
uv sync --frozen --extra sources    # also the source-building tools
uv sync --frozen --extra process    # also the dataset-processing DE stage
cp common.sh.example common.sh      # then fill it in; common.sh is gitignored
```

`common.sh` sets `WANDB_ENTITY`, `WANDB_PROJECT`, `PIE_DATA_ROOT`, `PIE_RUNS_ROOT`, `PIE_CACHE_DIR`
and, optionally, `NCBI_API_KEY` and `NCBI_EMAIL`. Values are literal (no `$VAR` or `~` expansion, no
inline comments). Every command loads the first of `$PIE_ENV_FILE`, `./common.sh` and
`~/.config/pie/common.sh`; variables already set in the environment win,
and a CLI stops at once when a variable it needs is missing. The text-embedding steps also need
`OPENAI_API_KEY`: export it in your own environment, never in `common.sh`.

## CLI tools

Every command takes Hydra-style `key=value` overrides on its config in `src/pie/configs/`
(`pie process` → `process.yaml`, `pie prep` → `prep.yaml`, `pie sources` → `sources.yaml`,
`pie train` → `train.yaml`, `pie eval` → `eval.yaml`, `pie infer` → `infer.yaml`); `pie` lists the
commands and `pie <command> --help` prints the usage. Overlays are
picked with `dataset=<name>`, `label_format=<name>` or `experiment=<name>`, unknown keys are errors,
relative paths are relative to the current directory, and an existing output is an error unless you pass
`overwrite=true`.

Preprocessed dataset inputs (`data.preprocessed_dirs` in train; `preprocessed_dirs` in
eval/infer/sources) and model source inputs (`data.source_dirs`, `data.gene_text_dir`) accept
local paths or `hf://datasets/<owner>/<repo>[@revision]/<directory>` references. Bare repo IDs
are local paths; use the explicit URI for HF. The canonical experiment overlays pin full HF
commits for every dataset and source, including gene text. Dataset repos use `preprocessed/`;
`arcinstitute/PIE_sources` uses one directory per source name.

HF assets download only the selected directory to
`$PIE_DATA_ROOT/hf/datasets/<owner>/<repo>/<commit>/<directory>/`; download metadata, locks,
revision records, completion manifests and the Xet transfer cache also stay under this root.
Do not resolve HF references through `resolve_path` or turn them into `Path` before resolution:
use `resolve_asset(value, kind="preprocessed"|"source")`. Config composition and checkpoint
loading must stay free of network/download side effects. Training pins remote references before
saving config/checkpoints; resume replays the previous run's commits. An unpinned revision resolves
once per data root; use a new explicit commit for an update. Fully downloaded assets need no
network. `HF_HUB_OFFLINE=1` refuses missing/incomplete assets. Authenticate when needed with
`uv run hf auth login` or `HF_TOKEN` in the environment; never store tokens in configs/common.sh.

`pie process` turns raw count h5ads into the log1p expression h5ads and per-context DE parquets
that `pie prep` consumes: `dataset=<name>` selects a per-dataset overlay enabling the stages it
needs among `filter` (on-target knockdown removal), `normalize` (CP10k plus natural log1p) and `de`
(needs a GPU and the `process` extra).

`pie prep` builds one preprocessed dir per dataset from DE label tables and expression h5ads: give
`labels` (glob), `h5ad` (glob) and `output_dir`. `dataset=<name>` sets the h5ad obs columns (`obs.*`)
and the label rewrites; `label_format=table` (default) reads PIE label tables and
`label_format=pie_process` reads `pie process` DE parquets. `obs.pert_id_col` records each
perturbation's Ensembl id in `meta.json` (Replogle: `gene_id`), which `pie sources` uses. A label
context without h5ad or control cells is an error; `genes=<file>` fixes the gene axis;
`controls_only=true` writes only the gene axis, vocab and control means (inference on new
contexts). Without `genes` the derived axis can differ from a canonical one (Tahoe: 17845 vs. 18151
genes), so reproducing a canonical dir needs the gene list shipped with it. `contexts=<file>`
copies a context map into the dir as `contexts.yaml`; `pie sources` needs it for `context_text`.

```bash
uv run pie prep dataset=replogle label_format=pie_process \
  labels="$PIE_DATA_ROOT/replogle/de/*.parquet" \
  h5ad="$PIE_DATA_ROOT/replogle/expression/*.h5ad" \
  output_dir="$PIE_DATA_ROOT/replogle/preprocessed"
```

`pie sources` builds knowledge sources into `<output_root>/<name>/` for the datasets in
`preprocessed_dirs`. `tools=[...]` adds and orders dependencies (`with_deps=false` builds only the
named tools); builder settings live under `options.*`; `mode=verify` reports per-dataset coverage
and, with `verify.reference=<dir>`, the difference to published sources. `prior_root` extends
earlier text sources instead of re-embedding them. `depmap_gene_effect` needs a manual download
(`options.depmap_csv`), the chemical sources need `options.drug_metadata`, and `ncbi_text` and
`esm2` need a GPU (`options.device`, default `cuda`). Context text reads
`<preprocessed dir>/contexts.yaml`; aliases come from `<source dir>/aliases.yaml`, plus an optional
multi-source file that you pass in the config (`verify.aliases`, `data.aliases_path`), loaded as
given.

```bash
uv run pie sources \
  "tools=[esm2,ncbi_text,string_space,depmap_gene_effect,context_text,perturbation_text,gene_text]" \
  "preprocessed_dirs=[$PIE_DATA_ROOT/replogle/preprocessed]" \
  options.depmap_csv=CRISPRGeneEffect.csv output_root="$PIE_DATA_ROOT/sources"
```

`pie train` trains from `train.yaml` plus Hydra overrides (for example
`experiment=replogle_wdataset vars.fold=k562`). At setup it fits the delta-p grid on the training
rows, loads or builds the evidence cache and writes `data_stats.json`. A non-empty run dir is an
error unless you pass `resume=true` (continue from `last.ckpt`) or `overwrite=true`.
`logger.enabled=false` trains without wandb.

`pie eval` scores a checkpoint on a split file and writes `<run_dir>/eval/<row_set>/metrics_<ckpt>.csv`
(one row per context plus an `all` row) and `granular_<ckpt>.csv` (one row per pair);
`save_predictions=true` also writes the predictions.

`pie infer` predicts without labels and writes one parquet row per pair (`p_de`, `lfc_pred`,
`delta_p_pred`; the gene axis is in the file metadata). For new contexts, build a
`controls_only=true` dir with `pie prep` and pass a query file in split format.

```bash
uv run pie infer experiment_name=replogle_wdataset/k562 ckpt=best_auprc \
  rows_kind=query rows_path=query.json \
  "preprocessed_dirs=[$PIE_DATA_ROOT/my_screen/preprocessed]" output_path=predictions.parquet
```

## End-to-end flow

raw count h5ads → `pie process` (optional: knockdown filter, log1p normalize, DE) → labels + h5ad
→ `pie prep` → `pie sources tools=[…]` → `pie train` (grid and evidence built at setup)
→ `pie eval` / `pie infer`.

```
$PIE_DATA_ROOT/<dataset>/counts/         count h5ads (knockdown-filtered when pie process filters)
$PIE_DATA_ROOT/<dataset>/expression/     log1p expression h5ads
$PIE_DATA_ROOT/<dataset>/de/             per-context DE parquets
$PIE_DATA_ROOT/<dataset>/preprocessed/   meta.json and the .npy arrays
$PIE_DATA_ROOT/sources/<name>/           meta.json, embeddings.npy [offsets.npy, descriptions.json]
$PIE_RUNS_ROOT/<experiment_name>/        config.yaml, data_stats.json, best_auprc.ckpt, last.ckpt,
                                         eval/<row_set>/
$PIE_CACHE_DIR/evidence/<key>/           evidence, keyed by the data, train split and settings
$PIE_CACHE_DIR/http/, embed/             downloads and resumable embedding progress
```

## Experiments

Evaluate `best_auprc.ckpt` (maximum `val/binary_auprc`) on every test set.
Both recipes fetch their pinned preprocessed datasets and knowledge sources automatically at
runtime; `uv sync --frozen` is sufficient to use them. For distributed runs, a shared
`PIE_DATA_ROOT` lets repository locks coordinate downloads across ranks and nodes. With local
data roots, each node downloads its own assets. Run directories must still use shared storage.

**replogle_wdataset**: Replogle only, four folds (hepg2, jurkat, k562, rpe1). A fold trains and
validates on the other three cell lines and is tested in three settings: `unseen_ctx` (held-out
line, seen perturbations), `unseen_pert` (seen lines, unseen perturbations) and `unseen_ctx_pert`.
Hardware: 2 GPUs per fold (H100 80GB: about 1.2 h and 20 GiB per GPU); eval on 1 GPU.

```bash
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
for fold in hepg2 jurkat k562 rpe1; do
  uv run pie train experiment=replogle_wdataset vars.fold=$fold
  for setting in unseen_ctx unseen_pert unseen_ctx_pert; do
    uv run pie eval experiment_name=replogle_wdataset/$fold ckpt=best_auprc \
      split_path=$SPLITS/replogle_wdataset/$setting/$fold/test.json row_set=$setting
  done
done
```

**replogle_xdataset**: trains on Tahoe, Jiang, ARC VCC 25 and Orion (weights 0.68, 0.01, 0.01, 0.3)
and is tested zero-shot on Replogle, which stays in the config with weight 0 to fix the gene axis.
`test_seen` holds the pairs whose perturbation is in `train.json`, `test_unseen` the rest; the
per-context rows of each metrics file are the per-cell-line results. Hardware: 2 nodes × 4 GPUs
(H100 80GB: about 4.5 h, peak 78.5 GiB per GPU, about 46 GiB host RAM per rank), a run dir on
shared storage; eval on 1 GPU. Run `torchrun` on each node with its rank and one fresh `RDZV_ID`
shared by both nodes:

```bash
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
uv run torchrun --nnodes=2 --nproc-per-node=4 --node-rank=<0|1> \
  --master-addr=<node-0 host> --master-port=29500 --rdzv-id="$RDZV_ID" \
  --no-python pie train experiment=replogle_xdataset
for rows in test_seen test_unseen; do
  uv run pie eval experiment_name=replogle_xdataset ckpt=best_auprc \
    split_path=$SPLITS/replogle_xdataset/$rows.json row_set=$rows
done
```

## Conventions

- `tests/` mirrors `src/pie/`; tests run on CPU with synthetic fixtures: `uv run pytest -q`.
- Lint with `uv run ruff check .` (100 columns); type-check with `uv run mypy src`.
- Direct dependencies use version ranges in `pyproject.toml` (lower bound = the tested version);
  `uv.lock` holds the exact versions and is committed (`uv lock`).
- Configs are strict: each value lives once in YAML, and a new key needs a schema field.
- Never commit machine-specific paths, `common.sh`, API keys or other secrets.
