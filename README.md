# PIE: Generalizing perturbation effects across unseen perturbations, contexts and datasets

PIE is a perturbation effect prediction model that generalizes to unseen biological context, perturbations and combinations of both. It is based on the Perceiver IO architecture (Jaegle et al., 2022), and leverages curated Knowledge Sources, mean of control cells on context query and perturbation response evidence from contexts and perturbations in the train set. 

Preprint available at: [biorxiv](https://doi.org/10.64898/2026.10.02.756297)\
Data and Knowledge Sources (🤗 Hugging Face): [PIE collection](https://huggingface.co/collections/arcinstitute/pie)

## Installation

### From PyPI

```bash
pip install arc-pie                 # or: uv tool install arc-pie
pip install "arc-pie[sources]"      # also the knowledge-source building tools
pip install "arc-pie[process]"      # also the GPU differential-expression stage (gpudge)
```

### From source (reproduces the paper environment exactly)

```bash
git clone https://github.com/ArcInstitute/pie.git && cd pie
uv sync --frozen                    # add --extra sources / --extra process as needed
```

From source, prefix every command with `uv run` (for example `uv run pie train ...`).

### Settings

PIE reads `WANDB_ENTITY`, `WANDB_PROJECT`, `PIE_DATA_ROOT`, `PIE_RUNS_ROOT` and `PIE_CACHE_DIR`
from the environment, else from the first of `$PIE_ENV_FILE`, `./common.sh` and
`~/.config/pie/common.sh`. Copy `common.sh.example` from the repository and fill it in.
Set `logger.enabled=false` to train without wandb. Building text embeddings also needs
`OPENAI_API_KEY`; export it in your own environment, never in `common.sh`.
Run `pie` to list the commands and `pie <command> --help` for each one.

## Published datasets and knowledge sources

- **Ready to run:** the experiment configs pin full HF commits of every dataset, knowledge
  source and split, and PIE downloads what a run needs. Rebuilding data is optional.
- **On Hugging Face:** preprocessed datasets (with their `contexts.yaml`), `PIE_sources` (with
  per-source `aliases.yaml`) and `PIE_splits` (train/val/test files), in the
  [PIE collection](https://huggingface.co/collections/arcinstitute/pie).
- **References:** `hf://datasets/<owner>/<repo>[@revision]/<directory>`; mix them freely with
  local paths. Runs and checkpoints record the resolved commit.
- **Cache:** downloads go to `$PIE_DATA_ROOT/hf/`; `HF_HUB_OFFLINE=1` reuses them without
  network. Use `hf auth login` or `HF_TOKEN` if a repository needs a login.

## Optional: build Replogle data and sources locally

Install the source-building and dataset-processing tools:

```bash
uv sync --frozen --extra sources --extra process
```

Download the Replogle-Nadig count matrix (K562, RPE1, Jurkat, HepG2) from [PLACEHOLDER] into `$PIE_DATA_ROOT/replogle/counts/`.

Normalize the counts and compute per-cell-line differential expression (uses a GPU for running `gpudge`):

```bash
uv run pie process dataset=replogle input="$PIE_DATA_ROOT/replogle/counts/replogle_4cl.h5ad" \
  normalize.output_dir="$PIE_DATA_ROOT/replogle/expression" \
  de.output_dir="$PIE_DATA_ROOT/replogle/de"
```

Build the preprocessed dataset. `contexts=` copies the context map (context to Cellosaurus
accession) into the dir; `pie sources` needs it for `context_text`. The published map is in the
dataset repo:

```bash
uv run hf download arcinstitute/PIE_replogle_nadig_essential preprocessed/contexts.yaml \
  --repo-type dataset --local-dir .
uv run pie prep dataset=replogle label_format=pie_process \
  labels="$PIE_DATA_ROOT/replogle/de/*.parquet" \
  h5ad="$PIE_DATA_ROOT/replogle/expression/replogle_4cl.h5ad" \
  contexts=preprocessed/contexts.yaml output_dir="$PIE_DATA_ROOT/replogle/preprocessed"
```

Build the knowledge sources:

You need to download the CRISPRGeneEffect.csv file and point the `pie sources` tool to it. 

```bash
uv run pie sources \
  "tools=[esm2,ncbi_text,string_space,depmap_gene_effect,context_text,perturbation_text,gene_text]" \
  "preprocessed_dirs=[$PIE_DATA_ROOT/replogle/preprocessed]" \
  options.depmap_csv=CRISPRGeneEffect.csv options.device=cuda output_root="$PIE_DATA_ROOT/sources"
```

To use these local assets for a canonical experiment, override `data.preprocessed_dirs`,
`data.source_dirs`, and `data.gene_text_dir`; the published recipes select HF assets by default.

## Train and evaluate: replogle_wdataset

Each fold holds out one cell line and trains on the other three. For the K562 fold:

```bash
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
uv run pie train experiment=replogle_wdataset vars.fold=k562
for setting in unseen_ctx unseen_pert unseen_ctx_pert; do
  uv run pie eval experiment_name=replogle_wdataset/k562 ckpt=best_auprc \
    split_path=$SPLITS/replogle_wdataset/$setting/k562/test.json row_set=$setting
done
```

The three test settings are unseen cell line (`unseen_ctx`), unseen perturbation (`unseen_pert`)
and both (`unseen_ctx_pert`). Loop over `hepg2`, `jurkat`, `k562` and `rpe1` to run all four
folds. See [AGENTS.md](AGENTS.md) for the cross-dataset experiment, inference and every CLI option.

## Train and evaluate: replogle_xdataset

This experiment trains on Tahoe, Jiang, ARC VCC 25, and Orion and evaluates zero-shot on Replogle.
It uses two nodes with four GPUs each and a run directory on shared storage. Run the following
on each node with its rank, a shared rendezvous ID, and the node-0 hostname:

```bash
uv run torchrun --nnodes=2 --nproc-per-node=4 --node-rank=<0|1> \
  --master-addr=<node-0 host> --master-port=29500 --rdzv-id="$RDZV_ID" \
  --no-python pie train experiment=replogle_xdataset
```

Evaluate the best checkpoint on one GPU:

```bash
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
for rows in test_seen test_unseen; do
  uv run pie eval experiment_name=replogle_xdataset ckpt=best_auprc \
    split_path=$SPLITS/replogle_xdataset/$rows.json row_set=$rows
done
```

## License

PIE is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license (CC BY-NC-SA 4.0).
You may use, share and adapt it for non-commercial purposes with attribution, and adaptations
must be shared under the same license. See [LICENSE.md](LICENSE.md) for the full text.
