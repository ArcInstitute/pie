# PIE: Generalizing perturbation effects across unseen perturbations, contexts and datasets

PIE is a perturbation effect prediction model that generalizes to unseen biological context, perturbations and combinations of both. It is based on the Perceiver IO architecture (Jaegle et al., 2022), and leverages curated Knowledge Sources, mean of control cells on context query and perturbation response evidence from contexts and perturbations in the train set. 

## Getting Started

This codebase uses `uv` for package management. 

```bash
git clone https://github.com/ArcInstitute/pie.git && cd pie
uv sync --frozen --extra sources --extra process
cp common.sh.example common.sh      # then fill it in; common.sh is gitignored
```

`common.sh` sets `WANDB_ENTITY`, `WANDB_PROJECT`, `PIE_DATA_ROOT`, `PIE_RUNS_ROOT` and
`PIE_CACHE_DIR`. The text-embedding steps also need `OPENAI_API_KEY`; export it in your own
environment, never in `common.sh`.

## Data: Replogle

Download the Replogle-Nadig count matrix (K562, RPE1, Jurkat, HepG2) from [PLACEHOLDER] into `$PIE_DATA_ROOT/replogle/counts/`.

Normalize the counts and compute per-cell-line differential expression (uses a GPU for running `gpudge`):

```bash
uv run pie-process dataset=replogle input="$PIE_DATA_ROOT/replogle/counts/replogle_4cl.h5ad" \
  normalize.output_dir="$PIE_DATA_ROOT/replogle/expression" \
  de.output_dir="$PIE_DATA_ROOT/replogle/de"
```

Build the preprocessed dataset:

```bash
uv run pie-prep dataset=replogle label_format=pie_process \
  labels="$PIE_DATA_ROOT/replogle/de/*.parquet" \
  h5ad="$PIE_DATA_ROOT/replogle/expression/replogle_4cl.h5ad" \
  output_dir="$PIE_DATA_ROOT/replogle/preprocessed"
```

Build the knowledge sources:

You need to download the CRISPRGeneEffect.csv file and point the `pie-sources` tool to it. 

```bash
uv run pie-sources \
  "tools=[esm2,ncbi_text,string_space,depmap_gene_effect,context_text,perturbation_text,gene_text]" \
  "preprocessed_dirs=[$PIE_DATA_ROOT/replogle/preprocessed]" \
  options.depmap_csv=CRISPRGeneEffect.csv options.device=cuda output_root="$PIE_DATA_ROOT/sources"
```

## Train and evaluate: replogle_wdataset

Each fold holds out one cell line and trains on the other three. For the K562 fold:

```bash
uv run pie-train experiment=replogle_wdataset vars.fold=k562
for setting in unseen_ctx unseen_pert unseen_ctx_pert; do
  uv run pie-eval experiment_name=replogle_wdataset/k562 ckpt=best_auprc \
    split_path=data/splits/replogle_wdataset/$setting/k562/test.json row_set=$setting
done
```

The three test settings are unseen cell line (`unseen_ctx`), unseen perturbation (`unseen_pert`)
and both (`unseen_ctx_pert`). Loop over `hepg2`, `jurkat`, `k562` and `rpe1` to run all four
folds. See [AGENTS.md](AGENTS.md) for the cross-dataset experiment, inference and every CLI option.

## License

<!-- TODO -->
