# PIE: Generalizing perturbation effects across unseen perturbations, contexts and datasets

PIE is a perturbation effect prediction model that generalizes to unseen biological context, perturbations and combinations of both. It is based on the Perceiver IO architecture (Jaegle et al., 2022), and leverages curated Knowledge Sources, mean of control cells on context query and perturbation response evidence from contexts and perturbations in the train set. 

Data and Knowledge Sources (🤗 Hugging Face): [PIE collection](https://huggingface.co/collections/arcinstitute/pie)

## Getting Started

This codebase uses `uv` for package management. 

```bash
git clone https://github.com/ArcInstitute/pie.git && cd pie
uv sync --frozen
cp common.sh.example common.sh      # then fill it in; common.sh is gitignored
```

`common.sh` sets `WANDB_ENTITY`, `WANDB_PROJECT`, `PIE_DATA_ROOT`, `PIE_RUNS_ROOT` and
`PIE_CACHE_DIR`. These directories must be writable. Set `logger.enabled=false` to train without
wandb. Building text embeddings also needs `OPENAI_API_KEY`; export it in your own environment,
never in `common.sh`.

## Published datasets and knowledge sources

The [PIE Hugging Face collection](https://huggingface.co/collections/arcinstitute/pie) contains
the preprocessed datasets and knowledge sources used by both canonical experiments. Their
configs pin full HF commits, including `gene_text`, so you can launch the training commands below
directly after setup. PIE downloads the required asset directories automatically; rebuilding
datasets or embeddings is optional. If a repository requires authentication, use
`uv run hf auth login` or export `HF_TOKEN` in your environment.

Dataset and source inputs accept local paths or explicit HF references, and you can mix them:

```yaml
data:
  preprocessed_dirs:
    - hf://datasets/arcinstitute/pie_replogle_nadig_essential@20c9faef76fc96fdc809871340bd93499413dfe7/preprocessed
    - /path/to/my_screen/preprocessed
  source_dirs:
    esm2: hf://datasets/arcinstitute/pie_sources@cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7/esm2
  gene_text_dir: hf://datasets/arcinstitute/pie_sources@cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7/gene_text
```

The syntax is `hf://datasets/<owner>/<repo>[@revision]/<directory>`. A revision can be a full
commit, tag, or branch; omit it to select `main`. Revisions containing `/` must be percent-encoded
(for example `refs%2Fpr%2F1`). Unpinned references resolve once per data root; PIE records the
immutable commit in the run config and checkpoints. Resume, evaluation, and inference use the
saved commit. To select newer data, provide its new commit explicitly in a new run.
If a local revision record is corrupt, PIE asks for an explicit commit rather than silently
fetching a potentially newer version of the data.

All downloaded assets and their transfer caches live under `PIE_DATA_ROOT`:

```text
$PIE_DATA_ROOT/hf/datasets/<owner>/<repo>/<commit>/<directory>/
```

Only the selected directories are downloaded: dataset `preprocessed/` arrays, or the named
source directories. Complete assets are reused without contacting HF. Interrupted downloads
retain download metadata so rerunning can recover them. PIE checks format, array shapes/dtypes,
and completion metadata before use; missing or truncated cached files trigger a new download.
Repository locks coordinate concurrent jobs and ranks sharing a data root. For multiple nodes,
use a shared `PIE_DATA_ROOT`, or warm the assets independently on each node.

After the assets have been downloaded, set `HF_HUB_OFFLINE=1` to require offline operation.
Missing or incomplete assets then fail with their reference and destination. Checkpoints keep
HF references, so a new machine can fetch the same commits under its own `PIE_DATA_ROOT`.
The same syntax works for `pie-eval`/`pie-infer` `preprocessed_dirs` overrides and for
`pie-sources` dataset inputs.

## Optional: build Replogle data and sources locally

Install the source-building and dataset-processing tools:

```bash
uv sync --frozen --extra sources --extra process
```

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

To use these local assets for a canonical experiment, override `data.preprocessed_dirs`,
`data.source_dirs`, and `data.gene_text_dir`; the published recipes select HF assets by default.

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

## Train and evaluate: replogle_xdataset

This experiment trains on Tahoe, Jiang, ARC VCC 25, and Orion and evaluates zero-shot on Replogle.
It uses two nodes with four GPUs each and a run directory on shared storage. Run the following
on each node with its rank, a shared rendezvous ID, and the node-0 hostname:

```bash
uv run torchrun --nnodes=2 --nproc-per-node=4 --node-rank=<0|1> \
  --master-addr=<node-0 host> --master-port=29500 --rdzv-id="$RDZV_ID" \
  --no-python pie-train experiment=replogle_xdataset
```

Evaluate the best checkpoint on one GPU:

```bash
for rows in test_seen test_unseen; do
  uv run pie-eval experiment_name=replogle_xdataset ckpt=best_auprc \
    split_path=data/splits/replogle_xdataset/$rows.json row_set=$rows
done
```

## License

<!-- TODO -->
