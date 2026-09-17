# What Brain Data Adds to Language Model Training

This repository contains the reproduction pipeline for the CoNLL 2026 paper
[What Brain Data Adds to Language Model Training](https://aclanthology.org/2026.conll-main.12/).
It covers BERT and GPT-2 on the Harry Potter and Subset Moth reading datasets.

The repository contains code and configuration only. fMRI data, checkpoints,
evaluation results, and generated figures are intentionally not distributed.

## Installation

The supported environment uses Python 3.10 or 3.11.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

PyTorch wheels are platform-specific. On CUDA systems, install the appropriate
PyTorch 2.3.1 wheel before installing the remaining requirements.

## Data

Export the preprocessed numerical arrays from the provider archive and describe
them with a source manifest. The preparation command validates the subject
order, four-fold run mapping, response dimensions, ROI masks, and noise
ceilings, then creates a safe canonical bundle:

```bash
lm-brain-tuning prepare-data \
  --source-manifest source-hp.json \
  --output data/hp
```

See [docs/data.md](docs/data.md) and
[docs/source-manifest.example.json](docs/source-manifest.example.json). Data
directories are ignored by Git. The `prepare-data` input boundary is the
documented NPY/JSONL source manifest.

## Paper configuration

The canonical configuration is [`configs/paper/paper.yaml`](configs/paper/paper.yaml).
It records the complete four-fold matrix, subject ordering, LoRA settings, loss
weights, context construction, ridge grid, and downstream seeds. Both model
families use a causal language-modeling objective. FlashHolmes is evaluated with
seeds 0 through 4.

The four training conditions are:

| Condition | LM weight | Brain weight | LR | LoRA rank |
|---|---:|---:|---:|---:|
| Pretrained | — | — | — | — |
| Brain-Tuned | 0 | 1 | 5e-5 | 4 |
| Stimulus-Tuned | 1 | 0 | 5e-5 | 4 |
| Jointly-Tuned | 0.1 | 10 | 5e-4 | 8 |

## Running one experiment

```bash
lm-brain-tuning train \
  --model bert --dataset hp --regime brain --fold 0 --subject F \
  --data data/hp --output outputs/selection-training

lm-brain-tuning select-checkpoint \
  --log outputs/selection-training/bert-hp-brain-fold0/sub-F/log.csv \
  --checkpoint-root outputs/selection-training/bert-hp-brain-fold0/sub-F \
  --output outputs/selection-training/bert-hp-brain-fold0/sub-F/selection.json

lm-brain-tuning train \
  --model bert --dataset hp --regime brain --fold 0 --subject F \
  --data data/hp --output outputs/training \
  --selection outputs/selection-training/bert-hp-brain-fold0/sub-F/selection.json

lm-brain-tuning evaluate-brain \
  --model bert --dataset hp --regime brain --fold 0 --subject F \
  --data data/hp \
  --selection outputs/training/bert-hp-brain-fold0/sub-F/selection.json \
  --output outputs/brain-evaluation
```

Each run writes a JSON manifest with its resolved identity, seed, fold, dataset
checksum, runtime versions, and completion state. Checkpoint-selection manifests
also record the source log checksum and are rejected if reused for another
dataset, subject, fold, or run. Numerical arrays use non-pickled NPY files;
tables and summaries use CSV and JSON.

The first training pass reserves the final 20% of the ordered training data for
checkpoint selection, evaluating every 3 steps for Harry Potter and every 20
steps for Subset Moth. The second pass runs on all three training folds for the
complete five-epoch schedule and retains the selected global step. The
configured seed is applied when `Trainer` is created, after model and LoRA
initialization; the run manifest records this scope explicitly. The brain head
predicts every language-ROI voxel above the
noise-ceiling threshold separately; its loss is the mean of the per-voxel batch
Pearson losses, with no voxel aggregation.

## Full matrix and Slurm

Inspect the full matrix before running it:

```bash
lm-brain-tuning reproduce \
  --data-root data --output outputs/reproduction --dry-run
```

Replace `--dry-run` with `--local` for sequential execution or `--slurm` to
materialize a Slurm array launcher. Add `--submit` only together with `--slurm`
when the generated launcher should be submitted immediately. Fine-tuned array
elements run validation training, checkpoint selection, full-data retraining,
and brain evaluation sequentially; pretrained elements run evaluation only.
Generated-output commands refuse to mix with a non-empty destination.

## FlashHolmes

Holmes is kept as a separate, pinned checkout and is not vendored here. The
checkout metadata lives in [`configs/holmes.json`](configs/holmes.json). Export
the language-model backbone from a trained checkpoint, run the external tool,
then normalize its result table. The run command performs the backbone export
automatically for repository checkpoints:

```bash
git clone https://github.com/Holmes-Benchmark/holmes-evaluation.git external/holmes
git -C external/holmes checkout bded82c8e8941fbc651c2eee7203b01f6a6c1de6

lm-brain-tuning evaluate-holmes --run \
  --checkout external/holmes \
  --model-path outputs/training/bert-hp-brain-fold0/sub-F/checkpoint-STEP \
  --output outputs/holmes/raw

lm-brain-tuning evaluate-holmes \
  --normalize-csv results_flash-holmes.csv \
  --system brain \
  --comparison-unit bert-hp-fold0-subF \
  --analysis-unit hp-subF \
  --aggregation-unit bert-hp \
  --task-metadata external/holmes/results/flash-holmes.csv \
  --output outputs/holmes/bert-hp-brain.csv
```

The `analyze` command computes per-task tests, win rates, Holm-corrected
Wilcoxon comparisons, and the final win-rate plot. `comparison-unit` identifies
a matched model/dataset/subject/fold across the four regimes; `analysis-unit`
identifies the subject used for the final paired Wilcoxon test; and
`aggregation-unit` identifies the model family averaged within that subject.
Task metadata supplies the linguistic subfield when it is absent from the raw
result CSV. The five Holmes seeds are compared within each matched unit before
averaging across folds, tasks within subfield, subfields, and model families.

## License

Original code in this repository is released under the MIT License. Dataset and
external benchmark licenses remain with their respective providers.
