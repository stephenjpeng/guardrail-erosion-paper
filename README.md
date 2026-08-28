# Guardrail erosion audit

This repository reproduces the experiments reported in *Guardrail Erosion in Domain Fine-tuned Large Language Models: A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs*. The audit compares 26 biomedical adaptations with 12 lineage-related base checkpoints on MedAbstain and ClinDet-Bench.

The reported runs used a common plain-text completion prompt, greedy decoding, a 2,048-token generation limit, seed 42, and top-20 generated-token log probabilities for up to 2,048 positions. `study.json` records the 76 selected runs, file hashes, expected claims, and required figures. `models.csv` records the model revisions and lineage pairs.

## Repository contents

The repository contains the analysis and inference code, frozen study configuration, model table, and dependency lock. The compact paper artifact and model weights are not included.

## Install

CPU analysis requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run python experiment.py self-test
```

The reported inference runs used two vLLM environments:

```bash
uv sync --frozen --extra inference-l40s  # vLLM 0.26.0
uv sync --frozen --extra inference-h100  # vLLM 0.27.1
```

The extras are intentionally mutually exclusive.

## Reproduce the reported analysis

The 76 hashes in `study.json` select the reported runs from the mixed source directory. Timestamps do not determine inclusion:

```bash
uv run python experiment.py compact \
  --runs-source /path/to/data/runs/v2 \
  --out paper-items.parquet
uv run python analyze.py \
  --items paper-items.parquet \
  --out results \
  --check
```

Compaction reads about 69 GB of raw JSONL. The output retains the derived item signals used in the reported analysis and omits completions and raw top-token payloads. Checking the parsing logic requires the original runs or a full rerun.

## Prepare benchmark data

ClinDet is built from commit `1a05c632512be8270be863ed0d304c4742a535bd`. MedAbstain uses only the public MedQA-USMLE half; AMBOSS data is not used or redistributed.

```bash
uv run python experiment.py prepare \
  --medabstain-repo /path/to/MedAbstain \
  --clindet-repo /path/to/ClinDet_Benchmark \
  --out data
```

The command fails unless the prepared files match the hashes and row counts in `study.json`.

## Run inference

```bash
uv run python experiment.py run \
  --data-root data \
  --runs-out runs \
  --models all \
  --benchmarks medabstain,clindet
```

Run one model per process. The script writes deterministic `<model>__<benchmark>.jsonl` files with one manifest record followed by item records. It does not overwrite existing files unless `--force` is set. The complete cohort requires gated checkpoints and multi-GPU CUDA hardware. `models.csv` records revisions, tensor parallelism, dtype, and access notes.

## Outputs

`analyze.py` writes:

- `pair_metrics.csv`, one row per adapted/base pair;
- `claim_registry.csv`, the 15-claim primary family and corrected inference;
- sensitivity and moderator tables;
- `provenance.json` with input and output hashes;
- the 16 PDF filenames listed in `study.json` and referenced by the paper.

`--check` fails on protocol, cohort, claim, or figure drift.

## Data and model terms

- MedAbstain: CC BY-NC 4.0, public MedQA-USMLE half only. The upstream revision was not retained, so the prepared-file SHA-256 in `study.json` is the frozen data identity.
- ClinDet-Bench: MIT, pinned upstream commit recorded above.
- Model checkpoints: not redistributed. Users must obtain each checkpoint from its upstream repository and comply with its license and access terms.
- Generated outputs: publication terms remain under review.

## Limitations

This audit measures benchmark-defined capability and correctness, confidence and caution, and calibration and commitment. It does not evaluate clinical deployment or establish a causal fine-tuning mechanism.
