# Semantic Flow Regularization (SFR)

> Semantic Flow Regularization: Teaching LLMs to Generate Diverse Yet Coherent Responses

arXiv: xxx

## Overview
A conditional flow-matching auxiliary loss teaches LLM backbones to preserve multi-modal output diversity during fine-tuning, then is discarded at inference for zero deployment cost.

## Repository Structure

```text
sfr/
├── train/                     # Modified and added files for LLaMA-Factory
└── data_curation/             # Data cleaning scripts for open-source datasets
```

## Installation

This repository only contains the modified or newly added files. Please first prepare a full copy of `LLaMA-Factory`, then install it:

```bash
cd /LLaMA-Factory
pip install -e .
```

## LLaMA-Factory Integration

Copy the files in `train/` to the corresponding locations in `LLaMA-Factory`:

| This repository | LLaMA-Factory path |
| --- | --- |
| `train/collator.py` | `src/llamafactory/data/collator.py` |
| `train/finetuning_args.py` | `src/llamafactory/hparams/finetuning_args.py` |
| `train/parser.py` | `src/llamafactory/hparams/parser.py` |
| `train/tuner.py` | `src/llamafactory/train/tuner.py` |
| `train/llama3_lora_fm.yaml` | `examples/train_lora/llama3_lora_fm.yaml` |
| `train/fm_head.py` | `src/llamafactory/model/model_utils/fm_head.py` |
| `train/fm/` | `src/llamafactory/train/fm/` |

```bash
cp /sfr/train/collator.py src/llamafactory/data/collator.py
cp /sfr/train/finetuning_args.py src/llamafactory/hparams/finetuning_args.py
cp /sfr/train/parser.py src/llamafactory/hparams/parser.py
cp /sfr/train/tuner.py src/llamafactory/train/tuner.py
cp /sfr/train/fm_head.py src/llamafactory/model/model_utils/fm_head.py
cp /sfr/train/llama3_lora_fm.yaml examples/train_lora/llama3_lora_fm.yaml
cp -r /sfr/train/fm src/llamafactory/train/
```

The experiments also involve modifications to the following original `LLaMA-Factory` files. If you maintain these modified files separately, place them at their original paths:

- `src/llamafactory/third_party/muon/muon.py`
- `src/llamafactory/train/trainer_utils.py`

## Training

Edit `examples/train_lora/llama3_lora_fm.yaml` before training, especially:

- `model_name_or_path`
- `dataset`
- `target_encoder_name_or_path`
- `output_dir`

Then launch training with:

```bash
cd /LLaMA-Factory
llamafactory-cli train examples/train_lora/llama3_lora_fm.yaml
```

The new training pipeline is enabled by setting `stage: fm` in the YAML config.

## Data Preparation

The `data_curation/` directory contains scripts for cleaning two open-source code datasets.

### OpenCodeInstruct

```bash
cd /sfr/data_curation/opencodeinstruct
```

Set paths in `data_curation.py`:

```python
INPUT_DIR = "data/to/opencodeinstruct/data"
OUTPUT_DIR = "your/output/dir"
```

Run:

```bash
python data_curation.py
```

Then set paths in `convert_to_train_data.py`:

```python
INPUT_FILE = "your/output/dir/opencodeinstruct_best.jsonl"
SYSTEM_PROMPT_FILE = "system_prompt.md"
OUTPUT_FILE = "your/output/train_data.jsonl"
```

Run:

```bash
python convert_to_train_data.py
```

### rStar-Coder

```bash
cd /sfr/data_curation/rstar-code
```

Set paths in `data_curation.py`:

```python
DATA_DIR = "data/to/rStar-Coder/seed_sft"
OUTPUT_DIR = "your/output/dir"
```

Run:

```bash
python data_curation.py
```

The script writes `seed_sft_cleaned.jsonl` and `seed_sft_cleaned.json` to `OUTPUT_DIR`.
