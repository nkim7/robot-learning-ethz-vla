# Eval3 SmolVLM / LoRA Utilities

This folder contains a small, token-safe setup for the Eval3 resources:

- Model/checkpoint: `robot-learning-group47/eval3_phase2_TOY`
- Dataset: `robot-learning-group47/eval3_90_permutation`

Artifacts are stored under the requested local paths:

- Models/checkpoints: `checkpoint/`
- Datasets: `data/`

## Setup

```bash
scripts/eval3_vlm_lora/setup_env.sh
```

The setup script installs PyTorch, Transformers, PEFT, Hugging Face Hub, Datasets, and `lerobot[smolvla]`.

To add a PyTorch wheel index:

```bash
TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 scripts/eval3_vlm_lora/setup_env.sh
```

## Authentication

Do not put your Hugging Face token in any file. Export it in your shell:

```bash
export HF_TOKEN=hf_...
```

Every script that touches Hugging Face checks `HF_TOKEN` and exits clearly if it is missing.

## Download Assets

```bash
.venv/bin/python scripts/eval3_vlm_lora/download_assets.py
```

Expected output paths:

- `checkpoint/eval3_phase2_TOY`
- `data/eval3_90_permutation`

## Inspect Checkpoint Type

```bash
.venv/bin/python scripts/eval3_vlm_lora/inspect_checkpoint.py
```

Detection rules:

- `adapter_config.json` exists: PEFT/LoRA adapter.
- `config.json` and model weights exist: full fine-tuned model.
- Otherwise: print files and explain what is missing.

## Run Image + Prompt Inference

```bash
.venv/bin/python scripts/eval3_vlm_lora/run_vlm_lora_inference.py \
  --image /path/to/image.jpg \
  --prompt "Describe the robot scene and the next action."
```

If the LoRA adapter does not include `base_model_name_or_path`, pass it explicitly:

```bash
.venv/bin/python scripts/eval3_vlm_lora/run_vlm_lora_inference.py \
  --base-model HuggingFaceTB/SmolVLM-Instruct \
  --image /path/to/image.jpg \
  --prompt "What should the robot do next?"
```
