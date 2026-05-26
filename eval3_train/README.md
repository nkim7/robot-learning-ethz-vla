# Eval 3 Code Export

This archive contains the code and lightweight metadata from the Brev workspace.
It intentionally excludes heavy runtime artifacts such as datasets, images,
checkpoints, model weights, logs, W&B runs, caches, and the Python virtualenv.

## Archive Layout

```text
scripts/
  eval3_vlm_lora/
  eval3_cotrain/
  eval3_bridge_train/
  eval3_alignment/
  eval3_run1_action_train/
  eval3_run2_lmr210_action_train/
  eval3_e2e_aux_vlm/

data_metadata/
  README.md files, dataset summaries, target-slot sidecars, and action dataset
  manifests copied from data/.

checkpoint_metadata/
  README/config files copied from checkpoint/ adapters, without model weights.

site_packages_patches/
  Selected installed-package source files that were patched during the Brev
  session, copied out of .venv for inspection/reapplication.

init.sh
  Workspace bootstrap script from /home/shadeform.
```

## Main Script Groups

### `scripts/eval3_vlm_lora`

Utilities for VLM LoRA work:

- Prepare position-QA data from robot frames.
- Fine-tune SmolVLM LoRA adapters.
- Evaluate LoRA adapters on position recognition.
- Inspect adapter checkpoints.
- Upload/download related Hugging Face assets.

Useful entry points:

```bash
python scripts/eval3_vlm_lora/finetune_position_lora.py
python scripts/eval3_vlm_lora/evaluate_position_jsonl.py
python scripts/eval3_vlm_lora/inspect_checkpoint.py --path <adapter_dir>
```

### `scripts/eval3_cotrain`

Alternating VLM/action training workflows:

- VLM grounding repair on `data/eval3_position_qa_3frames`.
- Action binding repair on Eval 3 robot datasets.
- Cycle 06 VLM repair followed by frozen-LoRA action training.
- Evaluation of cotrain VLM grounding.

Useful entry points:

```bash
bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
bash scripts/eval3_cotrain/train_eval3_cycle06_vlm_repair_then_action.sh
python scripts/eval3_cotrain/eval_cotrain_vlm_grounding.py --checkpoint <adapter_dir>
```

### `scripts/eval3_bridge_train`

Bridge training for celebrity-prompt action policies:

- Relabels action datasets with target slots.
- Adds auxiliary bridge losses: hidden-state alignment and spatial
  classification.
- Trains bridge checkpoints such as 2k and 3k step policies.
- Provides a bridge policy-server launcher.

Useful entry points:

```bash
bash scripts/eval3_bridge_train/train_eval3_bridge.sh
python scripts/eval3_bridge_train/check_bridge_checkpoint.py --output-dir <bridge_run>
python scripts/eval3_bridge_train/bridge_policy_server_launcher.py --host 0.0.0.0 --port 8080
```

### `scripts/eval3_alignment`

LoRA alignment fine-tuning:

- Loads an existing action checkpoint.
- Injects a better frozen VLM LoRA adapter.
- Trains only the action/state/action-output modules to realign with the new
  VLM embedding distribution.

Useful entry point:

```bash
bash scripts/eval3_alignment/train_yann_left_lora_alignment.sh
```

### `scripts/eval3_run1_action_train`

Run 1 action-expert training with a frozen Eval 3 VLM LoRA. Contains the LoRA
patch that injects a PEFT adapter into the SmolVLA VLM submodule and freezes it.

Useful entry points:

```bash
bash scripts/eval3_run1_action_train/train_eval3_run1.sh
bash scripts/eval3_run1_action_train/resume_eval3_run1_from_1500.sh
python scripts/eval3_run1_action_train/sanity_check_run1.py
```

### `scripts/eval3_run2_lmr210_action_train`

Run 2 action training for the 90+LMR210 dataset, plus a policy-server launcher
with first-frame VLM probing / instruction rewrite logic.

Useful entry points:

```bash
bash scripts/eval3_run2_lmr210_action_train/train_eval3_run2_lmr210_30k.sh
python scripts/eval3_run2_lmr210_action_train/run2_policy_server_launcher.py --host 0.0.0.0 --port 8080
```

### `scripts/eval3_e2e_aux_vlm`

End-to-end action training with auxiliary VLM losses. Includes a patch module
and training launcher.

Useful entry point:

```bash
bash scripts/eval3_e2e_aux_vlm/train_eval3_e2e_aux_vlm.sh
```

## Metadata Folders

### `data_metadata`

This contains small metadata files from datasets, not the full datasets. The
full datasets are too large for a code export and should be downloaded from
Hugging Face or regenerated with the scripts.

Examples:

- Position-QA summaries.
- Target-slot sidecar labels for bridge training.
- Action dataset manifests.

### `checkpoint_metadata`

This contains small adapter/checkpoint config files, not `.safetensors` model
weights or large tokenizer files. Use these files to inspect base model names,
LoRA rank/alpha, processor settings, and adapter provenance.

### `site_packages_patches`

This contains selected patched installed-package source files. It is not a full
replacement for the package. Copy or diff these files against the same package
version if you need to reapply local changes.

Currently included:

```text
site_packages_patches/lerobot/async_inference/policy_server.py
```

## What Is Not Included

The archive excludes:

- `.venv/`
- `.git/`
- `.codex/`, `.claude/`, `.ssh/`, credentials, and shell history
- `outputs/`
- full `data/`
- full `checkpoint/` weights
- `logs/`
- W&B run directories
- cache directories
- image dumps

## Restoring The Workspace Shape

On another machine, unzip the archive into a project folder, then restore heavy
assets separately:

```bash
unzip eval3_code_scripts_export_YYYYMMDD_HHMMSS.zip
```

Then download checkpoints/datasets from Hugging Face as needed. The scripts
expect paths similar to the Brev layout:

```text
/home/shadeform/data/...
/home/shadeform/checkpoint/...
/home/shadeform/outputs/train/...
```

You can override many paths via environment variables in the shell scripts.
