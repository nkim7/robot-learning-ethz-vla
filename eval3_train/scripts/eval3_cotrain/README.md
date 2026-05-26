# Eval 3 Alternating Co-Training

This folder implements Path 1: an alternating phase pipeline for Eval 3
SmolVLA. It does not start training automatically.

The requested setup is:

- Start action checkpoint: `outputs/train/eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k/checkpoints/014000/pretrained_model`
- Start VLM LoRA: `checkpoint/eval3_phase2_TOY`
- Action data sources: `data/eval3_90_permutation` plus `data/eval3_lmr`
- Prepared action dataset: `data/eval3_cotrain_action_90toy_plus_lmr`
- VLM grounding data: `data/eval3_position_qa_3frames`
- Output dir: `outputs/train/eval3_cotrain_action_lora_auxvlm_5k`

I interpreted "checkpoint 1400" as the existing run2 checkpoint `014000`.

## Why This Pipeline

The action policy already learned most of the coke grasp and picture-area motion,
but celebrity/location grounding can weaken during full SmolVLA inference. This
pipeline keeps both skills active by alternating:

- Action phase: update action expert and state projection on robot episodes.
- Grounding phase: refresh the SmolVLM LoRA on position QA images.

The base VLM and base vision weights stay frozen. The fallback action phase uses
the existing Eval 3 LoRA loader, which freezes the LoRA during action training.
The grounding phase then trains only the LoRA adapter.

## Implemented Files

```text
scripts/eval3_cotrain/cotrain_common.py
scripts/eval3_cotrain/check_cotrain_inputs.py
scripts/eval3_cotrain/prepare_action_dataset.py
scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
scripts/eval3_cotrain/train_eval3_binding_repair_toy4_lmr1.sh
scripts/eval3_cotrain/eval_cotrain_vlm_grounding.py
scripts/eval3_cotrain/README.md
```

## Binding Repair Run

After cycle 05, use this short end-to-end VLA repair run if the robot motion is
good but celebrity-to-position choice is inconsistent.

It starts from:

```text
outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_action/checkpoints/000500/pretrained_model
outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_05_vlm_lora
```

It materializes this TOY-heavy dataset:

```text
data/eval3_binding_repair_action_90toy4_plus_lmr1
```

The dataset contains:

```text
390 episodes
87,626 frames
90 TOY episodes x 4
30 L/M/R episodes x 1
```

Default training settings:

```text
steps: 1500
save_freq: 500
action/state LR: 5e-6
VLM LoRA LR: 1e-6
base VLM: frozen
base vision: frozen
VLM LoRA: trainable
```

Run:

```bash
bash scripts/eval3_cotrain/train_eval3_binding_repair_toy4_lmr1.sh
```

Recommended tmux:

```bash
mkdir -p logs
tmux new-session -d -s eval3_binding_repair \
  'cd /home/shadeform && bash scripts/eval3_cotrain/train_eval3_binding_repair_toy4_lmr1.sh 2>&1 | tee logs/eval3_binding_repair_$(date +%s).log'
```

Use overrides for a shorter or longer repair run:

```bash
STEPS=1000 bash scripts/eval3_cotrain/train_eval3_binding_repair_toy4_lmr1.sh
STEPS=2000 bash scripts/eval3_cotrain/train_eval3_binding_repair_toy4_lmr1.sh
```

The training checkpoint includes the updated LoRA parameters inside the saved
SmolVLA policy. For policy-server construction, keep using the cycle 05 LoRA
path as `EVAL3_VLM_LORA_PATH`; loading the repaired policy checkpoint should
then overwrite the matching adapter weights from `model.safetensors`.

## Cycle Design

Default run:

```text
5 cycles
500 action steps per cycle
100 VLM LoRA refresh steps per cycle
```

Each cycle writes:

```text
outputs/train/eval3_cotrain_action_lora_auxvlm_5k/
  cycle_01_action/
  cycle_01_vlm_lora/
  cycle_01_manifest.txt
  ...
  latest_manifest.txt
```

The action checkpoint for a cycle is expected at:

```text
cycle_XX_action/checkpoints/000500/pretrained_model
```

The refreshed VLM LoRA for that cycle is:

```text
cycle_XX_vlm_lora
```

## Data

Robot action data:

```text
data/eval3_90_permutation
data/eval3_lmr
```

The training launcher first materializes a single local LeRobot dataset at:

```text
data/eval3_cotrain_action_90toy_plus_lmr
```

This prepared dataset contains 120 episodes:

```text
90 TOY celebrity permutation episodes
30 L/M/R motion episodes
```

The old premerged `data/eval3_90_plus_lmr_210` is not used by this pipeline.

Grounding data:

```text
data/eval3_position_qa_3frames
```

This contains 270 image QA samples, three frames per episode. Labels use the
manual episode rule you provided:

```text
0-29: Taylor Swift
30-59: Barack Obama
60-89: Yann LeCun

Within each 30:
0-9: left
10-19: middle
20-29: right
```

The old `data/eval3_90_permutation/eval3_episode_permutation_map.csv` is not
used.

## Run

From `/home/shadeform`:

```bash
bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
```

Recommended tmux command:

```bash
mkdir -p logs
tmux new-session -d -s eval3_cotrain \
  'cd /home/shadeform && bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh 2>&1 | tee logs/eval3_cotrain_$(date +%s).log'
```

The script refuses to overwrite an existing output dir. To intentionally replace
it:

```bash
COTRAIN_OVERWRITE=1 bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
```

Useful overrides:

```bash
CYCLES=3 ACTION_STEPS=500 VLM_STEPS=100 bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
START_ACTION_CHECKPOINT=/path/to/pretrained_model bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
START_VLM_LORA=/path/to/lora_adapter bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
ACTION_LMR_DATASET_ROOT=/home/shadeform/data/eval3_lmr_210 bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
```

To rebuild the prepared action dataset:

```bash
COTRAIN_PREPARE_OVERWRITE=1 bash scripts/eval3_cotrain/train_eval3_cotrain_alternating.sh
```

## Validate Inputs

```bash
python scripts/eval3_cotrain/check_cotrain_inputs.py
```

This checks:

- Starting action checkpoint exists.
- Starting LoRA has `adapter_model.safetensors`.
- Robot dataset has `meta/info.json`.
- Grounding train/eval JSONL files exist.

## Evaluate Grounding

Evaluate any refreshed LoRA adapter:

```bash
python scripts/eval3_cotrain/eval_cotrain_vlm_grounding.py \
  --checkpoint outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_01_vlm_lora \
  --data data/eval3_position_qa_3frames/eval.jsonl \
  --output outputs/train/eval3_cotrain_action_lora_auxvlm_5k/cycle_01_vlm_lora/eval_predictions.jsonl
```

The evaluator normalizes answers to:

```text
left
middle
right
unknown
```

It prints accuracy and example correct/incorrect predictions.

## Important Notes

- Do not assume the last checkpoint is best.
- Test action checkpoints after each cycle on the robot.
- Good candidates to test are cycle 1 through cycle 5, not only the final one.
- If grounding drops, keep the earlier cycle LoRA.
- If robot motion gets worse, keep the earlier action checkpoint.
- This path is intentionally conservative; true multi-loss co-training should be
  implemented only after this alternating baseline is tested.
