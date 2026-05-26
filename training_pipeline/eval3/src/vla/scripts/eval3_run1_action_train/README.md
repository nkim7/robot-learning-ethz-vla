# Eval 3 Run 1 Action Expert Training

Run 1 uses the good VLM LoRA that already recognizes celebrity position, but keeps it frozen. Training updates only the SmolVLA action expert and state projection on the 90 permutation robot episodes.

This prevents the robot action loss from damaging identity/location recognition.

## Workflow

```bash
cd ~/robot-learning-ethz-vla
source .venv/bin/activate

# login once if needed
huggingface-cli login
wandb login

python scripts/eval3_run1_action_train/sanity_check_run1.py

python scripts/eval3_run1_action_train/check_trainable_params.py

bash scripts/eval3_run1_action_train/train_eval3_run1.sh

bash scripts/eval3_run1_action_train/find_latest_run1_checkpoint.sh
```

In this sandbox the working tree is currently `/home/shadeform`; run the same commands there unless you move the files into `~/robot-learning-ethz-vla`.

## What Gets Trained

- Frozen: SmolVLM2 VLM backbone.
- Frozen: Eval 3 VLM LoRA at `checkpoint/eval3_phase2_TOY`.
- Trainable: SmolVLA action expert.
- Trainable: SmolVLA state projection.

The launcher uses LeRobot's installed trainer, with a small construction-time patch that loads the external PEFT adapter into `vlm_with_expert.vlm` using `PeftModel.from_pretrained(..., is_trainable=False)`, then freezes every VLM/LoRA parameter.

## Run 1 Defaults

- Dataset: `data/eval3_90_permutation`
- Base VLM: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Output dir: `outputs/train/eval3_run1_action_expert_frozen_vlm_lora`
- Hub repo: `robot-learning-group47/eval3_run1_action_expert_frozen_vlm_lora_smolvla`
- W&B project: `robot-learning-group47-eval3`
- W&B run name: `eval3_run1_action_expert_frozen_vlm_lora_lr3e5`
- LR: `3e-5`
- Steps: `8000`
- Batch size: `16`
- Save frequency: `500`
- Eval frequency: `0`
- AMP: enabled
- Device: `cuda`
- Video backend: `pyav`
- VLM layers: full backbone with `policy.num_vlm_layers=-1`, so the LoRA module names match the adapter.

Use batch size 16 first. If training is stable and memory is comfortable, batch size 32 is a later option. If it OOMs, reduce batch size to 8.

Test checkpoints at 2000, 4000, 6000, and 8000 steps. Do not assume the last checkpoint is best.

Do not overwrite previous outputs unless you explicitly delete or rename the output folder. Do not unfreeze the LoRA in Run 1.
