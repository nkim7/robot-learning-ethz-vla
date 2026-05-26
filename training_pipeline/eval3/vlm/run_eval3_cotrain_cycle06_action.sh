#!/usr/bin/env bash
# Eval 3 Cotrain — Cycle 06 Action Training (resumes from cycle_05)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"
CYCLE05_CKPT="$CHECKPOINT_ROOT/eval3_cotrain/cycle_05_action_checkpoint"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/eval3_cotrain/cycle_06_action}"
DATASET_REPO="robot-learning-group47/eval3_cotrain_action_90toy_plus_lmr"
DATA_ROOT="${DATA_ROOT:-/tmp/eval3_cotrain_action_90toy_plus_lmr}"

[[ -f "$CYCLE05_CKPT/model.safetensors" ]] || { echo "ERROR: cycle_05 checkpoint not found: $CYCLE05_CKPT"; exit 1; }

echo "Pretrained: $CYCLE05_CKPT  |  Output: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$REPO_ROOT/lerobot/src:${PYTHONPATH:-}" \
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --policy.pretrained_path="$CYCLE05_CKPT" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.use_peft=false \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.tokenizer_max_length=48 \
    --policy.optimizer_lr=1e-5 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 \
    --policy.scheduler_decay_lr=2.5e-6 \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATA_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --job_name=eval3_cotrain_cycle_06_action \
    --resume=false \
    --seed=1000 \
    --batch_size=16 \
    --steps=500 \
    --save_freq=500 \
    --log_freq=25 \
    --eval_freq=0 \
    --num_workers=4 \
    --use_policy_training_preset=true \
    --wandb.enable="${WANDB_ENABLE:-false}" \
    --wandb.project=robot-learning-group47-eval3 \
    "$@"
