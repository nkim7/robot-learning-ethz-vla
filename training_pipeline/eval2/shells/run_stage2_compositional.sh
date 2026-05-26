#!/usr/bin/env bash
# ============================================================
# Stage 2 — Official-Style Compositional Prompt Training
#
# Goal: connect official Eval2 prompts (ordinal, relative-to-color,
#       exclusion) to the left/middle/right motor modes learned in Stage 1.
#
# Design:
#   Backbone:  strict10 (loaded from Stage 1 15k checkpoint — FROZEN)
#   Expert:    init from Stage 1 15k checkpoint (position-trained)
#   Dataset:   robot-learning-group47/eval2_compositional_augmented
#              720 episodes  |  22 prompt variants  |  all validated Stage 0
#   LR:        3e-5 peak  →  cosine decay  →  7.5e-7  (0.3× Stage 1 LR)
#              warmup: 500 steps  |  decay: 20 000 steps
#   Steps:     40 000
#   Save:      every 5 000 steps
#
# Prompt-family sampling during Stage 2:
#   ordinal       35 %
#   relative      40 %
#   exclusion     25 %
#   color_direct   0 %
#   target slots left / middle / right are balanced 1:1:1 per batch.
#
# Usage (run AFTER Stage 1 finishes):
#   bash ~/run_stage2_compositional.sh
#
# Or use run_stage1_then_stage2.sh for automated sequential execution.
# ============================================================
set -euo pipefail

# ── Manual gate: robot evaluation before Stage 2 ──────────────────────────
# Stage 1 completed. Before launching Stage 2, verify the model physically.
# When ready to proceed, delete the hold file:
#   rm /home/shadeform/HOLD_BEFORE_STAGE2
#   then re-run:  bash ~/run_stage2_compositional.sh
HOLD_FILE="/home/shadeform/HOLD_BEFORE_STAGE2"
if [[ -f "${HOLD_FILE}" ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  STAGE 1 COMPLETE — ROBOT EVALUATION GATE               ║"
  echo "║                                                          ║"
  echo "║  Stage 2 is HELD. Evaluate the Stage 1 checkpoint on    ║"
  echo "║  the robot first:                                        ║"
  echo "║                                                          ║"
  echo "║    Checkpoint: checkpoints/015000/pretrained_model       ║"
  echo "║    HF model:   smolvla-eval2-stage1-position-bootstrap   ║"
  echo "║                                                          ║"
  echo "║  Pass condition (test left / middle / right prompts):    ║"
  echo "║    1. grabs banana                                       ║"
  echo "║    2. finishes within 20s                                ║"
  echo "║    3. placement direction changes correctly              ║"
  echo "║                                                          ║"
  echo "║  When ready to launch Stage 2:                           ║"
  echo "║    rm /home/shadeform/HOLD_BEFORE_STAGE2                 ║"
  echo "║    bash ~/run_stage2_compositional.sh                    ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  exit 0
fi

STAGE1_CKPT="${STAGE1_CKPT:-/home/shadeform/outputs/eval2_stage1_position_bootstrap/checkpoints/015000/pretrained_model}"
DATASET="robot-learning-group47/eval2_compositional_augmented"
RUN_NAME="eval2-stage2-compositional"
OUT="/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"
LOG="${OUT}.log"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/home/shadeform/train_eval2_target_routing_stage2.py}"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"

mkdir -p "${DATASET_ROOT}" "$(dirname "${LOG}")"

# ── Verify Stage 1 checkpoint ──────────────────────────────────────────────
if [[ ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: Stage 1 checkpoint not found: ${STAGE1_CKPT}/model.safetensors"
  echo "       Run run_stage1_position_bootstrap.sh first."
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: Stage 2 trainer not found: ${TRAIN_SCRIPT}"
  exit 1
fi

echo "============================================================"
echo "  Stage 2 — Compositional Prompt Training"
echo "  Init:     ${STAGE1_CKPT}"
echo "  Dataset:  ${DATASET}"
echo "  Output:   ${OUT}"
echo "  Steps:    40 000  |  LR: 3e-5  |  save: every 5k"
echo "  Sampling: ordinal 35% | relative 40% | exclusion 25% | color_direct 0%"
echo "  Slots:    left / middle / right balanced 1:1:1"
echo "  Backbone: FROZEN (train_expert_only=true, freeze_vision_encoder=true)"
echo "============================================================"
echo ""

PYTHONPATH="/home/shadeform/lerobot/src:/home/shadeform" \
/home/shadeform/lerobot/.venv/bin/python "${TRAIN_SCRIPT}" \
  --policy.path="${STAGE1_CKPT}" \
  --dataset="${DATASET}" \
  --dataset-root="${DATASET_ROOT}" \
  --out-dir="${OUT}" \
  --steps=40000 \
  --batch-size=30 \
  --lr=3e-5 \
  --save-every=5000 \
  --log-every=25 \
  --balanced-augmented \
  --wandb \
  --wandb-project=lerobot \
  --wandb-name="${RUN_NAME}" \
  2>&1 | tee "${LOG}"

echo ""
echo "[$(date)] Stage 2 training complete."
echo "Best checkpoint: ${OUT}/checkpoints/last/pretrained_model"
echo "HF model:        https://huggingface.co/robot-learning-group47/smolvla-eval2-stage2-compositional"
