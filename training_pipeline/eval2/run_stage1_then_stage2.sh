#!/usr/bin/env bash
# ============================================================
# Stage 1 → Stage 2  (sequential, single tmux session)
#
# Runs Stage 1 (position bootstrap), then automatically
# continues into Stage 2 (compositional prompts) using the
# Stage 1 last checkpoint as initialisation.
#
# Usage:
#   tmux new -s eval2_training
#   bash ~/run_stage1_then_stage2.sh 2>&1 | tee ~/outputs/stage1_stage2.log
# ============================================================
set -euo pipefail

SCRIPTS_DIR="${SCRIPTS_DIR:-${HOME}}"
STAGE1_OUT="/home/shadeform/outputs/eval2_stage1_position_bootstrap"
STAGE1_CKPT="${STAGE1_OUT}/checkpoints/last/pretrained_model"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"

echo "════════════════════════════════════════════════════════════"
echo "  Eval 2 Training Pipeline"
echo "  Stage 1: position bootstrap  → eval2_stage1_position_bootstrap"
echo "  Stage 2: compositional prompts → eval2_stage2_compositional"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Stage 1 ────────────────────────────────────────────────────────────────
echo "[$(date)] ▶ Starting Stage 1 …"
bash "${SCRIPTS_DIR}/run_stage1_position_bootstrap.sh"
echo "[$(date)] ✓ Stage 1 complete."
echo ""

# ── Gate: verify Stage 1 produced a checkpoint ────────────────────────────
if [[ ! -f "${STAGE1_CKPT}/model.safetensors" ]]; then
  echo "ERROR: Stage 1 checkpoint missing at ${STAGE1_CKPT}"
  echo "       Stage 2 cannot proceed."
  exit 1
fi

echo "[$(date)] Stage 1 checkpoint verified: ${STAGE1_CKPT}"
echo ""

# ── Stage 2 ────────────────────────────────────────────────────────────────
echo "[$(date)] ▶ Starting Stage 2 …"
STAGE1_CKPT="${STAGE1_CKPT}" bash "${SCRIPTS_DIR}/run_stage2_compositional.sh"
echo "[$(date)] ✓ Stage 2 complete."
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  Pipeline finished."
echo "  Stage 2 checkpoint: /home/shadeform/outputs/eval2_stage2_compositional/checkpoints/last"
echo "  HF model: https://huggingface.co/robot-learning-group47/smolvla-eval2-stage2-compositional"
echo "════════════════════════════════════════════════════════════"
