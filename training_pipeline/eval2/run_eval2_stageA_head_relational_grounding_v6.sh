#!/usr/bin/env bash
# Stage A v6: clockwise-convention hard-case fine-tune from v5@300
#
# Key changes from v5:
#   - New family: clockwise_convention (30% of batches).
#     4 scaffold prompts per (side, anchor, target, layout) that explicitly
#     teach right_of/left_of as next/prev in clockwise bowl order.
#   - Per-sample loss weights:
#       bgr/gbr right_of raw_relative  → 4.0×  (clockwise_conflict)
#       clockwise_convention prompts   → 3.0×
#       other bgr/gbr                  → 2.0×
#       all others                     → 1.0×
#   - Family weights: raw=0.40, clockwise_convention=0.30,
#                     scaffolded=0.20, semi=0.05, simple=0.05
#   - Init: v5@300 (all three heads warm-started)
#   - Steps: 200, eval every 50
#   - LR: lora=2e-6, head=2e-5  (conservative; near-convergence targeted fix)
#
# Stage B gate — ALL must reach 95% (pair-primary):
#   raw relative pair_acc          >= 95%
#   raw anti-shortcut pair_acc     >= 95%
#   every layout raw pair_acc      >= 95%   (brg/bgr/rbg/rgb/gbr/grb)
#   every anchor raw pair_acc      >= 95%   (blue/red/green)
#   left_of / right_of raw pair_acc >= 95%
#
# bgr ep=10 rep=4 cases (right_of/blue and right_of/green) are NOT excluded.
# They are critical hard cases. Gate requires them to pass.
#
# Script prints STAGE_B_GATE_PASS or STAGE_B_GATE_FAIL at each eval step.
# Do NOT start Stage B until STAGE_B_GATE_PASS.
#
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_INIT_CKPT=/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model
INIT_CKPT=/home/shadeform/outputs/eval2_stageA_head_relational_grounding_v5_pairprimary_from_v4_750/checkpoints/000300/pretrained_model

DATASET=robot-learning-group47/eval2_fixed_bowl_fixed_banana_108
DATASET_ROOT=/home/shadeform/lerobot_cache/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108

OUT_DIR=/home/shadeform/outputs/eval2_stageA_head_relational_grounding_v6_clockwise_from_v5_300
TRAIN_SCRIPT=/home/shadeform/train_eval2_stageA_head_relational_grounding_v6.py
LOG_FILE=/home/shadeform/eval2_stageA_head_relational_grounding_v6.log
PID_FILE=/home/shadeform/eval2_stageA_head_relational_grounding_v6.pid

PYTHON=python3

# ── Environment ────────────────────────────────────────────────────────────
export HF_HOME=/home/shadeform/hf_home
export HF_HUB_CACHE=/home/shadeform/hf_home/hub
export HF_LEROBOT_HOME=/home/shadeform/lerobot_cache
export PYTHONPATH=/home/shadeform

# ── Dry-inspect mode ───────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-inspect" ]]; then
  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --init                "${INIT_CKPT}" \
    --base-init           "${BASE_INIT_CKPT}" \
    --dataset             "${DATASET}" \
    --dataset-root        "${DATASET_ROOT}" \
    --out-dir             "${OUT_DIR}" \
    --steps               200 \
    --batch-size          64 \
    --eval-batch-size     64 \
    --num-workers         8 \
    --family-weights      "raw_relative=0.40,clockwise_convention=0.30,scaffolded_relative=0.20,semi_resolved=0.05,simple=0.05" \
    --dry-inspect
  exit 0
fi

# ── Guard: don't double-start ──────────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID=$(cat "${PID_FILE}")
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "ERROR: Stage A v6 is already running (PID=${OLD_PID}). Aborting."
    exit 1
  fi
fi

echo "========================================================"
echo " Eval2 Stage A v6 — clockwise-convention from v5@300"
echo "========================================================"
echo " Init checkpoint : ${INIT_CKPT}"
echo " Base policy     : ${BASE_INIT_CKPT}"
echo " Output          : ${OUT_DIR}"
echo " Log             : ${LOG_FILE}"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null || true

# ── Launch ─────────────────────────────────────────────────────────────────
nohup "${PYTHON}" -u "${TRAIN_SCRIPT}" \
  --init                "${INIT_CKPT}" \
  --base-init           "${BASE_INIT_CKPT}" \
  --dataset             "${DATASET}" \
  --dataset-root        "${DATASET_ROOT}" \
  --out-dir             "${OUT_DIR}" \
  --steps               200 \
  --batch-size          64 \
  --eval-batch-size     64 \
  --lora-rank           16 \
  --lora-alpha          32 \
  --lora-dropout        0.05 \
  --lora-lr             2e-6 \
  --head-lr             2e-5 \
  --weight-decay        1e-10 \
  --warmup-steps        20 \
  --decay-start-step    130 \
  --grad-clip-norm      10.0 \
  --save-every          50 \
  --eval-every          50 \
  --log-every           10 \
  --num-workers         8 \
  --family-weights      "raw_relative=0.40,clockwise_convention=0.30,scaffolded_relative=0.20,semi_resolved=0.05,simple=0.05" \
  --use-amp \
  --seed                52 \
  --wandb \
  --wandb-project       lerobot \
  --wandb-name          "eval2-stageA-v6-clockwise-from-v5-300" \
  "$@" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Stage A v6 started: PID=${PID}"
sleep 5
echo "──── Last 50 lines of log ────"
tail -n 50 "${LOG_FILE}" || true
