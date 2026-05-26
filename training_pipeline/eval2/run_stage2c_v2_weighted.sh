#!/usr/bin/env bash
# ============================================================
# Stage 2C v2 — Counterfactual Repair with Weighted Slot Loss
#
# Fixes the middle-default shortcut found in Stage 2C v1:
#   v1 broke the layout shortcut (same frame → different prompts
#   now give different outputs) but the model settled into
#   predicting "middle" for all left_of_X cases because the
#   natural 1:2:1 left:middle:right batch distribution makes
#   middle the lowest-loss default.
#
# Changes vs v1:
#   slot loss weights  [2.0, 1.0, 2.0] for [left, middle, right]
#     → doubles gradient signal for left/right targets
#   train_connector=True
#     → unfreezes the VLM visual-text connector so the model
#       can learn to ground anchor-color positions in the image
#   lr=5e-5  (halved from v1 1e-4 to reduce grad-norm spikes)
#
# Frozen:
#   vision encoder, base VLM weights (except LoRA), action expert
#
# Trainable:
#   VLM q/v LoRA rank-16, target slot head, VLM connector
#
# Gate threshold (success):
#   relative_slot_head_accuracy > 85%
#   blue / green / red all > 75%
#   left_of cases stop collapsing to "middle"
#
# Usage:
#   # Wait for Stage 2C v1 baseline to finish (2000 steps), then:
#   bash ~/run_stage2c_v2_weighted.sh
# ============================================================
set -euo pipefail

STAGE2_CKPT="/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
DATASET="robot-learning-group47/eval2_compositional_augmented"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"
OUT="/home/shadeform/outputs/eval2_stage2c_counterfactual_v2"
TRAIN_SCRIPT="/home/shadeform/train_eval2_counterfactual_repair.py"
GATE_SCRIPT="/home/shadeform/gate_eval2_relational_vlm_repair.py"
PROBE_SCRIPT="/home/shadeform/probe_counterfactual_routing.py"
LOG="${OUT}.log"
GATE_LOG="${OUT}_gates.log"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"
PYTHON="/home/shadeform/lerobot/.venv/bin/python"
PYTHONPATH_VAL="/home/shadeform/lerobot/src:/home/shadeform"

mkdir -p "${OUT}" "$(dirname "${LOG}")"

echo "============================================================"
echo "  Stage 2C v2 — Weighted Counterfactual Relational Repair"
echo "  Base:     ${STAGE2_CKPT}"
echo "  Output:   ${OUT}"
echo "  Steps:    2000  |  save/gate/probe every 500"
echo "  Weights:  left=2.0  middle=1.0  right=2.0"
echo "  Connector: TRAINABLE"
echo "  LR:       5e-5 (reduced from v1)"
echo "============================================================"
echo ""

# ── Verify prerequisites ──────────────────────────────────────────────────────
if [[ ! -f "${STAGE2_CKPT}/model.safetensors" ]]; then
  echo "ERROR: Stage 2 base checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi

# ── Training ──────────────────────────────────────────────────────────────────
echo "[$(date)] Starting Stage 2C v2 training …" | tee -a "${LOG}"
PYTHONPATH="${PYTHONPATH_VAL}" \
  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --init="${STAGE2_CKPT}" \
    --dataset="${DATASET}" \
    --dataset-root="${DATASET_ROOT}" \
    --out-dir="${OUT}" \
    --steps=2000 \
    --batch-base=24 \
    --blue-oversample=2 \
    --lr=5e-5 \
    --lora-rank=16 \
    --train-connector \
    --slot-weights 2.0 1.0 2.0 \
    --save-every=500 \
    --log-every=25 \
    --wandb \
    --wandb-project=lerobot \
    --wandb-name="eval2-stage2c-counterfactual-v2" \
    2>&1 | tee -a "${LOG}"

echo ""
echo "[$(date)] Training complete." | tee -a "${LOG}"

# ── Gate + probe at 500, 1000, 1500, 2000 ─────────────────────────────────────
GATE_PASS_ANY=0

for STEP in 500 1000 1500 2000; do
  CKPT="${OUT}/checkpoints/$(printf '%06d' ${STEP})/pretrained_model"

  if [[ ! -d "${CKPT}" ]]; then
    echo "[$(date)] Checkpoint ${STEP} missing — skipping." | tee -a "${GATE_LOG}"
    continue
  fi

  echo "" | tee -a "${GATE_LOG}"
  echo "══════════════════════════════════════════════════════" | tee -a "${GATE_LOG}"
  echo "  STEP ${STEP}: counterfactual routing probe"         | tee -a "${GATE_LOG}"
  echo "══════════════════════════════════════════════════════" | tee -a "${GATE_LOG}"

  PYTHONPATH="${PYTHONPATH_VAL}" \
    "${PYTHON}" "${PROBE_SCRIPT}" \
      --ckpt="${CKPT}" \
      --all-layouts \
      2>&1 | tee -a "${GATE_LOG}"

  echo "" | tee -a "${GATE_LOG}"
  echo "══════════════════════════════════════════════════════" | tee -a "${GATE_LOG}"
  echo "  STEP ${STEP}: full 24-case gate"                    | tee -a "${GATE_LOG}"
  echo "══════════════════════════════════════════════════════" | tee -a "${GATE_LOG}"

  RESULT_FILE="${OUT}/gate_$(printf '%06d' ${STEP}).txt"
  PYTHONPATH="${PYTHONPATH_VAL}" \
    "${PYTHON}" "${GATE_SCRIPT}" \
      --ckpt="${CKPT}" \
      --base-policy="${STAGE2_CKPT}" \
      --dataset-root="${DATASET_ROOT}" \
      2>&1 | tee "${RESULT_FILE}" | tee -a "${GATE_LOG}"

  if grep -q "GATE_PASS=True" "${RESULT_FILE}"; then
    echo "" | tee -a "${GATE_LOG}"
    echo "╔══════════════════════════════════════════════════╗" | tee -a "${GATE_LOG}"
    echo "║  GATE PASSED at step ${STEP}                         ║" | tee -a "${GATE_LOG}"
    echo "║  relative_slot_head_accuracy > 85%               ║" | tee -a "${GATE_LOG}"
    echo "║  Ready for robot evaluation.                     ║" | tee -a "${GATE_LOG}"
    echo "╚══════════════════════════════════════════════════╝" | tee -a "${GATE_LOG}"
    GATE_PASS_ANY=1
  fi

done

echo "" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
echo "  Stage 2C v2 complete."                                      | tee -a "${GATE_LOG}"
if [[ "${GATE_PASS_ANY}" -eq 1 ]]; then
  echo "  ✓ Gate PASSED — robot evaluation permitted."              | tee -a "${GATE_LOG}"
else
  echo "  ✗ Gate did not pass at any checkpoint."                   | tee -a "${GATE_LOG}"
  echo ""                                                            | tee -a "${GATE_LOG}"
  echo "  Diagnostic checklist:"                                    | tee -a "${GATE_LOG}"
  echo "    1. Are left_of cases still predicting 'middle'?"        | tee -a "${GATE_LOG}"
  echo "       → If yes, increase slot-weights to [3.0, 1.0, 3.0]" | tee -a "${GATE_LOG}"
  echo "    2. Is logit spread < 1.0 for any layout?"               | tee -a "${GATE_LOG}"
  echo "       → LoRA may be saturated; try lora-rank=32"          | tee -a "${GATE_LOG}"
  echo "    3. Are grad norms still > 10 at step 1000?"             | tee -a "${GATE_LOG}"
  echo "       → Reduce LR to 2e-5"                                 | tee -a "${GATE_LOG}"
  echo "    4. Does probe show routing on all 6 layouts?"           | tee -a "${GATE_LOG}"
  echo "       → If yes but gate <85%: need more steps (5000)"     | tee -a "${GATE_LOG}"
fi
echo "============================================================" | tee -a "${GATE_LOG}"
