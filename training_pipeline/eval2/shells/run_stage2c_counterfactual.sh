#!/usr/bin/env bash
# ============================================================
# Stage 2C — Counterfactual Paired Relational Repair
#
# Fixes the shortcut-learning failure of Stage 2B:
#   Same image frame appears with ALL 4 relative prompt variants
#   (left_of/right_of every anchor) so the model cannot use
#   episode-level image cues — it MUST route by prompt semantics.
#
# Key changes vs Stage 2B:
#   - Slot loss ONLY (action loss removed; it conflicted with
#     counterfactual targets vs original demo actions)
#   - Blue anchor oversampled 2x (was 0% in all prior gates)
#   - Batch built per-step, not from a pre-shuffled index
#   - Steps reduced to 2000; gate every 500 steps
#
# Gate threshold: relative_slot_head_accuracy > 85%
#                 AND blue/green/red all > 75%
#
# Usage:
#   # Run sanity check first (optional but recommended):
#   bash ~/run_stage2c_counterfactual.sh --sanity-only
#
#   # Full run:
#   bash ~/run_stage2c_counterfactual.sh
# ============================================================
set -euo pipefail

SANITY_ONLY=0
for arg in "$@"; do [[ "$arg" == "--sanity-only" ]] && SANITY_ONLY=1; done

STAGE2_CKPT="/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
STAGE2B_2K="/home/shadeform/outputs/eval2_stage2b_vlm_relational_repair_from_20k/checkpoints/002000/pretrained_model"
DATASET="robot-learning-group47/eval2_compositional_augmented"
DATASET_ROOT="/home/shadeform/lerobot_cache/${DATASET}"
OUT="/home/shadeform/outputs/eval2_stage2c_counterfactual"
TRAIN_SCRIPT="/home/shadeform/train_eval2_counterfactual_repair.py"
GATE_SCRIPT="/home/shadeform/gate_eval2_relational_vlm_repair.py"
SANITY_SCRIPT="/home/shadeform/sanity_check_logits.py"
LOG="${OUT}.log"
GATE_LOG="${OUT}_gates.log"

export HF_HOME="/home/shadeform/hf_home"
export HF_HUB_CACHE="/home/shadeform/hf_home/hub"
export HF_LEROBOT_HOME="/home/shadeform/lerobot_cache"
PYTHON="/home/shadeform/lerobot/.venv/bin/python"
PYTHONPATH_VAL="/home/shadeform/lerobot/src:/home/shadeform"

mkdir -p "${OUT}" "$(dirname "${LOG}")"

echo "============================================================"
echo "  Stage 2C — Counterfactual Relational Repair"
echo "  Base:     ${STAGE2_CKPT}"
echo "  Dataset:  ${DATASET}"
echo "  Output:   ${OUT}"
echo "  Steps:    2000  |  save/gate every 500"
echo "  Design:   same-frame quads, slot loss only, blue 2x"
echo "============================================================"
echo ""

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "[$(date)] Running logit sanity check …"
PYTHONPATH="${PYTHONPATH_VAL}" \
  "${PYTHON}" "${SANITY_SCRIPT}" 2>&1 | tee -a "${GATE_LOG}"
echo ""

if [[ "${SANITY_ONLY}" -eq 1 ]]; then
  echo "Sanity-only mode — exiting before training."
  exit 0
fi

# ── Verify prerequisites ──────────────────────────────────────────────────────
if [[ ! -f "${STAGE2_CKPT}/model.safetensors" ]]; then
  echo "ERROR: Stage 2 checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

# ── Dry run: verify batch shape ───────────────────────────────────────────────
echo "[$(date)] Dry run — verifying batch statistics …"
PYTHONPATH="${PYTHONPATH_VAL}" \
  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --init="${STAGE2_CKPT}" \
    --dataset="${DATASET}" \
    --dataset-root="${DATASET_ROOT}" \
    --out-dir="${OUT}" \
    --dry-run 2>&1 | tee -a "${LOG}"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
echo "[$(date)] Starting Stage 2C training …"
PYTHONPATH="${PYTHONPATH_VAL}" \
  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --init="${STAGE2_CKPT}" \
    --dataset="${DATASET}" \
    --dataset-root="${DATASET_ROOT}" \
    --out-dir="${OUT}" \
    --steps=2000 \
    --batch-base=24 \
    --blue-oversample=2 \
    --lr=1e-4 \
    --lora-rank=16 \
    --save-every=500 \
    --log-every=25 \
    --wandb \
    --wandb-project=lerobot \
    --wandb-name="eval2-stage2c-counterfactual" \
    2>&1 | tee -a "${LOG}"
echo ""
echo "[$(date)] Training complete."

# ── Gate at 500, 1000, 1500, 2000 ─────────────────────────────────────────────
echo "[$(date)] Running gates …" | tee -a "${GATE_LOG}"
GATE_PASS_ANY=0

for STEP in 500 1000 1500 2000; do
  CKPT="${OUT}/checkpoints/$(printf '%06d' ${STEP})/pretrained_model"
  RESULT_FILE="${OUT}/gate_$(printf '%06d' ${STEP}).txt"

  if [[ ! -d "${CKPT}" ]]; then
    echo "[$(date)] Checkpoint ${STEP} not found — skipping gate." | tee -a "${GATE_LOG}"
    continue
  fi

  echo "" | tee -a "${GATE_LOG}"
  echo "=== GATE ${STEP} ===" | tee -a "${GATE_LOG}"
  PYTHONPATH="${PYTHONPATH_VAL}" \
    "${PYTHON}" "${GATE_SCRIPT}" \
      --ckpt="${CKPT}" \
      --base-policy="${STAGE2_CKPT}" \
      --dataset-root="${DATASET_ROOT}" \
      2>&1 | tee "${RESULT_FILE}" | tee -a "${GATE_LOG}"

  # Extract pass/fail
  if grep -q "GATE_PASS=True" "${RESULT_FILE}"; then
    echo "" | tee -a "${GATE_LOG}"
    echo "╔══════════════════════════════════════════════════╗" | tee -a "${GATE_LOG}"
    echo "║  GATE PASSED at step ${STEP} — ready for robot test  ║" | tee -a "${GATE_LOG}"
    echo "╚══════════════════════════════════════════════════╝" | tee -a "${GATE_LOG}"
    GATE_PASS_ANY=1
  fi
done

echo "" | tee -a "${GATE_LOG}"
echo "============================================================" | tee -a "${GATE_LOG}"
echo "  Stage 2C complete." | tee -a "${GATE_LOG}"
if [[ "${GATE_PASS_ANY}" -eq 1 ]]; then
  echo "  ✓ At least one gate PASSED — proceed to robot evaluation." | tee -a "${GATE_LOG}"
  echo "  Threshold: relative_slot_acc > 85% AND all anchors > 75%" | tee -a "${GATE_LOG}"
else
  echo "  ✗ No gate passed — do NOT robot-test." | tee -a "${GATE_LOG}"
  echo "  Next steps:" | tee -a "${GATE_LOG}"
  echo "    1. Check ${GATE_LOG} for per-anchor accuracy trends." | tee -a "${GATE_LOG}"
  echo "    2. If blue still 0%: increase --blue-oversample to 3." | tee -a "${GATE_LOG}"
  echo "    3. If all anchors >50% but <75%: extend to --steps=5000." | tee -a "${GATE_LOG}"
  echo "    4. If slot_loss not decreasing: check LoRA is not frozen." | tee -a "${GATE_LOG}"
fi
echo "============================================================" | tee -a "${GATE_LOG}"
