#!/usr/bin/env bash
# ============================================================
# Run 2 — eval2_strict10_frozenbackbone_expert_full_v1
#
# Purpose:
#   Test whether the strict10 VQA backbone alone gives the
#   Eval1 action expert enough target-slot information to learn
#   Eval2 target routing with ordinary behavior cloning.
#
# Design:
#   Backbone:  robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged
#              FULLY FROZEN (train_expert_only=true + freeze_vision_encoder=true)
#   Expert:    eval1_final_final  (warm-start, trainable)
#   Dataset:   robot-learning-group47/eval2_compositional_augmented
#   Loss:      L = L_action  (no auxiliary target-slot loss)
#   LR:        5e-6  (NOT 1e-4 — use_policy_training_preset=false)
#   Steps:     40 000  (diagnostic: check probes at 2k/4k/6k/10k first)
#
# This is an ablation baseline.
# If this run shows real prompt-conditioned action separation,
# the strict10 backbone alone was sufficient.
# If it collapses, Stage 2 (auxiliary loss + grouped sampler) is needed.
#
# Probe plan (run after every 2k checkpoint):
#   python probe_action_conditioning.py \
#     --model /data/lerobot_outputs/train/eval2_strict10_frozenbackbone_expert_full_v1/checkpoints/002000/pretrained_model
#
# Usage:
#   bash run_train_eval2_strict10_frozenbackbone_v1.sh
#   SKIP_MERGE=1 bash run_train_eval2_strict10_frozenbackbone_v1.sh  # if init already exists
# ============================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/lerobot_outputs/train/smolvla-eval2-strict10frozenbackbone-init}"
DATASET="${DATASET:-robot-learning-group47/eval2_compositional_augmented}"
RUN_NAME="${RUN_NAME:-eval2_strict10_frozenbackbone_expert_full_v1}"
OUT="${OUT:-/data/lerobot_outputs/train/${RUN_NAME}}"
LOG="${LOG:-/data/lerobot_outputs/${RUN_NAME}.log}"
DATASET_ROOT="${DATASET_ROOT:-/data/lerobot_cache/${DATASET}}"

# Dataset rename: eval2 stores 'front', SmolVLA policy expects 'camera1'
RENAME_MAP='{"observation.images.front":"observation.images.camera1"}'
EMPTY_CAMERAS=2

export HF_HOME="${HF_HOME:-/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/hf_home/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot_cache}"

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DATASET_ROOT}"

echo "============================================================"
echo "  Run 2: eval2_strict10_frozenbackbone_expert_full_v1"
echo "  Init:       ${INIT_CHECKPOINT}"
echo "  Dataset:    ${DATASET}"
echo "  Output:     ${OUT}"
echo "  Log:        ${LOG}"
echo "  Backbone:   FROZEN (train_expert_only=true)"
echo "  Expert LR:  5e-6"
echo "  Sched:      cosine 1k warmup → 40k decay → 5e-8"
echo "  Steps:      40 000"
echo "  No aux loss. No grouped sampler. No LoRA."
echo "============================================================"

# ── Step 1: Merge strict10 backbone into eval1 expert ─────────────────────────
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  echo "[$(date)] SKIP_MERGE=1 — skipping merge step."
else
  if [[ -f "${INIT_CHECKPOINT}/model.safetensors" ]]; then
    echo "[$(date)] Init checkpoint already exists at ${INIT_CHECKPOINT} — skipping merge."
  else
    echo "[$(date)] Building Run 2 init checkpoint (strict10 backbone + eval1 expert) …"
    python3 merge_strict10_into_eval1.py \
      --hf-home "${HF_HOME}/hub" \
      --out-dir "${INIT_CHECKPOINT}"
    echo "[$(date)] Merge complete → ${INIT_CHECKPOINT}"
  fi
fi

# ── Step 2: Verify init checkpoint exists ─────────────────────────────────────
if [[ ! -f "${INIT_CHECKPOINT}/model.safetensors" ]]; then
  echo "ERROR: Init checkpoint not found at ${INIT_CHECKPOINT}/model.safetensors"
  echo "       Run without SKIP_MERGE=1, or check merge_strict10_into_eval1.py output."
  exit 1
fi
echo "[$(date)] Init checkpoint OK: ${INIT_CHECKPOINT}"

# ── Step 3: Print trainable parameter count ───────────────────────────────────
echo ""
echo "[$(date)] Checking trainable parameter count …"
python3 - <<'PARAM_CHECK'
import sys
sys.path.insert(0, "lerobot/src")
import os, torch
from safetensors.torch import load_file

INIT = os.environ.get("INIT_CHECKPOINT",
    "/data/lerobot_outputs/train/smolvla-eval2-strict10frozenbackbone-init")

sd = load_file(f"{INIT}/model.safetensors")

VLM_PREFIX    = "model.vlm_with_expert.vlm."
EXPERT_SUBSTR = ["lm_expert", "state_proj", "action_in_proj",
                 "action_out_proj", "action_time_mlp"]

backbone  = {k: v for k, v in sd.items() if k.startswith(VLM_PREFIX)}
trainable = {k: v for k, v in sd.items()
             if any(s in k for s in EXPERT_SUBSTR)}
other     = {k: v for k, v in sd.items()
             if not k.startswith(VLM_PREFIX)
             and not any(s in k for s in EXPERT_SUBSTR)}

total_p    = sum(v.numel() for v in sd.values())
frozen_p   = sum(v.numel() for v in backbone.values())
train_p    = sum(v.numel() for v in trainable.values())
other_p    = sum(v.numel() for v in other.values())

print(f"  Total params:            {total_p/1e6:8.2f} M")
print(f"  Backbone (FROZEN):       {frozen_p/1e6:8.2f} M   ({len(backbone)} tensors)")
print(f"  Expert+proj (trainable): {train_p/1e6:8.2f} M   ({len(trainable)} tensors)")
print(f"  Other:                   {other_p/1e6:8.2f} M   ({len(other)} tensors)")

# Top trainable groups
from collections import Counter
def group(keys, depth=4):
    counts = Counter()
    for k in keys:
        parts = k.split(".")
        counts[".".join(parts[:depth])] += 1
    return counts.most_common(8)

print()
print("  Top trainable key groups:")
for name, cnt in group(trainable.keys()):
    print(f"    {name:55s}  {cnt} tensors")
print()
print("  Top frozen key groups:")
for name, cnt in group(backbone.keys()):
    print(f"    {name:55s}  {cnt} tensors")

# Safety checks
if train_p < 1e6:
    print("ERROR: trainable < 1M — expert may be frozen!")
    sys.exit(1)
if frozen_p < 100e6:
    print("ERROR: frozen backbone < 100M — backbone may not be loaded!")
    sys.exit(1)
print()
print("  ✅ Parameter split looks correct.")
PARAM_CHECK

echo ""
echo "[$(date)] Launch checks:"
echo "  ✅ Backbone:     smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
echo "  ✅ Expert:       eval1_final_final (warm-start)"
echo "  ✅ train_expert_only=true  → VQA backbone FROZEN"
echo "  ✅ freeze_vision_encoder=true  → vision encoder FROZEN"
echo "  ✅ use_policy_training_preset=false  → LR will NOT be overwritten to 1e-4"
echo "  ✅ LR: 5e-6 (peak), 5e-8 (decay)"
echo "  ✅ No auxiliary loss"
echo "  ✅ No grouped sampler"
echo "  ✅ No VLM LoRA"
echo "  ✅ Dataset: eval2_compositional_augmented (original Eval2 prompts)"
echo ""

# ── Step 4: Train ─────────────────────────────────────────────────────────────
echo "[$(date)] Starting Run 2 training …"
lerobot-train \
  --policy.path="${INIT_CHECKPOINT}" \
  \
  --policy.device=cuda \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.use_amp=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.resize_imgs_with_padding='[256,256]' \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.push_to_hub=false \
  \
  --dataset.repo_id="${DATASET}" \
  --dataset.revision=main \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}' \
  --dataset.use_imagenet_stats=true \
  --dataset.video_backend=torchcodec \
  --rename_map="${RENAME_MAP}" \
  \
  --batch_size=32 \
  --num_workers=12 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  \
  --use_policy_training_preset=false \
  \
  --steps=40000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=40000 \
  \
  --optimizer.type=adamw \
  --optimizer.lr=5e-6 \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10 \
  --optimizer.betas='[0.9,0.95]' \
  --optimizer.eps=1e-8 \
  \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=40000 \
  --scheduler.peak_lr=5e-6 \
  --scheduler.decay_lr=5e-8 \
  \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.run_name="${RUN_NAME}" \
  \
  --output_dir="${OUT}" \
  --job_name="${RUN_NAME}" \
  2>&1 | tee -a "${LOG}"

echo ""
echo "[$(date)] Training complete."
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PROBE INSTRUCTIONS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Run action-conditioning probe on each 2k checkpoint:"
echo ""
echo "  for STEP in 002000 004000 006000 010000 014000 020000; do"
echo "    CKPT=${OUT}/checkpoints/\${STEP}/pretrained_model"
echo "    echo \"--- Step \${STEP} ---\""
echo "    python probe_action_conditioning.py --model \"\${CKPT}\""
echo "  done"
echo ""
echo "  KEY SIGNALS TO WATCH:"
echo "    - shoulder_pan range across color prompts (red/green/blue)"
echo "    - shoulder_pan range across ordinal prompts (1st/2nd/3rd from left)"
echo "    - If pan range < 5°  → collapsed (model ignoring prompt)"
echo "    - If pan range > 15° → grounding (model routing to correct bowl)"
echo ""
echo "  INTERPRETATION:"
echo "    ✅ Run succeeds → strict10 backbone alone is sufficient"
echo "       → no need for auxiliary target-slot loss"
echo "    ❌ Run fails    → expert cannot read backbone on its own"
echo "       → proceed to Stage 2 (aux loss + grouped sampler)"
echo ""
echo "  Checkpoints: ${OUT}/checkpoints/"
echo "  Log:         ${LOG}"
echo "════════════════════════════════════════════════════════════"
