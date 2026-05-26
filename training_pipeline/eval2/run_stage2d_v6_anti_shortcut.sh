#!/usr/bin/env bash
# Approach D2_v6 — LoRA-only anti-shortcut contrastive relative sampling
# LoRA LR 2e-5 | hold peak to step 1200 | 2000 steps | save 200,500,1000,1500,2000
# Relative: 60% anti-shortcut (right_of_left/left_of_right->middle), 40% edge
# Distillation: beta=0.5 on preservation samples only, 0 on relative
set -euo pipefail

REMOTE_DIR="/home/shadeform"
LOG_FILE="${REMOTE_DIR}/stage2d_v6.log"
PID_FILE="${REMOTE_DIR}/stage2d_v6.pid"
TRAIN_SCRIPT="${REMOTE_DIR}/train_eval2_stage2d_v6_anti_shortcut.py"

echo "=== Approach D2_v6: LoRA-only anti-shortcut (lora=2e-5, anti_frac=0.6, beta=0.5 pres-only) ==="
echo "Remote log: ${LOG_FILE}"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "--- DRY RUN ---"
    python3 "${TRAIN_SCRIPT}" --dry-run --batch-size 32 --steps 2000
    exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Killing old run PID=${OLD_PID}..."
        kill "${OLD_PID}" || true
        sleep 2
    fi
fi

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits

echo "Starting D2_v6 training..."
nohup python3 -u "${TRAIN_SCRIPT}" \
    --init /home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model \
    --out-dir /home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_d2_v6_anti_shortcut \
    --steps 2000 \
    --batch-size 32 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-lr 2e-5 \
    --lora-min-lr 1e-7 \
    --beta 0.5 \
    --anti-shortcut-frac 0.60 \
    --weight-decay 1e-10 \
    --grad-clip 10.0 \
    --warmup-steps 100 \
    --hold-steps 1200 \
    --frame-min 5 \
    --frame-max 80 \
    --save-steps 200,500,1000,1500,2000 \
    --log-every 25 \
    --use-amp \
    --seed 42 \
    --wandb \
    --wandb-project lerobot \
    --wandb-name eval2-stage2d-v6-antishortcut-lora2e5 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
echo "Training started: PID=${TRAIN_PID}"
echo "Tail log:  tail -f ${LOG_FILE}"

sleep 5
tail -n 5 "${LOG_FILE}" 2>/dev/null || true
