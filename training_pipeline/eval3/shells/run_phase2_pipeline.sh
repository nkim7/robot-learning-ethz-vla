#!/bin/bash
# Wait for the current GPU job to finish, then generate dataset + train phase2.
set -e
cd ~/eval3_audit

LEROBOT_PID=36389

echo "=== Phase 2 pipeline — waiting for PID $LEROBOT_PID to finish ==="
while kill -0 $LEROBOT_PID 2>/dev/null; do
    echo "  [$(date +%H:%M:%S)] still running, sleeping 60s..."
    sleep 60
done
echo "[$(date +%H:%M:%S)] GPU free — starting phase 2 pipeline"

echo ""
echo "=== Step 1: Generate 150 synthetic scene images ==="
python3 gen_phase2_dataset.py

echo ""
echo "=== Step 2: Train SmolVLM2 LoRA phase 2 ==="
python3 train_eval3_phase2.py \
    --manifest ./data/phase2_manifest.jsonl \
    --output_dir ./checkpoints/phase2 \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_rank 16

echo ""
echo "=== Pipeline complete ==="
