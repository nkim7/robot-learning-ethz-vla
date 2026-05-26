#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/shadeform
export TOKENIZERS_PARALLELISM=false

OUT=/home/shadeform/outputs/eval2_stageB_pair_aux_cotrain_v1_from_v6_100
LOG=/home/shadeform/eval2_stageB_pair_aux_cotrain_v1_from_v6_100.log
PID=/home/shadeform/eval2_stageB_pair_aux_cotrain_v1_from_v6_100.pid

mkdir -p "$OUT"

nohup python3 /home/shadeform/train_eval2_stageB_pair_aux_cotrain.py \
  --stage2-base /home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model \
  --stageA /home/shadeform/outputs/eval2_stageA_head_relational_grounding_v6_clockwise_from_v5_300/checkpoints/000100/pretrained_model \
  --dataset robot-learning-group47/eval2_fixed_bowl_fixed_banana_108 \
  --dataset-root /home/shadeform/lerobot_cache/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108 \
  --combo-map /home/shadeform/.cache/lerobot/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108/combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv \
  --out-dir "$OUT" \
  --steps 500 \
  --batch-size 32 \
  --lora-lr 1e-6 \
  --expert-lr 5e-7 \
  --proj-lr 5e-7 \
  --pair-head-lr 2e-5 \
  --pair-weight 0.2 \
  --beta-distill 0.5 \
  --warmup-steps 50 \
  --save-steps 100,200,300,500 \
  --log-every 10 \
  --wandb \
  --wandb-name eval2-stageB-pair-aux-cotrain-v1 \
  > "$LOG" 2>&1 &

echo $! > "$PID"
echo "launched pid=$(cat "$PID")"
echo "log=$LOG"
echo "out=$OUT"
