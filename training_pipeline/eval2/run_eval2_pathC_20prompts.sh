#!/usr/bin/env bash
# ============================================================
# Eval 2 Path C — 20 prompt sweep
# Runs lerobot-record for each prompt (1 episode each).
# Prints all prompts upfront so you know what's coming.
# ============================================================
set -euo pipefail

POLICY_PATH="${1:-models/smolvla-eval2-pathC/step8000}"
PORT="${2:-/dev/tty.usbmodem5B140319121}"
CALIB="calibration/so101_group47/follower"
CAMERAS='{"camera1":{"type":"opencv_crop","index_or_path":0,"source_width":640,"source_height":480,"roi_x":215,"roi_y":0,"roi_size":425,"width":256,"height":256,"fps":15,"warmup_s":1}}'
EPISODES_PER_PROMPT=1
EPISODE_TIME=20

# ── 20 prompts ───────────────────────────────────────────────
PROMPTS=(
  # COLOR (3)
  "Put the banana in the red colored bowl."
  "Put the banana in the green colored bowl."
  "Put the banana in the blue colored bowl."
  # ORDINAL FROM LEFT (3)
  "Put the banana into the 1st bowl from the left from the robot perspective."
  "Put the banana into the 2nd bowl from the left from the robot perspective."
  "Put the banana into the 3rd bowl from the left from the robot perspective."
  # ORDINAL FROM RIGHT (3)
  "Put the banana into the 1st bowl from the right from the robot perspective."
  "Put the banana into the 2nd bowl from the right from the robot perspective."
  "Put the banana into the 3rd bowl from the right from the robot perspective."
  # MIDDLE (1)
  "Put the banana into the middle bowl from the robot perspective."
  # RELATIVE POSITION (4)
  "Put the banana into the bowl on the left of the red bowl from the robot perspective."
  "Put the banana into the bowl on the right of the red bowl from the robot perspective."
  "Put the banana into the bowl on the left of the green bowl from the robot perspective."
  "Put the banana into the bowl on the right of the green bowl from the robot perspective."
  # NEGATION (6)
  "Put the banana into the bowl that is not red and not green."
  "Put the banana into the bowl that is not red and not blue."
  "Put the banana into the bowl that is not green and not blue."
  "Put the banana into the bowl that is not green and not red."
  "Put the banana into the bowl that is not blue and not red."
  "Put the banana into the bowl that is not blue and not green."
)

TOTAL=${#PROMPTS[@]}
STEP=$(basename "$POLICY_PATH")  # e.g. step8000

# ── Print plan ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         EVAL 2 PATH C — 20 PROMPT SWEEP                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Policy:   %-51s║\n" "$POLICY_PATH"
printf "║  Port:     %-51s║\n" "$PORT"
printf "║  Episodes: %-51s║\n" "$EPISODES_PER_PROMPT per prompt  ($TOTAL total runs)"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  PROMPTS:                                                    ║"
for i in "${!PROMPTS[@]}"; do
  NUM=$(( i + 1 ))
  printf "║  %2d. %-57s║\n" "$NUM" "${PROMPTS[$i]:0:57}"
done
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Enter to start, Ctrl+C to abort."
read -r

# ── Run each prompt ──────────────────────────────────────────
for i in "${!PROMPTS[@]}"; do
  NUM=$(( i + 1 ))
  PROMPT="${PROMPTS[$i]}"
  SLUG=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | cut -c1-40)
  REPO_ID="local/eval_pathC_${STEP}_${NUM}_${SLUG}"
  CACHE_DIR="$HOME/.cache/huggingface/lerobot/local/eval_pathC_${STEP}_${NUM}_${SLUG}"

  echo ""
  echo "┌──────────────────────────────────────────────────────────────┐"
  printf "│  PROMPT %2d / %2d                                              │\n" "$NUM" "$TOTAL"
  echo "│                                                              │"
  # Word-wrap prompt at 60 chars
  echo "$PROMPT" | fold -s -w 60 | while IFS= read -r line; do
    printf "│  %-62s│\n" "$line"
  done
  echo "│                                                              │"
  printf "│  Dataset: %-53s│\n" "$(basename "$REPO_ID")"
  echo "└──────────────────────────────────────────────────────────────┘"

  # Clean up any leftover dataset dir from a previous failed run
  rm -rf "$CACHE_DIR"

  lerobot-record \
    --robot.type=so101_follower \
    --robot.id=my_follower \
    --robot.port="$PORT" \
    --robot.calibration_dir="$CALIB" \
    --robot.cameras="$CAMERAS" \
    --policy.path="$POLICY_PATH" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.push_to_hub=false \
    --dataset.fps=15 \
    --dataset.num_episodes="$EPISODES_PER_PROMPT" \
    --dataset.episode_time_s="$EPISODE_TIME" \
    --dataset.reset_time_s=0 \
    --dataset.single_task="$PROMPT" \
    --display_data=true

  echo "  ✅ Prompt $NUM done."
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL $TOTAL PROMPTS COMPLETE                                      ║"
echo "║  Results in: ~/.cache/huggingface/lerobot/local/            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
