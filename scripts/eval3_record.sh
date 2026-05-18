#!/usr/bin/env bash
# record.sh — track + run the 180-episode coke-placement Eval 3 recording.
#
# Usage:
#   source record.sh
#   clean_start   # only before starting a fresh dataset
#   record        # records the next episode
#   preview       # show next episode details
#   status        # show progress
#   verify        # verify dataset has video/image feature
#   set_ep N      # manually set counter
#   reset         # reset counter only
#   redo          # remove latest episode and redo it
#
# Requires:
#   export FOLLOWER_PORT=/dev/tty.usbmodem5B141138561
#   export LEADER_PORT=/dev/tty.usbmodem5B141126211

# ---------------------------------------------------------------------
# zsh compatibility
# ---------------------------------------------------------------------

# If sourced from zsh, make arrays 0-indexed like bash.
# This MUST be before array definitions.
if [ -n "${ZSH_VERSION:-}" ]; then
    setopt KSH_ARRAYS
fi

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

STATE_FILE="$HOME/.lerobot_eval3_video_progress"
TOTAL_EPISODES=180

DATASET_REPO_ID="robot-learning-group47/eval3_toy_video"
DATASET_ROOT="data/eval3_toy_video"

TRIPLES=(
    "T0 O0 Y0"
    "T1 O1 Y1"
    "T2 O2 Y2"
    "T3 O3 Y3"
    "T4 O4 Y4"
    "T0 O2 Y4"
    "T1 O3 Y0"
    "T2 O4 Y1"
    "T3 O0 Y2"
    "T4 O1 Y3"
)

ORDERS=(
    "Taylor Obama Yann"
    "Taylor Yann Obama"
    "Obama Taylor Yann"
    "Obama Yann Taylor"
    "Yann Taylor Obama"
    "Yann Obama Taylor"
)

PROMPTS=(
    "Taylor Swift"
    "Barack Obama"
    "Yann LeCun"
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_prompt_enter() {
    local msg="$1"
    printf "%s" "$msg"
    read -r _
}

_prompt_value() {
    local msg="$1"
    local varname="$2"
    printf "%s" "$msg"
    read -r "$varname"
}

_episode_info() {
    local ep="$1"

    EP="$ep"
    TRIPLE_IDX=$(( ep / 18 ))
    ORDER_IDX=$(( (ep % 18) / 3 ))
    PROMPT_IDX=$(( ep % 3 ))
    SETUP_NUM=$(( ep / 3 ))

    if [ "$PROMPT_IDX" -eq 0 ]; then
        IS_NEW_SETUP=1
    else
        IS_NEW_SETUP=0
    fi

    TRIPLE="${TRIPLES[$TRIPLE_IDX]}"
    ORDER="${ORDERS[$ORDER_IDX]}"
    PROMPT="${PROMPTS[$PROMPT_IDX]}"

    local t_img=""
    local o_img=""
    local y_img=""

    read -r t_img o_img y_img <<< "$TRIPLE"
    read -r LEFT CENTER RIGHT <<< "$ORDER"

    case "$LEFT" in
        Taylor) LEFT_IMG="$t_img" ;;
        Obama) LEFT_IMG="$o_img" ;;
        Yann) LEFT_IMG="$y_img" ;;
        *) LEFT_IMG="" ;;
    esac

    case "$CENTER" in
        Taylor) CENTER_IMG="$t_img" ;;
        Obama) CENTER_IMG="$o_img" ;;
        Yann) CENTER_IMG="$y_img" ;;
        *) CENTER_IMG="" ;;
    esac

    case "$RIGHT" in
        Taylor) RIGHT_IMG="$t_img" ;;
        Obama) RIGHT_IMG="$o_img" ;;
        Yann) RIGHT_IMG="$y_img" ;;
        *) RIGHT_IMG="" ;;
    esac
}

_current_ep() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo 0
    fi
}

_print_episode_block() {
    local ep="$1"
    local label="$2"

    _episode_info "$ep"

    echo "=============================================================="
    echo "  $label   Episode $EP / $TOTAL_EPISODES"
    echo "          Setup $((SETUP_NUM + 1))/60 | Prompt $((PROMPT_IDX + 1))/3"
    echo "--------------------------------------------------------------"
    echo "  Dataset repo: $DATASET_REPO_ID"
    echo "  Dataset root: $DATASET_ROOT"
    echo "--------------------------------------------------------------"
    echo "  Triple #$TRIPLE_IDX: $TRIPLE"
    echo "  Order  #$ORDER_IDX:"
    printf "    LEFT   = %-7s (%s)\n" "$LEFT" "$LEFT_IMG"
    printf "    CENTER = %-7s (%s)\n" "$CENTER" "$CENTER_IMG"
    printf "    RIGHT  = %-7s (%s)\n" "$RIGHT" "$RIGHT_IMG"
    echo "--------------------------------------------------------------"
    echo "  TASK: Place the coke on $PROMPT."
    echo "=============================================================="
}

# ---------------------------------------------------------------------
# User commands
# ---------------------------------------------------------------------

status() {
    local ep
    local pct

    ep=$(_current_ep)
    pct=$(( ep * 100 / TOTAL_EPISODES ))

    echo "Progress: $ep / $TOTAL_EPISODES episodes done ($pct%)"
    echo "Remaining: $(( TOTAL_EPISODES - ep ))"
    echo "State file: $STATE_FILE"
    echo "Dataset repo: $DATASET_REPO_ID"
    echo "Dataset root: $DATASET_ROOT"
}

preview() {
    local ep
    ep=$(_current_ep)

    if [ "$ep" -ge "$TOTAL_EPISODES" ]; then
        echo "All $TOTAL_EPISODES episodes already complete."
        return 0
    fi

    _print_episode_block "$ep" "NEXT:"
}

set_ep() {
    if [ -z "${1:-}" ]; then
        echo "Usage: set_ep <episode_number 0-$TOTAL_EPISODES>" >&2
        return 1
    fi

    case "$1" in
        ''|*[!0-9]*)
            echo "Usage: set_ep <episode_number 0-$TOTAL_EPISODES>" >&2
            return 1
            ;;
    esac

    if [ "$1" -gt "$TOTAL_EPISODES" ]; then
        echo "Value $1 exceeds max $TOTAL_EPISODES" >&2
        return 1
    fi

    echo "$1" > "$STATE_FILE"
    echo "Counter set to $1"
}

reset() {
    rm -f "$STATE_FILE"
    echo "Counter reset. Next episode: 0"
}

clean_start() {
    local ans=""

    echo "This will delete:"
    echo "  $DATASET_ROOT"
    echo "  $STATE_FILE"
    echo ""

    printf "Type DELETE to continue: "
    read -r ans

    if [ "$ans" != "DELETE" ]; then
        echo "Aborted."
        return 1
    fi

    rm -rf "$DATASET_ROOT"
    rm -f "$STATE_FILE"

    echo "Clean start ready."
    echo "Next: record"
}

verify() {
    python - <<PY
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "$DATASET_REPO_ID"
root = "$DATASET_ROOT"

ds = LeRobotDataset(repo_id, root=root)

print("frames:", len(ds))
print("episodes:", len(ds.meta.episodes))
print()
print("Features:")
for k, v in ds.features.items():
    print(k, ":", v)

image_keys = [k for k in ds.features if "image" in k.lower()]
print()
print("Image/video keys:", image_keys)

if not image_keys:
    raise SystemExit("ERROR: no image/video feature found. Do not continue recording.")

media = []
for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.png", "*.jpg", "*.jpeg", "*.webp"]:
    media.extend(Path(root).rglob(ext))

print("media files:", len(media))
for p in media[:10]:
    print(" ", p)
PY
}

redo() {
    local ep
    local redo_ep
    local ans=""

    ep=$(_current_ep)

    if [ "$ep" -le 0 ]; then
        echo "No previous episode to redo. Counter is at $ep."
        return 1
    fi

    redo_ep=$(( ep - 1 ))

    echo "Redo requested."
    echo "Current counter: $ep"
    echo "Last recorded episode: $redo_ep"
    echo ""
    echo "This will:"
    echo "  1. Remove the latest recorded episode from $DATASET_ROOT"
    echo "  2. Set counter back to $redo_ep"
    echo "  3. Let you record episode $redo_ep again"
    echo ""

    _print_episode_block "$redo_ep" "REDO:"

    printf "Type REDO to delete the latest episode and redo it: "
    read -r ans

    if [ "$ans" != "REDO" ]; then
        echo "Aborted."
        return 1
    fi

    echo ""
    echo "Removing latest episode..."

    python remove.py

    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo ""
        echo "remove.py failed. Counter was NOT changed."
        return "$rc"
    fi

    echo "$redo_ep" > "$STATE_FILE"

    echo ""
    echo "✓ Latest episode removed."
    echo "✓ Counter set back to $redo_ep."
    echo ""
    echo "Now run:"
    echo "  record"
}

record() {
    local ep
    local task
    local next_ep
    local resume_flag

    ep=$(_current_ep)

    if [ "$ep" -ge "$TOTAL_EPISODES" ]; then
        echo "✓ All $TOTAL_EPISODES episodes complete. Nothing left to record."
        return 0
    fi

    if [ -z "${FOLLOWER_PORT:-}" ] || [ -z "${LEADER_PORT:-}" ]; then
        echo "ERROR: FOLLOWER_PORT and/or LEADER_PORT not set." >&2
        echo "Run:"
        echo "  export FOLLOWER_PORT=/dev/tty.usbmodem5B141138561"
        echo "  export LEADER_PORT=/dev/tty.usbmodem5B141126211"
        return 1
    fi

    _episode_info "$ep"

    task="Place the coke on $PROMPT."
    next_ep=$(( ep + 1 ))
    resume_flag=true

    if [ "$ep" -eq 0 ]; then
        resume_flag=false
    fi

    echo ""
    _print_episode_block "$ep" "NOW:"

    if [ "$IS_NEW_SETUP" -eq 1 ]; then
        echo ""
        echo ">>> NEW TABLE SETUP — arrange the 3 images exactly as shown above. <<<"
    else
        echo ""
        echo "(Same physical setup as previous episode — only reset the coke can.)"
    fi

    echo ""
    if [ "$next_ep" -lt "$TOTAL_EPISODES" ]; then
        local n_prompt_idx=$(( next_ep % 3 ))
        local n_setup_num=$(( next_ep / 3 ))

        if [ "$n_prompt_idx" -eq 0 ]; then
            echo "AFTER THIS: new setup coming: Setup $((n_setup_num + 1))/60."
        else
            local n_prompt="${PROMPTS[$n_prompt_idx]}"
            echo "AFTER THIS: same setup, next task = Place the coke on $n_prompt."
        fi
    else
        echo "AFTER THIS: done — final episode."
    fi

    echo ""
    echo "Resume flag: $resume_flag"
    printf "Press ENTER to start recording, Ctrl+C to abort: "
    read -r _

    lerobot-record \
        --robot.type=so101_follower \
        --robot.port="$FOLLOWER_PORT" \
        --robot.calibration_dir=calibration/so101_group47/follower \
        --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 15, "warmup_s": 5}}' \
        --teleop.type=so101_leader \
        --teleop.port="$LEADER_PORT" \
        --teleop.calibration_dir=calibration/so101_group47/leader \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --dataset.root="$DATASET_ROOT" \
        --dataset.fps=15 \
        --dataset.num_episodes=1 \
        --dataset.episode_time_s=10 \
        --dataset.reset_time_s=2 \
        --dataset.single_task="$task" \
        --dataset.video=true \
        --dataset.push_to_hub=false \
        --display_data=true \
        --resume="$resume_flag"

    local rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "$next_ep" > "$STATE_FILE"
        echo ""
        echo "✓ Episode $ep recorded. Counter advanced to $next_ep / $TOTAL_EPISODES."

        if [ "$ep" -eq 0 ]; then
            echo ""
            echo "Verifying video/image feature after first episode..."
            verify
        fi
    else
        echo ""
        echo "✗ Recording exited with code $rc."
        echo "Counter NOT advanced. Still at episode $ep."
        echo "Fix the issue, then run record again to retry this episode."
        return "$rc"
    fi
}

# Auto-run only when executed with bash, not when sourced.
if [ -n "${BASH_VERSION:-}" ]; then
    if [ "${BASH_SOURCE[0]}" = "$0" ]; then
        record
    fi
fi