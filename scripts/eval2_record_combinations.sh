#!/usr/bin/env bash
set -euo pipefail

cd ~/robot-learning-ethz-vla
source .venv/bin/activate

export LEADER_PORT=/dev/tty.usbmodem5B141126211
export FOLLOWER_PORT=/dev/tty.usbmodem5B141138561

# ============================================================
# Eval 2 combination recording
#
# Records 18 combinations:
#   6 bowl orders x 3 target colors
#
# For each combination:
#   - creates one folder under data/eval2_fixed_combinations/
#   - records 10 episodes
#   - before the FIRST episode: robot goes to fixed initial pose
#   - after EVERY episode: robot immediately goes back to fixed initial pose
#   - after each episode: ask keep/delete
#   - after each combination: ask keep/delete whole folder
#
# Folder examples:
#   data/eval2_fixed_combinations/rgb-r
#   data/eval2_fixed_combinations/rgb-g
#   data/eval2_fixed_combinations/rgb-b
# ============================================================

BASE_DIR="data/eval2_fixed_combinations"
mkdir -p "$BASE_DIR"

FPS=15
EPISODE_TIME_S=12
RESET_TIME_S=0
EPISODES_PER_COMBO=10

INITIAL_POSE_SCRIPT="scripts/go_eval1_both_initial.py"
DELETE_SCRIPT="scripts/delete_latest_eval2_episode.py"

mkdir -p scripts

# ============================================================
# Generate delete helper only if not already present
# ============================================================
if [[ ! -f "$DELETE_SCRIPT" ]]; then
cat > "$DELETE_SCRIPT" <<'PY'
#!/usr/bin/env python3
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def remove_path(path: Path):
    if not path.exists():
        return

    print(f"Removing: {path}")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def file_index(path: Path):
    match = re.search(r"file-(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def update_info_json(info_path: Path, remaining_episode_count: int, remaining_rows: int):
    if not info_path.exists():
        return

    try:
        info = read_json(info_path)
    except Exception as e:
        print(f"Warning: could not read {info_path}: {e}")
        return

    if "total_episodes" in info:
        info["total_episodes"] = remaining_episode_count

    if "total_frames" in info:
        info["total_frames"] = remaining_rows

    if "splits" in info and isinstance(info["splits"], dict):
        info["splits"]["train"] = f"0:{remaining_episode_count}"

    write_json(info_path, info)
    print(f"Updated: {info_path}")


def remove_empty_dirs(root: Path):
    for base_name in ["data", "videos", "meta"]:
        base = root / base_name
        if not base.exists():
            continue

        for d in sorted([x for x in base.rglob("*") if x.is_dir()], reverse=True):
            try:
                d.rmdir()
                print(f"Removed empty dir: {d}")
            except OSError:
                pass


def main():
    if len(sys.argv) != 2:
        print("Usage: delete_latest_eval2_episode.py DATASET_ROOT")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        print(f"Dataset root does not exist: {root}")
        sys.exit(1)

    print("")
    print("========================================")
    print("Deleting latest LeRobot v3 episode")
    print(f"Dataset root: {root}")
    print("========================================")

    data_files = sorted(
        root.glob("data/chunk-*/file-*.parquet"),
        key=file_index,
    )

    if not data_files:
        print("No data/chunk-*/file-*.parquet files found.")
        return

    latest_data_file = data_files[-1]

    rel = latest_data_file.relative_to(root)
    chunk_dir = rel.parts[1]
    file_name = rel.name
    file_stem = latest_data_file.stem

    latest_episode_index = None
    latest_rows = 0

    try:
        df_latest = pd.read_parquet(latest_data_file)
        latest_rows = len(df_latest)

        if "episode_index" in df_latest.columns and len(df_latest) > 0:
            latest_episode_index = int(df_latest["episode_index"].max())
    except Exception as e:
        print(f"Warning: could not read latest data file {latest_data_file}: {e}")

    print(f"Latest data file: {latest_data_file}")
    print(f"Chunk: {chunk_dir}")
    print(f"File stem: {file_stem}")
    print(f"Rows in latest file: {latest_rows}")

    if latest_episode_index is not None:
        print(f"Episode index inside latest file: {latest_episode_index}")

    remove_path(latest_data_file)

    videos_dir = root / "videos"
    if videos_dir.exists():
        for video_file in videos_dir.glob(f"*/{chunk_dir}/{file_stem}.*"):
            remove_path(video_file)

    remove_path(root / "meta" / "episodes" / chunk_dir / file_name)
    remove_path(root / "meta" / "episodes_stats" / chunk_dir / file_name)
    remove_path(root / "meta" / "episode_stats" / chunk_dir / file_name)

    remove_empty_dirs(root)

    remaining_data_files = sorted(root.glob("data/chunk-*/file-*.parquet"), key=file_index)

    remaining_eps = set()
    remaining_rows = 0

    for p in remaining_data_files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
            continue

        remaining_rows += len(df)

        if "episode_index" in df.columns:
            remaining_eps.update(int(x) for x in df["episode_index"].dropna().unique())

    remaining_episode_count = len(remaining_eps)

    update_info_json(root / "meta" / "info.json", remaining_episode_count, remaining_rows)

    print("")
    print("Deletion complete.")
    print(f"Deleted data file: {rel}")
    print(f"Deleted matching videos: videos/*/{chunk_dir}/{file_stem}.*")
    print(f"Deleted matching meta: meta/episodes/{chunk_dir}/{file_name}")
    print(f"Remaining episode indices: {sorted(remaining_eps)}")
    print(f"Remaining episode count: {remaining_episode_count}")
    print(f"Remaining frames: {remaining_rows}")
    print("")


if __name__ == "__main__":
    main()
PY

chmod +x "$DELETE_SCRIPT"
fi

# ============================================================
# count_episodes: fast bash file count (each episode = one parquet)
# ============================================================
count_episodes() {
  local root="$1"
  [[ -d "$root/data" ]] || { echo 0; return; }
  find "$root/data" -name "file-*.parquet" 2>/dev/null | wc -l | tr -d ' '
}

target_name() {
  local target="$1"

  case "$target" in
    r) echo "red" ;;
    g) echo "green" ;;
    b) echo "blue" ;;
    *) echo "unknown" ;;
  esac
}

task_for_target() {
  local target="$1"

  case "$target" in
    r)
      echo "Put the banana in the red colored bowl."
      ;;
    g)
      echo "Put the banana in the green colored bowl."
      ;;
    b)
      echo "Put the banana in the blue colored bowl."
      ;;
    *)
      echo "ERROR"
      ;;
  esac
}

display_order() {
  local order="$1"

  local left="${order:0:1}"
  local center="${order:1:1}"
  local right="${order:2:1}"

  echo "Bowl order from LEFT to RIGHT:"
  echo "  LEFT   = $(target_name "$left")"
  echo "  CENTER = $(target_name "$center")"
  echo "  RIGHT  = $(target_name "$right")"
}

# record_one_episode: records a single episode.
# Does NOT move to initial before recording — caller handles that once per combo.
# Always moves to initial AFTER recording so robot is ready for the next episode.
record_one_episode() {
  local combo="$1"
  local order="$2"
  local target="$3"
  local ep_root="$4"
  local repo_id="$5"
  local task="$6"
  local ep_num="$7"

  echo ""
  echo "--------------------------------------------------------------"
  echo "Combination: $combo"
  echo "Episode:    $ep_num / $EPISODES_PER_COMBO"
  echo "Task:       $task"
  echo "--------------------------------------------------------------"
  display_order "$order"
  echo "Target bowl: $(target_name "$target")"
  echo "--------------------------------------------------------------"

  if [[ -f "$ep_root/meta/info.json" ]]; then
    RESUME_FLAG="true"
    echo "Existing dataset found. Recording with --resume=true"
  else
    RESUME_FLAG="false"
    echo "No existing dataset found. Creating new dataset with --resume=false"

    if [[ -d "$ep_root" ]]; then
      echo "Removing incomplete dataset folder:"
      echo "$ep_root"
      rm -rf "$ep_root"
    fi
  fi

  lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.calibration_dir=calibration/so101_group47/follower \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 15, "warmup_s": 2}}' \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.calibration_dir=calibration/so101_group47/leader \
    --dataset.repo_id="$repo_id" \
    --dataset.root="$ep_root" \
    --resume="$RESUME_FLAG" \
    --dataset.fps="$FPS" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$task" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.vcodec=auto \
    --display_data=true

  echo ""
  echo "Moving follower and leader to fixed initial position AFTER recording..."
  python "$INITIAL_POSE_SCRIPT" || true
}

record_combo() {
  local combo="$1"
  local order="$2"
  local target="$3"
  local ep_root="$4"
  local repo_id="$5"
  local task="$6"

  if [[ -d "$ep_root" ]]; then
    local existing_count
    existing_count="$(count_episodes "$ep_root")"

    if [[ "$existing_count" -ge "$EPISODES_PER_COMBO" ]]; then
      echo ""
      echo "Existing complete folder found:"
      echo "$ep_root"
      read -rp "Keep existing and skip? [Y/n/delete]: " EXISTING_CHOICE

      case "${EXISTING_CHOICE:-}" in
        n|N|delete|d|D)
          echo "Deleting existing folder:"
          rm -rf "$ep_root"
          ;;
        *)
          echo "Skipping existing complete combination."
          return 2
          ;;
      esac
    elif [[ "$existing_count" -gt 0 ]]; then
      echo ""
      echo "Existing partial folder found:"
      echo "$ep_root"
      echo "Existing episodes: $existing_count / $EPISODES_PER_COMBO"
      read -rp "Resume this combination? [Y/n/delete]: " EXISTING_CHOICE

      case "${EXISTING_CHOICE:-}" in
        n|N|delete|d|D)
          echo "Deleting existing folder:"
          rm -rf "$ep_root"
          ;;
        *)
          echo "Resuming from episode $((existing_count + 1))."
          ;;
      esac
    else
      echo ""
      echo "Existing empty/broken folder found:"
      echo "$ep_root"
      echo "Deleting it before recording."
      rm -rf "$ep_root"
    fi
  fi

  # Move to initial once before the first episode of this combination.
  # After each episode the robot is already returned to initial, so no
  # redundant pre-episode reset is needed for subsequent episodes.
  echo ""
  echo "Moving follower and leader to fixed initial position BEFORE first episode..."
  python "$INITIAL_POSE_SCRIPT"

  while true; do
    local done_count
    done_count="$(count_episodes "$ep_root")"

    if [[ "$done_count" -ge "$EPISODES_PER_COMBO" ]]; then
      echo ""
      echo "Combination reached $done_count / $EPISODES_PER_COMBO episodes."
      break
    fi

    local next_ep
    next_ep=$((done_count + 1))

    record_one_episode "$combo" "$order" "$target" "$ep_root" "$repo_id" "$task" "$next_ep"

    echo ""
    echo "Episode finished:"
    echo "$ep_root"
    echo ""

    while true; do
      read -rp "Keep this episode? [ENTER/y = keep / d = delete latest / q = quit]: " KEEP_KEY

      case "${KEEP_KEY:-}" in
        y|Y|"")
          echo "Kept episode."
          break
          ;;
        d|D)
          echo ""
          echo "Deleting latest episode and fixing metadata..."
          python "$DELETE_SCRIPT" "$ep_root"
          echo "Deleted. Robot already at initial — re-recording this episode next."
          break
          ;;
        q|Q)
          echo "Exiting."
          exit 0
          ;;
        *)
          echo "Invalid input. Use ENTER/y, d, or q."
          ;;
      esac
    done
  done

  return 0
}

ORDERS=(
  "rgb"
  "rbg"
  "grb"
  "gbr"
  "brg"
  "bgr"
)

TARGETS=(
  "r"
  "g"
  "b"
)

echo "========================================"
echo "Eval 2 automatic combination recording"
echo "Base directory: $BASE_DIR"
echo "Total combinations: 18"
echo "Episodes per combination: $EPISODES_PER_COMBO"
echo "Episode time: $EPISODE_TIME_S seconds"
echo "Reset time inside LeRobot: $RESET_TIME_S seconds"
echo "Total target episodes: $((18 * EPISODES_PER_COMBO))"
echo ""
echo "Important:"
echo "  The reset is handled by: $INITIAL_POSE_SCRIPT"
echo "  It runs before the first episode and immediately after every episode."
echo ""
echo "Folder format:"
echo "  data/eval2_fixed_combinations/rgb-r"
echo "  data/eval2_fixed_combinations/rgb-g"
echo "  data/eval2_fixed_combinations/rgb-b"
echo "========================================"
echo ""

COMBO_INDEX=0

for ORDER in "${ORDERS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    COMBO_INDEX=$((COMBO_INDEX + 1))

    COMBO="${ORDER}-${TARGET}"
    TARGET_COLOR="$(target_name "$TARGET")"
    TASK="$(task_for_target "$TARGET")"

    EP_ROOT="${BASE_DIR}/${COMBO}"
    REPO_ID="local/eval2_${COMBO}"

    while true; do
      echo ""
      echo "=============================================================="
      echo "Eval 2 combination $COMBO_INDEX / 18"
      echo "Combination: $COMBO"
      echo "Episodes:    $EPISODES_PER_COMBO"
      echo "Save root:   $EP_ROOT"
      echo "Repo ID:     $REPO_ID"
      echo "Task:        $TASK"
      echo "--------------------------------------------------------------"
      display_order "$ORDER"
      echo "Target bowl: $TARGET_COLOR"
      echo "=============================================================="
      echo ""

      set +e
      record_combo "$COMBO" "$ORDER" "$TARGET" "$EP_ROOT" "$REPO_ID" "$TASK"
      RC=$?

      if [[ "$RC" -eq 2 ]]; then
        break
      fi

      echo ""
      echo "Combination finished:"
      echo "$EP_ROOT"
      echo ""

      while true; do
        read -rp "Keep this combination? [ENTER/y = keep / d = delete whole folder and re-record / q = quit]: " KEEP_COMBO_KEY

        case "${KEEP_COMBO_KEY:-}" in
          y|Y|"")
            echo "Kept combination: $COMBO"
            break 2
            ;;
          d|D)
            echo ""
            echo "Deleting whole combination folder:"
            echo "$EP_ROOT"
            rm -rf "$EP_ROOT"
            echo "Deleted. Re-recording combination: $COMBO"
            echo ""
            break
            ;;
          q|Q)
            echo "Exiting."
            exit 0
            ;;
          *)
            echo "Invalid input. Use ENTER/y, d, or q."
            ;;
        esac
      done
    done
  done
done

echo ""
echo "✓ Eval 2 recording complete."
echo "Data saved under:"
echo "$BASE_DIR"
