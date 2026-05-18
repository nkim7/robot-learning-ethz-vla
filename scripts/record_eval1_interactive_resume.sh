#!/usr/bin/env bash
set -euo pipefail

cd ~/robot-learning-ethz-vla
source .venv/bin/activate

export LEADER_PORT=/dev/tty.usbmodem5B141126211
export FOLLOWER_PORT=/dev/tty.usbmodem5B141138561

BASE_DIR="data/eval1_fixed_interactive"
mkdir -p "$BASE_DIR"

# One dataset folder for all episodes
EP_ROOT="${BASE_DIR}/eval1_fixed_all"
REPO_ID="local/eval1_fixed_all"

FPS=15
EPISODE_TIME_S=12
RESET_TIME_S=0

mkdir -p scripts

# ============================================================
# Helper: delete latest LeRobot v3 episode and fix metadata
#
# Your dataset format:
#   data/chunk-000/file-003.parquet
#   videos/observation.images.front/chunk-000/file-003.mp4
#   meta/episodes/chunk-000/file-003.parquet
#
# This deletes the latest file-XXX pair, not episode_XXX files.
# ============================================================
cat > scripts/delete_latest_eval1_episode.py <<'PY'
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
    """
    Extract index from file-003.parquet or file-003.mp4.
    """
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
        print("Usage: delete_latest_eval1_episode.py DATASET_ROOT")
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
    chunk_dir = rel.parts[1]           # chunk-000
    file_name = rel.name               # file-003.parquet
    file_stem = latest_data_file.stem  # file-003

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

    # 1. Delete main data parquet.
    remove_path(latest_data_file)

    # 2. Delete matching videos:
    # videos/{video_key}/chunk-000/file-003.mp4
    videos_dir = root / "videos"
    if videos_dir.exists():
        for video_file in videos_dir.glob(f"*/{chunk_dir}/{file_stem}.*"):
            remove_path(video_file)

    # 3. Delete matching meta episode parquet:
    # meta/episodes/chunk-000/file-003.parquet
    remove_path(root / "meta" / "episodes" / chunk_dir / file_name)

    # 4. Delete matching episode stats parquet if present.
    remove_path(root / "meta" / "episodes_stats" / chunk_dir / file_name)
    remove_path(root / "meta" / "episode_stats" / chunk_dir / file_name)

    # 5. Remove empty directories.
    remove_empty_dirs(root)

    # 6. Recompute remaining episode count and frame count from remaining data parquet files.
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

    # 7. Update meta/info.json.
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

chmod +x scripts/delete_latest_eval1_episode.py

echo "========================================"
echo "Eval 1 interactive recording"
echo "Local dataset directory: $EP_ROOT"
echo "All episodes are saved into one dataset folder."
echo "Before every episode, follower + leader are moved to fixed initial pose."
echo "After each episode, choose keep or delete latest episode."
echo "========================================"
echo ""

EP_COUNT=0

while true; do
  echo ""
  echo "Choose task color:"
  echo "  b = blue"
  echo "  r = red"
  echo "  g = green"
  echo "  q = quit"
  read -rp "Color [b/r/g/q]: " COLOR_KEY

  case "$COLOR_KEY" in
    b|B)
      COLOR="blue"
      TASK="Put the banana in the blue colored bowl."
      ;;
    r|R)
      COLOR="red"
      TASK="Put the banana in the red colored bowl."
      ;;
    g|G)
      COLOR="green"
      TASK="Put the banana in the green colored bowl."
      ;;
    q|Q)
      echo "Finished recording."
      echo "Total kept episodes this session: $EP_COUNT"
      exit 0
      ;;
    *)
      echo "Invalid input. Use b, r, g, or q."
      continue
      ;;
  esac

  echo ""
  echo "========================================"
  echo "Next episode"
  echo "Color:      $COLOR"
  echo "Task:       $TASK"
  echo "Save root:  $EP_ROOT"
  echo "Repo ID:    $REPO_ID"
  echo "========================================"
  echo ""

  echo "Moving follower and leader to fixed initial position..."
  python scripts/go_eval1_both_initial.py

  echo "Waiting for USB/robot connection to settle..."
  sleep 3

  echo ""
  echo "Follower and leader are at fixed initial position."
  echo "Place banana and bowls now."
  read -rp "Press ENTER to start recording this episode..."

  if [[ -f "$EP_ROOT/meta/info.json" ]]; then
    RESUME_FLAG="true"
    echo "Existing dataset found. Recording with --resume=true"
  else
    RESUME_FLAG="false"
    echo "No existing dataset found. Creating new dataset with --resume=false"
  fi

  lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.calibration_dir=calibration/so101_group47/follower \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 15, "warmup_s": 5}}' \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.calibration_dir=calibration/so101_group47/leader \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$EP_ROOT" \
    --resume="$RESUME_FLAG" \
    --dataset.fps="$FPS" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.single_task="$TASK" \
    --display_data=true
  python scripts/go_eval1_both_initial.py || true

  echo ""
  echo "Episode finished:"
  echo "$EP_ROOT"
  echo ""

  while true; do
    read -rp "Keep this episode? [y = keep / d = delete latest / q = quit]: " KEEP_KEY

    case "$KEEP_KEY" in
      y|Y)
        EP_COUNT=$((EP_COUNT + 1))
        echo "Kept episode. Total kept this session: $EP_COUNT"
        break
        ;;
      d|D)
        echo ""
        echo "Deleting latest episode and fixing metadata..."
        python scripts/delete_latest_eval1_episode.py "$EP_ROOT"
        echo "Deleted latest episode."
        break
        ;;
      q|Q)
        echo "Exiting."
        echo "Total kept episodes this session: $EP_COUNT"
        exit 0
        ;;
      *)
        echo "Invalid input. Use y, d, or q."
        ;;
    esac
  done
done