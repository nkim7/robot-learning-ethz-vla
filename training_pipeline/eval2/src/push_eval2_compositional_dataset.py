"""
Creates eval2_compositional_augmented on HF.

Takes eval2_fixed_bowl_fixed_banana_108 and duplicates each episode once
per valid compositional prompt, so the same trajectory is seen with every
valid language description during training.

Original: 108 episodes × ~6-8 prompts = ~756 "virtual" episodes
All video/action data is identical; only task_index changes.

Usage (on GPU):
    python3 push_eval2_compositional_dataset.py
"""

import json, shutil, re
from pathlib import Path
from collections import defaultdict
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
SRC_REPO   = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DST_REPO   = "robot-learning-group47/eval2_compositional_augmented"
WORK_DIR   = Path("/data/eval2_aug_work")       # on GPU; change if running locally
HF_HOME    = "/data/hf_home"

import os; os.environ["HF_HOME"] = HF_HOME

# ── Combo map (from combo_episode_map CSV) ────────────────────────────────────
COMBOS = [
    ("bgr", "blue",   0,  5),
    ("bgr", "green",  6, 11),
    ("bgr", "red",   12, 17),
    ("brg", "blue",  18, 23),
    ("brg", "green", 24, 29),
    ("brg", "red",   30, 35),
    ("gbr", "blue",  36, 41),
    ("gbr", "green", 42, 47),
    ("gbr", "red",   48, 53),
    ("grb", "blue",  54, 59),
    ("grb", "green", 60, 65),
    ("grb", "red",   66, 71),
    ("rbg", "blue",  72, 77),
    ("rbg", "green", 78, 83),
    ("rbg", "red",   84, 89),
    ("rgb", "blue",  90, 95),
    ("rgb", "green", 96,101),
    ("rgb", "red",  102,107),
]

PREFIX = "Put the banana into the"
SUFFIX = "from the robot perspective."
COLORS = ["blue", "green", "red"]

def gen_prompts(arrangement: str, target: str) -> list[str]:
    color_map = {'b': 'blue', 'g': 'green', 'r': 'red'}
    order = [color_map[c] for c in arrangement]
    pos   = order.index(target)
    others = [c for c in COLORS if c != target]
    ordinal = ["1st", "2nd", "3rd"]
    prompts = []

    prompts.append(f"Put the banana in the {target} colored bowl.")
    prompts.append(f"{PREFIX} {ordinal[pos]} bowl from the left {SUFFIX}")
    prompts.append(f"{PREFIX} {ordinal[2-pos]} bowl from the right {SUFFIX}")
    if pos == 1:
        prompts.append(f"{PREFIX} middle bowl {SUFFIX}")
    if pos > 0:
        prompts.append(f"{PREFIX} bowl on the right of the {order[pos-1]} bowl {SUFFIX}")
    if pos < 2:
        prompts.append(f"{PREFIX} bowl on the left of the {order[pos+1]} bowl {SUFFIX}")
    a, b = others
    prompts.append(f"{PREFIX} bowl that is not {a} and not {b}.")
    prompts.append(f"{PREFIX} bowl that is not {b} and not {a}.")
    return prompts

# ── Build full task list & per-episode prompt lists ───────────────────────────
all_prompts: list[str] = []
ep_prompts: dict[int, list[str]] = {}  # original ep_idx → valid prompts

for arr, target, ep_start, ep_end in COMBOS:
    prompts = gen_prompts(arr, target)
    for ep in range(ep_start, ep_end + 1):
        ep_prompts[ep] = prompts
    for p in prompts:
        if p not in all_prompts:
            all_prompts.append(p)

task_index_map = {p: i for i, p in enumerate(all_prompts)}
print(f"Unique tasks: {len(all_prompts)}")

# ── Download source dataset (metadata + parquet only, skip videos) ─────────────
print("Downloading source dataset metadata…")
src_local = snapshot_download(
    SRC_REPO, repo_type="dataset",
    local_dir=str(WORK_DIR / "src"),
    ignore_patterns=["*.mp4"],
)
src_path = Path(src_local)

# Also download videos separately (we'll symlink/copy them)
print("Downloading source videos…")
src_full = snapshot_download(
    SRC_REPO, repo_type="dataset",
    local_dir=str(WORK_DIR / "src_full"),
)
src_full = Path(src_full)

# ── Load original parquet files ───────────────────────────────────────────────
data_dir = src_path / "data"
parquet_files = sorted(data_dir.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
print(f"Original rows: {len(df)}")

# ── Build augmented dataframe ─────────────────────────────────────────────────
print("Building augmented dataset…")
new_rows = []
new_ep_idx = 0
ep_mapping = []  # (new_ep_idx, orig_ep_idx, task_index, prompt)

for orig_ep_idx in range(108):
    orig_frames = df[df["episode_index"] == orig_ep_idx].copy()
    for prompt in ep_prompts[orig_ep_idx]:
        t_idx = task_index_map[prompt]
        chunk = orig_frames.copy()
        chunk["episode_index"] = new_ep_idx
        chunk["task_index"]    = t_idx
        # recompute global index later
        new_rows.append(chunk)
        ep_mapping.append((new_ep_idx, orig_ep_idx, t_idx, prompt))
        new_ep_idx += 1

new_df = pd.concat(new_rows, ignore_index=True)
new_df["index"] = range(len(new_df))
print(f"Augmented rows: {len(new_df)} | Augmented episodes: {new_ep_idx}")

# ── Write new dataset to disk ─────────────────────────────────────────────────
dst_path = WORK_DIR / "dst"
(dst_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
(dst_path / "meta").mkdir(parents=True, exist_ok=True)

# Save parquet (chunk by 1000 frames)
chunk_size = 1000
for chunk_i, start in enumerate(range(0, len(new_df), chunk_size)):
    chunk_dir = dst_path / "data" / f"chunk-{chunk_i:03d}"
    chunk_dir.mkdir(exist_ok=True)
    new_df.iloc[start:start+chunk_size].to_parquet(
        chunk_dir / "file-000.parquet", index=False
    )
print(f"Wrote {chunk_i+1} parquet chunks")

# ── Videos: create symlink structure ─────────────────────────────────────────
# new_ep_idx → orig_ep_idx mapping
orig_for_new = {new: orig for new, orig, _, _ in ep_mapping}

src_videos = src_full / "videos"
if src_videos.exists():
    video_keys = [d.name for d in src_videos.iterdir() if d.is_dir()]
    print(f"Video keys: {video_keys}")
    for vk in video_keys:
        for new_ep, orig_ep in orig_for_new.items():
            # find original video file
            orig_ep_chunk = orig_ep // 1000
            orig_ep_file  = orig_ep % 1000  # most datasets use 1 file per episode
            # scan for the actual file
            src_vk_dir = src_videos / vk
            # find the mp4 for orig_ep
            found = list(src_vk_dir.rglob(f"episode_{orig_ep:06d}.mp4"))
            if not found:
                # try flat structure
                found = list(src_vk_dir.rglob("*.mp4"))
                found = [f for f in found if f"episode_{orig_ep:06d}" in f.name]
            if not found:
                continue
            src_vid = found[0]
            dst_vid_dir = dst_path / "videos" / vk / src_vid.parent.relative_to(src_vk_dir)
            dst_vid_dir.mkdir(parents=True, exist_ok=True)
            dst_vid = dst_vid_dir / f"episode_{new_ep:06d}.mp4"
            if not dst_vid.exists():
                shutil.copy2(src_vid, dst_vid)
    print("Videos copied")
else:
    print("⚠ No videos directory found in source — skipping video copy")

# ── Write meta files ──────────────────────────────────────────────────────────
# tasks.jsonl
with open(dst_path / "meta" / "tasks.jsonl", "w") as f:
    for prompt, idx in task_index_map.items():
        f.write(json.dumps({"task_index": idx, "task": prompt}) + "\n")

# episodes.jsonl
with open(dst_path / "meta" / "episodes.jsonl", "w") as f:
    for new_ep, orig_ep, t_idx, prompt in ep_mapping:
        # compute length
        length = int((new_df["episode_index"] == new_ep).sum())
        f.write(json.dumps({
            "episode_index": new_ep,
            "task_index": t_idx,
            "length": length,
            "orig_episode_index": orig_ep,
            "prompt": prompt,
        }) + "\n")

# info.json — copy and patch
src_info = json.loads((src_path / "meta" / "info.json").read_text())
src_info["total_episodes"] = new_ep_idx
src_info["total_frames"]   = len(new_df)
src_info["total_tasks"]    = len(all_prompts)
src_info["splits"]         = {"train": f"0:{new_ep_idx}"}
(dst_path / "meta" / "info.json").write_text(json.dumps(src_info, indent=2))

print(f"\nDataset summary:")
print(f"  Episodes: {new_ep_idx}")
print(f"  Frames:   {len(new_df)}")
print(f"  Tasks:    {len(all_prompts)}")

# ── Push to HF ────────────────────────────────────────────────────────────────
print(f"\nPushing to {DST_REPO}…")
api = HfApi()
api.create_repo(DST_REPO, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(dst_path),
    repo_id=DST_REPO,
    repo_type="dataset",
    commit_message=f"Compositional augmentation: {new_ep_idx} eps × 22 prompts from {SRC_REPO}",
)
print(f"Done: https://huggingface.co/datasets/{DST_REPO}")

# Also save the combo CSV for reference
combo_csv = "combo,episode_start,episode_end,new_ep_start,prompts\n"
new_ep_ptr = 0
for arr, target, ep_start, ep_end in COMBOS:
    n_orig = ep_end - ep_start + 1
    n_prompts = len(gen_prompts(arr, target))
    combo_csv += f"{arr}_{target[0]},{ep_start},{ep_end},{new_ep_ptr},{n_prompts}\n"
    new_ep_ptr += n_orig * n_prompts
api.upload_file(
    path_or_fileobj=combo_csv.encode(),
    path_in_repo="combo_augmented_map.csv",
    repo_id=DST_REPO,
    repo_type="dataset",
)
print("Uploaded combo map CSV")
