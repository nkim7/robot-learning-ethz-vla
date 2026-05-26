"""
Mine old Eval 2 HF repos for "all 3 bowls visible after lift" frames.

For each repo we:
  1. Download only the parquet + first-chunk video via snapshot_download.
  2. Find the frame per episode whose observation.state is closest to the
     canonical "banana lifted, all bowls visible" pose.
  3. Extract that frame with ffmpeg (works on AV1 as well as h264).
  4. Assemble contact sheets (one per repo) for manual inspection.

Usage:
  python mine_eval2_resolver_frames.py [--out /tmp/eval2_resolver_mine]

After running, open the contact sheets in the output directory and decide
which repos/frames are usable for building resolver v3.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Target pose: banana lifted, all 3 bowls clearly visible, arm not yet moving
# toward the target.  Indices match observation.state order:
#   [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# ---------------------------------------------------------------------------
TARGET_POSE = np.array([7.43, 56.57, -82.95, 64.18, -121.27, 0.0], dtype=np.float32)

# Joint weights: down-weight wrist_roll and gripper (less diagnostic)
POSE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.2], dtype=np.float32)

ORG = "robot-learning-group47"

# Repos already baked into eval2_180_chengming_promptaug_clean_256_v3.
# These come from eval2_90_chengming (which merges the 9 per-layout buckets).
SKIP_REPOS = {
    # The 180-clean chain
    "eval2_180_chengming_promptaug_clean_256_v3",
    "eval2_180_chengming_promptaug_clean_256_v2",
    "eval2_180_chengming_promptaug_clean_256_v1",
    "eval2_180_chengming_promptaug_clean_v1",
    "eval2_180_chengming_combined_raw_v1",
    "eval2_180_Chengming",
    # 90/270 variants
    "eval2_90_chengming",
    "eval2_90_chengming_promptaug_v1",
    "eval2_90_chengming_2phase_promptmix_v1",
    "eval2_270_chengming_promptaug_v1",
    # All-prompts aggregates
    "eval2_all_prompts",
    "eval2_all_prompts_fixed_266",
    # Derived / inspection / eval run datasets
    "eval2_proaug",
    "eval2_2phase_contact_sheets_debug",
    "eval2_180_split_inspection_posebased",
    "eval2_180_2phase_resolvedcolor_clean256_v1",
    "eval2_merged_grb_combo_langaug",
    "eval2_fixed_combinations_1",
    # Compositional repos — excluded by user request
    "eval2_compositional_v1",
    "eval2_compositional_v2",
    # Raw per-layout repos — same recordings as eval2_90_chengming (source for 180-clean)
    "eval2_rgb_b", "eval2_rgb_r", "eval2_rgb_g",
    "eval2_rbg_b", "eval2_rbg_r", "eval2_rbg_g",
    "eval2_grb_b", "eval2_grb_r", "eval2_grb_g",
    "eval2_gbr_b", "eval2_gbr_r", "eval2_gbr_g",
    "eval2_brg_b", "eval2_brg_r", "eval2_brg_g",
    "eval2_bgr_b", "eval2_bgr_r", "eval2_bgr_g",
    # Per-layout eval1-prompt relabelings  (same raw data, different prompt)
    "eval2_brg_g_eval1prompt",
    "eval2_bgr_g_eval1prompt",
    "eval2_rgb_g_eval1prompt",
    "eval2_grb_b_eval1prompt",
    "eval2_grb_r_eval1prompt",
    "eval2_rgb_r_eval1prompt",
    "eval2_bgr_b_eval1prompt",
    "eval2_bgr_r_eval1prompt",
    "eval2_brg_b_eval1prompt",
    "eval2_brg_r_eval1prompt",
    "eval2_gbr_b_eval1prompt",
    "eval2_gbr_g_eval1prompt",
    "eval2_gbr_r_eval1prompt",
    "eval2_grb_g_eval1prompt",
    "eval2_rbg_b_eval1prompt",
    "eval2_rbg_g_eval1prompt",
    "eval2_rbg_r_eval1prompt",
    "eval2_rgb_b_eval1prompt",
}

# eval_ prefix means evaluation-run result datasets, not training data
SKIP_PREFIXES = ("eval_eval2",)

FPS = 15  # All eval2 recordings are 15 fps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pose_dist(state: np.ndarray) -> float:
    diff = (state - TARGET_POSE) * POSE_WEIGHTS
    return float(np.linalg.norm(diff))


def download_repo(repo_id: str, local_dir: Path) -> bool:
    """Download only chunk-000 parquet + video files. Returns True on success."""
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=[
                "data/chunk-000/*.parquet",
                "videos/observation.images.front/chunk-000/*.mp4",
                "meta/info.json",
            ],
            ignore_patterns=["*.json.lock"],
        )
        return True
    except Exception as exc:
        print(f"  [WARN] download failed for {repo_id}: {exc}")
        return False


def find_best_frame(parquet_path: Path) -> tuple[int, float, int]:
    """
    Returns (frame_index, pose_dist, episode_index) for the frame closest
    to TARGET_POSE within the search window [lo, hi).

    We restrict the search to frames 50-140 to avoid the very start (arm
    near initial position) and the very end (arm already lowering banana).
    """
    df = pd.read_parquet(parquet_path)
    if "observation.state" not in df.columns:
        raise KeyError(f"No observation.state in {parquet_path}")

    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    lo = min(50, len(states) // 4)
    hi = min(140, len(states) * 3 // 4)
    if hi <= lo:
        lo, hi = 0, len(states)

    window = states[lo:hi]
    dists = np.linalg.norm((window - TARGET_POSE) * POSE_WEIGHTS, axis=1)
    best_rel = int(np.argmin(dists))
    best_frame = lo + best_rel
    best_dist = float(dists[best_rel])
    ep_idx = int(df["episode_index"].iloc[best_frame])
    return best_frame, best_dist, ep_idx


def probe_video(video_path: Path) -> tuple[int, float]:
    """Return (nb_frames, fps) from ffprobe. Falls back to (0, FPS) on error."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    try:
        import json as _json
        data = _json.loads(result.stdout)
        stream = data["streams"][0]
        nb = int(stream.get("nb_frames", 0))
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        return nb, fps
    except Exception:
        return 0, float(FPS)


def extract_frame_ffmpeg(video_path: Path, frame_index: int, out_path: Path,
                          fps: float = FPS, size: int = 320) -> bool:
    """Extract a single frame by index using ffmpeg. Works for AV1 and h264."""
    t = frame_index / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t:.4f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={size}:{size}",
        "-q:v", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not out_path.exists():
        # Fallback: frame-accurate select filter (slower)
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_index}),scale={size}:{size}",
            "-vsync", "vfr",
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True)
        return result2.returncode == 0 and out_path.exists()
    return True


def score_frame(img_path: Path) -> dict:
    """
    Count pixels per color category (bowls: R/G/B, banana: yellow).
    Also tracks what fraction of yellow pixels are in the lower 55% of the
    image — when the banana is gripped the arm comes from the bottom, so
    yellow should be low in frame. When the banana is sitting on the table
    it appears small and high up.
    """
    import colorsys
    W, H = 160, 120
    img = Image.open(img_path).convert("RGB").resize((W, H))
    pixels = list(img.getdata())

    red = green = blue = 0
    yellow_total = 0
    yellow_lower = 0  # yellow pixels in lower 55% of image
    lower_thresh = int(H * 0.45)  # row index above which = upper portion

    for i, (r, g, b) in enumerate(pixels):
        if r + g + b < 40:
            continue
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s < 0.20 or v < 0.15:
            continue
        h360 = h * 360
        row = i // W
        if (h360 < 18 or h360 > 340) and s > 0.40:
            red += 1
        elif 25 < h360 < 70 and s > 0.35 and v > 0.45:
            yellow_total += 1
            if row >= lower_thresh:
                yellow_lower += 1
        elif 80 < h360 < 165 and s > 0.25:
            green += 1
        elif 175 < h360 < 265 and s > 0.28:
            blue += 1

    yellow_lower_frac = yellow_lower / yellow_total if yellow_total > 0 else 0.0
    return {
        "red": red, "green": green, "blue": blue,
        "yellow": yellow_total, "yellow_lower_frac": yellow_lower_frac,
    }


def passes_filter(scores: dict,
                  bowl_min: int = 150,
                  banana_min: int = 350,
                  banana_lower_frac_min: float = 0.45) -> bool:
    """
    True if:
    - All 3 bowl colors are clearly present (bowl_min pixels each)
    - Banana is large enough to be the gripped close-up banana, not a tiny
      dot on the table (banana_min pixels)
    - Most yellow pixels are in the lower half of the image — banana in
      gripper hangs low in frame; banana on table sits small and high up
    """
    return (scores["red"] >= bowl_min and
            scores["green"] >= bowl_min and
            scores["blue"] >= bowl_min and
            scores["yellow"] >= banana_min and
            scores["yellow_lower_frac"] >= banana_lower_frac_min)


def cluster_indices(indices: list[int], gap: int = 45) -> list[int]:
    """
    Collapse runs of consecutive passing frames into one representative each.
    Within each cluster, pick the middle index so we land in the steadiest
    part of the scene rather than the edge.
    gap: minimum frame distance to start a new cluster.
    """
    if not indices:
        return []
    clusters = []
    cur = [indices[0]]
    for idx in indices[1:]:
        if idx - cur[-1] <= gap:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)
    return [c[len(c) // 2] for c in clusters]


def extract_frames_color_filtered(
    video_path: Path,
    out_dir: Path,
    stem: str,
    size: int = 320,
    scan_fps: float = 4.0,
) -> list[tuple[Path, str]]:
    """
    Find every frame in the video where all 3 bowls AND a grabbed banana are
    simultaneously visible.

    Strategy:
      1. Extract thumbnails at scan_fps (default 4fps → every ~7-8 real frames
         at 30fps) — dense enough to catch all instances.
      2. Score each thumbnail with HSV color counts for R/G/B bowls + yellow banana.
      3. Cluster consecutive passing frames and pick the middle of each cluster,
         so near-duplicate frames from the same moment count as one.
      4. Extract the full-res frame for each cluster representative.
    """
    import tempfile, json as _json

    nb_frames, fps = probe_video(video_path)
    if nb_frames < 10:
        dur_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True,
        )
        try:
            dur = float(_json.loads(dur_result.stdout)["format"]["duration"])
            nb_frames = int(dur * fps)
        except Exception:
            nb_frames = 300

    # Step 1: dense thumbnail scan.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={scan_fps},scale=160:120",
            "-q:v", "5",
            str(tmp / "scan_%05d.jpg"),
        ]
        subprocess.run(cmd, capture_output=True)

        scan_frames = sorted(tmp.glob("scan_*.jpg"))
        stride = fps / scan_fps

        # Step 2: score and filter.
        passing = []
        for sf in scan_frames:
            n = int(sf.stem.split("_")[1]) - 1  # 1-based → 0-based
            real_idx = int(n * stride)
            try:
                sc = score_frame(sf)
                if passes_filter(sc):
                    passing.append(real_idx)
            except Exception:
                pass

    print(f"    color filter: {len(passing)}/{len(scan_frames)} scan frames pass "
          f"(3 bowls + banana)")

    if not passing:
        print(f"    no passing frames found — skipping")
        return []

    # Step 3: cluster to remove near-duplicates.
    # Use gap = ~1.5 seconds worth of real frames.
    gap = int(fps * 1.5)
    reps = cluster_indices(passing, gap=gap)
    print(f"    clustered into {len(reps)} distinct scenes")

    # Step 4: extract full-res frames.
    results = []
    for idx in reps:
        out_path = out_dir / f"{stem}_f{idx:05d}.jpg"
        ok = extract_frame_ffmpeg(video_path, idx, out_path, fps=fps, size=size)
        if ok:
            results.append((out_path, f"{stem} f{idx}/{nb_frames}"))
    return results


def make_contact_sheet(
    images: list[tuple[Path, str]],  # (image_path, label)
    out_path: Path,
    cols: int = 6,
    thumb: int = 320,
    label_height: int = 36,
) -> None:
    """Tile images into a contact sheet with text labels."""
    rows = (len(images) + cols - 1) // cols
    W = cols * thumb
    H = rows * (thumb + label_height)
    sheet = Image.new("RGB", (W, H), color=(30, 30, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, (img_path, label) in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * thumb, r * (thumb + label_height)
        try:
            img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
        except Exception:
            img = Image.new("RGB", (thumb, thumb), color=(80, 0, 0))
        sheet.paste(img, (x, y))

        draw = ImageDraw.Draw(sheet)
        # Semi-transparent background for label
        draw.rectangle([x, y + thumb, x + thumb, y + thumb + label_height],
                       fill=(20, 20, 20))
        draw.text((x + 4, y + thumb + 2), label, fill=(220, 220, 100), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path), quality=90)
    print(f"  Saved contact sheet: {out_path} ({len(images)} frames)")


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(repo_name: str, work_dir: Path, out_dir: Path) -> list[dict]:
    """
    Download, find best frames, extract images.
    Returns list of result dicts for the summary CSV.
    """
    repo_id = f"{ORG}/{repo_name}"
    local_dir = work_dir / repo_name
    results = []

    print(f"\n{'='*60}")
    print(f"Processing: {repo_id}")

    if not download_repo(repo_id, local_dir):
        return results

    # Find all parquet files (one per episode)
    parquet_files = sorted(local_dir.glob("data/chunk-000/*.parquet"))
    if not parquet_files:
        print(f"  [WARN] No parquet files found in {local_dir}")
        return results

    print(f"  Found {len(parquet_files)} parquet file(s)")

    images_for_sheet: list[tuple[Path, str]] = []

    for pq in parquet_files:
        vid_path = (local_dir / "videos/observation.images.front/chunk-000" /
                    f"{pq.stem}.mp4")
        frame_dir = out_dir / repo_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Try pose-matched extraction first.
        try:
            frame_idx, dist, ep_idx = find_best_frame(pq)
            if not vid_path.exists():
                print(f"  [WARN] Video not found: {vid_path}")
                continue
            out_img = frame_dir / f"{pq.stem}_f{frame_idx:04d}_d{dist:.1f}.jpg"
            ok = extract_frame_ffmpeg(vid_path, frame_idx, out_img, fps=float(FPS))
            if not ok:
                raise RuntimeError(f"ffmpeg extraction failed for frame {frame_idx}")
            label = f"{pq.stem} f{frame_idx} d={dist:.1f}"
            images_for_sheet.append((out_img, label))
            results.append({
                "repo": repo_name,
                "parquet": str(pq),
                "video": str(vid_path),
                "frame_index": frame_idx,
                "pose_dist": dist,
                "episode_index": ep_idx,
                "out_image": str(out_img),
                "method": "pose_match",
            })
            print(f"  {pq.name}: frame {frame_idx}, dist={dist:.2f} [pose match]")

        except Exception as exc:
            # Parquet is corrupt or unavailable — fall back to evenly-spaced sampling.
            print(f"  [WARN] {pq.name}: {exc} — falling back to video sampling")
            if not vid_path.exists():
                print(f"  [WARN] Video also not found: {vid_path}")
                continue
            sampled = extract_frames_color_filtered(vid_path, frame_dir, pq.stem)
            images_for_sheet.extend(sampled)
            for img_path, lbl in sampled:
                results.append({
                    "repo": repo_name,
                    "parquet": str(pq),
                    "video": str(vid_path),
                    "frame_index": int(img_path.stem.split("_f")[-1]),
                    "pose_dist": float("nan"),
                    "episode_index": -1,
                    "out_image": str(img_path),
                    "method": "video_sample",
                })
            print(f"  {pq.stem}: extracted {len(sampled)} sampled frames [video fallback]")

    if images_for_sheet:
        sheet_path = out_dir / "sheets" / f"{repo_name}.jpg"
        make_contact_sheet(images_for_sheet, sheet_path, cols=8, thumb=256)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_eval2_repos() -> list[str]:
    """Return all eval2 repos from HF, filtering out skipped/eval-run ones."""
    from huggingface_hub import list_datasets
    all_repos = list(list_datasets(author=ORG, search="eval2"))
    names = []
    for ds in all_repos:
        name = ds.id.split("/")[-1]
        if name in SKIP_REPOS:
            continue
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        names.append(name)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/eval2_resolver_mine",
                        help="Output directory for frames and contact sheets")
    parser.add_argument("--repos", nargs="*",
                        help="Specific repo names to process (default: all eval2)")
    parser.add_argument("--work-dir", default="/tmp/eval2_resolver_mine_dl",
                        help="Directory for downloaded repo snapshots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.repos:
        repos = args.repos
    else:
        print("Fetching repo list from HuggingFace...")
        repos = list_eval2_repos()

    print(f"\nRepos to process ({len(repos)}):")
    for r in repos:
        print(f"  {r}")

    all_results = []
    for repo_name in repos:
        results = process_repo(repo_name, work_dir, out_dir)
        all_results.extend(results)

    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = out_dir / "mining_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path} ({len(df)} frames)")
        print(df[["repo", "frame_index", "pose_dist"]].to_string(index=False))
    else:
        print("\nNo frames extracted.")

    print(f"\nContact sheets: {out_dir / 'sheets'}/")
    print("Review the sheets and note which repos have usable 'all bowls visible' frames.")


if __name__ == "__main__":
    main()
