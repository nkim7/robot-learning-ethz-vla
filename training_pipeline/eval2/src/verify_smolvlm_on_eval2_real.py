#!/usr/bin/env python3
"""
Verify smolvlm2-500m-vqa-init-position-strict10-color-slot-merged on 10 randomly
selected INITIAL frames from robot-learning-group47/eval2_fixed_bowl_fixed_banana_108.

Uses real robot video frames (not permutation_init_position test images).
Each sampled episode gets a DIFFERENT prompt type, covering all 4 types.

Layout ground truth comes from combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
DATASET_ID = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
N_SAMPLES  = 10
SEED       = 42
COLOR_ABBR = {"b": "blue", "g": "green", "r": "red"}
SLOTS      = ["left", "middle", "right"]

# ── Eval-2 prompt templates (fill c0=left, c1=mid, c2=right) ─────────────────
PROMPT_POOL = {
    "color": [
        "Put the banana in the {c0} colored bowl.",
        "Put the banana in the {c1} colored bowl.",
        "Put the banana in the {c2} colored bowl.",
    ],
    "ordinal": [
        "Put the banana into the 1st bowl from the left from the robot perspective.",
        "Put the banana into the 2nd bowl from the left from the robot perspective.",
        "Put the banana into the 3rd bowl from the left from the robot perspective.",
        "Put the banana into the 1st bowl from the right from the robot perspective.",
        "Put the banana into the 2nd bowl from the right from the robot perspective.",
        "Put the banana into the 3rd bowl from the right from the robot perspective.",
        "Put the banana into the middle bowl from the robot perspective.",
    ],
    "relative": [
        "Put the banana into the bowl on the left of the {c1} bowl from the robot perspective.",
        "Put the banana into the bowl on the right of the {c1} bowl from the robot perspective.",
        "Put the banana into the bowl on the left of the {c2} bowl from the robot perspective.",
        "Put the banana into the bowl on the right of the {c0} bowl from the robot perspective.",
    ],
    "negation": [
        "Put the banana into the bowl that is not {c1} and not {c2}.",
        "Put the banana into the bowl that is not {c0} and not {c2}.",
        "Put the banana into the bowl that is not {c0} and not {c1}.",
    ],
}

# ── Ground-truth resolution ───────────────────────────────────────────────────

def resolve_target(prompt: str, layout: list[str]) -> tuple[str, str]:
    """Return (target_color, target_slot) given prompt and layout [left,mid,right]."""
    p = prompt.lower()
    colors = ["blue", "red", "green"]

    if "not " in p:
        excluded = {c for c in colors if re.search(rf"\bnot\s+{c}\b", p)}
        remaining = [c for c in colors if c not in excluded]
        if len(remaining) == 1:
            tc = remaining[0]; return tc, SLOTS[layout.index(tc)]

    if "left of" in p or "right of" in p:
        mentioned = [c for c in colors if re.search(rf"\b{c}\b", p)]
        ref = mentioned[-1]; ref_idx = layout.index(ref)
        idx = ref_idx - 1 if "left of" in p else ref_idx + 1
        if 0 <= idx < 3:
            tc = layout[idx]; return tc, SLOTS[idx]
        raise ValueError(f"Relative points outside layout: {prompt!r} {layout}")

    if "1st bowl from the left" in p or "3rd bowl from the right" in p:
        return layout[0], "left"
    if "2nd bowl from the left" in p or "2nd bowl from the right" in p or "middle bowl" in p:
        return layout[1], "middle"
    if "3rd bowl from the left" in p or "1st bowl from the right" in p:
        return layout[2], "right"

    direct = [c for c in colors if re.search(rf"\b{c}\b", p)]
    if len(direct) == 1:
        tc = direct[0]; return tc, SLOTS[layout.index(tc)]

    raise ValueError(f"Cannot resolve: {prompt!r} layout={layout}")


def fill_prompt(tmpl: str, layout: list[str]) -> str:
    return tmpl.format(c0=layout[0], c1=layout[1], c2=layout[2])


def sample_prompt_for_layout(layout: list[str], ptype: str) -> tuple[str, str]:
    """Return (prompt, type). May skip relative if edge bowl is target."""
    rng = random.Random()  # uses global state
    pool = list(PROMPT_POOL[ptype])
    random.shuffle(pool)
    for tmpl in pool:
        p = fill_prompt(tmpl, layout)
        try:
            resolve_target(p, layout)
            return p, ptype
        except ValueError:
            continue
    # fallback to color
    tmpl = random.choice(PROMPT_POOL["color"])
    p = fill_prompt(tmpl, layout)
    return p, "color"


# ── Video frame extraction ────────────────────────────────────────────────────

def extract_frame_torchcodec(video_path: str, timestamp_sec: float) -> Image.Image:
    """Extract a single frame at the given timestamp using torchcodec."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(video_path, dimension_order="NHWC")
    frame = dec.get_frame_played_at(seconds=timestamp_sec)  # Returns Frame; .data is (H,W,C)
    arr = frame.data.numpy()
    return Image.fromarray(arr).convert("RGB")


def extract_frame_ffmpeg(video_path: str, timestamp_sec: float) -> Image.Image:
    """Fallback: use ffmpeg to extract a frame at a specific timestamp."""
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1",
        tmp, "-loglevel", "quiet"
    ]
    subprocess.run(cmd, check=True)
    img = Image.open(tmp).convert("RGB")
    os.unlink(tmp)
    return img


# ── Model prompt ──────────────────────────────────────────────────────────────

SYSTEM = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls. "
    "Then answer with:\n"
    "target color: <color>\n"
    "target slot: <left|middle|right>"
)


def build_chat_prompt(processor, instruction: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM}\n\nInstruction: {instruction}"},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_output(raw: str) -> dict:
    """Parse 'target color: X\ntarget slot: Y' model output."""
    result = {}
    raw_l = raw.lower()
    # Extract target color
    m = re.search(r"target\s+color\s*[:\-]\s*(\w+)", raw_l)
    if m:
        result["target_color"] = m.group(1).strip()
    # Extract target slot
    m = re.search(r"target\s+slot\s*[:\-]\s*(\w+)", raw_l)
    if m:
        result["target_slot"] = m.group(1).strip()
    # Also try to parse layout if present
    m = re.search(r"layout.*?:\s*(.+?)(?:\n|$)", raw_l)
    if m:
        parts = re.findall(r"\b(blue|red|green)\b", m.group(1))
        if len(parts) == 3:
            result["layout_left_to_right"] = parts
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)

    # ── Load episode map ──────────────────────────────────────────────────────
    import pandas as pd
    from huggingface_hub import hf_hub_download

    print(f"Fetching episode map from {DATASET_ID} ...")
    map_path = hf_hub_download(
        DATASET_ID, "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv",
        repo_type="dataset"
    )
    combo_df = pd.read_csv(map_path)
    print(f"  Combo map: {len(combo_df)} groups")
    print(combo_df[["combo","episode_start","episode_end","target_bowl_color","bowl_order_left_to_right"]].to_string(index=False))

    # Build episode → (layout, target_color) map
    ep_info: dict[int, tuple[list[str], str]] = {}
    for _, row in combo_df.iterrows():
        layout_code = str(row["bowl_order_left_to_right"])
        layout = [COLOR_ABBR[c] for c in layout_code]
        color  = str(row["target_bowl_color"])
        for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
            ep_info[ep] = (layout, color)

    all_eps = sorted(ep_info.keys())
    print(f"\n  Total episodes in map: {len(all_eps)}")

    # ── Load episode timestamp metadata ──────────────────────────────────────
    ep_meta_path = hf_hub_download(
        DATASET_ID, "meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset"
    )
    ep_meta = pd.read_parquet(ep_meta_path)
    # Index by episode_index
    ep_ts = {}
    for _, r in ep_meta.iterrows():
        eidx = int(r["episode_index"])
        # from_timestamp is a 1-element list (or scalar)
        ts = r["videos/observation.images.front/from_timestamp"]
        if isinstance(ts, (list, np.ndarray)):
            ts = float(ts[0])
        else:
            ts = float(ts)
        chunk_idx = int(r["videos/observation.images.front/chunk_index"]
                        if not isinstance(r["videos/observation.images.front/chunk_index"], (list, np.ndarray))
                        else r["videos/observation.images.front/chunk_index"][0])
        file_idx = int(r["videos/observation.images.front/file_index"]
                       if not isinstance(r["videos/observation.images.front/file_index"], (list, np.ndarray))
                       else r["videos/observation.images.front/file_index"][0])
        ep_ts[eidx] = (ts, chunk_idx, file_idx)

    # ── Sample 10 episodes spread across layouts ──────────────────────────────
    # Pick episodes uniformly across the 18 combo groups
    combo_to_eps: dict[str, list[int]] = {}
    for _, row in combo_df.iterrows():
        eps = list(range(int(row["episode_start"]), int(row["episode_end"]) + 1))
        combo_to_eps[str(row["combo"])] = eps

    combos = list(combo_to_eps.keys())
    random.shuffle(combos)
    sampled_eps = []
    for combo in combos:
        sampled_eps.append(random.choice(combo_to_eps[combo]))
        if len(sampled_eps) >= N_SAMPLES:
            break

    # Cycle through prompt types
    ptypes_cycle = ["color", "ordinal", "relative", "negation",
                    "color", "ordinal", "relative", "negation",
                    "color", "ordinal"]

    # ── Cache video files ─────────────────────────────────────────────────────
    # Download all needed video files
    video_cache: dict[tuple[int,int], str] = {}
    needed = set()
    for ep_idx in sampled_eps:
        if ep_idx in ep_ts:
            _, chunk_idx, file_idx = ep_ts[ep_idx]
            needed.add((chunk_idx, file_idx))
    print(f"\nDownloading {len(needed)} video chunk(s) ...")
    for (ci, fi) in needed:
        vpath = f"videos/observation.images.front/chunk-{ci:03d}/file-{fi:03d}.mp4"
        local = hf_hub_download(DATASET_ID, vpath, repo_type="dataset")
        video_cache[(ci, fi)] = local
        print(f"  {vpath} → {local}")

    # ── Load model ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    print(f"\nLoading model: {MODEL_ID} on {device} ...")
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, _attn_implementation="eager"
    ).to(device)
    model.eval()
    print("  Model ready.\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = []
    header = f"{'#':<3} {'ep':>4}  {'type':<8}  {'gt_slot':>7}  {'pred_slot':>9}  {'pred_color':>10}  {'layout_ok':>9}  {'ok':>3}"
    print(header)
    print("-" * len(header) + "-" * 30)

    for i, ep_idx in enumerate(sampled_eps):
        layout, _default_color = ep_info.get(ep_idx, (None, None))
        if layout is None:
            print(f"  ep {ep_idx}: not in combo map — skip")
            continue

        # Timestamp for initial frame
        if ep_idx not in ep_ts:
            print(f"  ep {ep_idx}: no timestamp metadata — skip")
            continue
        ts, chunk_idx, file_idx = ep_ts[ep_idx]
        video_path = video_cache.get((chunk_idx, file_idx))
        if video_path is None:
            print(f"  ep {ep_idx}: video not downloaded — skip")
            continue

        # Extract initial frame (timestamp = from_timestamp of episode)
        try:
            img = extract_frame_torchcodec(video_path, ts)
        except Exception as e:
            print(f"  ep {ep_idx}: torchcodec failed ({e}), trying ffmpeg ...")
            try:
                img = extract_frame_ffmpeg(video_path, ts)
            except Exception as e2:
                print(f"  ep {ep_idx}: ffmpeg also failed ({e2}) — skip")
                continue

        # Sample a prompt
        ptype = ptypes_cycle[i % len(ptypes_cycle)]
        prompt, ptype = sample_prompt_for_layout(layout, ptype)
        try:
            gt_color, gt_slot = resolve_target(prompt, layout)
        except ValueError as e:
            print(f"  ep {ep_idx}: cannot resolve GT ({e}) — skip")
            continue

        # Run model
        chat_text = build_chat_prompt(processor, prompt)
        inputs = processor(text=[chat_text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        raw = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        parsed      = parse_output(raw)
        pred_color  = str(parsed.get("target_color", "?")).lower()
        pred_slot   = str(parsed.get("target_slot",  "?")).lower()
        pred_layout = parsed.get("layout_left_to_right", [])
        ok_slot    = pred_slot  == gt_slot
        ok_color   = pred_color == gt_color
        ok_layout  = (pred_layout == layout) if pred_layout else None  # may not be in output

        ok_mark = "✓" if ok_slot else "✗"
        layout_mark = "✓" if ok_layout else ("?" if ok_layout is None else "✗")
        print(f"{i+1:<3} {ep_idx:>4}  {ptype:<8}  {gt_slot:>7}  {pred_slot:>9}  {pred_color:>10}  "
              f"{layout_mark:>9}  {ok_mark:>3}  {prompt[:55]}")
        if not ok_slot:
            print(f"    layout_gt  = {layout}")
            print(f"    raw output = {raw[:200]!r}")

        results.append({
            "ep": ep_idx, "type": ptype, "layout": layout, "prompt": prompt,
            "gt_color": gt_color, "gt_slot": gt_slot,
            "pred_color": pred_color, "pred_slot": pred_slot, "pred_layout": pred_layout,
            "ok_slot": ok_slot, "ok_color": ok_color, "ok_layout": ok_layout, "raw": raw,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results:
        print("\nNo results collected.")
        sys.exit(1)

    total = len(results)
    def acc(key):
        vals = [r[key] for r in results if r[key] is not None]
        return (sum(vals) / len(vals) * 100) if vals else 0.0, len(vals)

    print("\n" + "=" * 80)
    print(f"  Tested {total} samples on REAL VIDEO frames from {DATASET_ID}")
    print(f"  Model: {MODEL_ID}")
    print()
    slot_acc, slot_n   = acc("ok_slot")
    color_acc, color_n = acc("ok_color")
    layout_acc, lay_n  = acc("ok_layout")
    print(f"  target_slot  accuracy:  {slot_acc:.1f}%   ({sum(r['ok_slot'] for r in results)}/{slot_n})")
    print(f"  target_color accuracy:  {color_acc:.1f}%   ({sum(r['ok_color'] for r in results)}/{color_n})")
    if lay_n:
        print(f"  layout       accuracy:  {layout_acc:.1f}%   (parsed from {lay_n} outputs)")

    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["ok_slot"])
    print("\n  By prompt type (target_slot acc):")
    for t in ["color", "ordinal", "relative", "negation"]:
        if t in by_type:
            vals = by_type[t]
            print(f"    {t:<10}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals)*100:.1f}%")

    out_path = Path("logs/verify_smolvlm_on_eval2_real.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
