#!/usr/bin/env python3
"""Offline action probe for Eval2 Stage B checkpoints.

Diagnostic only.  This does not resolve prompts externally for inference; it
compares the actions produced by one SmolVLA policy under different raw prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

sys.path.insert(0, "/home/shadeform")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


COLOR_FULL = {"b": "blue", "g": "green", "r": "red"}
SLOTS = ("left", "middle", "right")
ORDINALS = ("1st", "2nd", "3rd")
COMBO_CSV = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
DEFAULT_DATASET = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_DATASET_ROOT = (
    "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
)
DEFAULT_COMBO_MAP = (
    "/home/shadeform/.cache/lerobot/robot-learning-group47/"
    "eval2_fixed_bowl_fixed_banana_108/combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
)
DEFAULT_BASE = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/"
    "checkpoints/020000/pretrained_model"
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--combo-map", default=DEFAULT_COMBO_MAP)
    ap.add_argument("--baseline", default=DEFAULT_BASE)
    ap.add_argument("--stageA-adapter", default="")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--layout", action="append", default=["bgr", "brg", "gbr", "grb", "rbg", "rgb"])
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--frame-min", type=int, default=10)
    ap.add_argument("--frame-max", type=int, default=70)
    ap.add_argument("--out-json", default="")
    return ap.parse_args()


def ordinal_prompt(slot_idx: int) -> str:
    return f"Put the banana into the {ORDINALS[slot_idx]} bowl from the left from the robot perspective."


def color_prompt(color: str) -> str:
    return f"Put the banana into the {color} bowl."


def relative_cases(layout: list[str]) -> list[dict]:
    out = []
    requested = [
        ("right_of", "blue"),
        ("left_of", "green"),
        ("right_of", "red"),
        ("left_of", "red"),
    ]
    for side, anchor in requested:
        if anchor not in layout:
            continue
        ai = layout.index(anchor)
        ti = ai + 1 if side == "right_of" else ai - 1
        if ti < 0 or ti >= len(layout):
            continue
        side_words = "right of" if side == "right_of" else "left of"
        out.append({
            "key": f"{side}_{anchor}",
            "prompt": f"Put the banana into the bowl on the {side_words} the {anchor} bowl from the robot perspective.",
            "expected_slot": SLOTS[ti],
            "expected_color": layout[ti],
            "target_slot_idx": ti,
        })
    return out


def combo_rows(combo_map: Path):
    with combo_map.open() as f:
        yield from csv.DictReader(f)


def layout_frames(ds, combo_map: Path, layout_code: str, n_frames: int, frame_min: int, frame_max: int) -> list[int]:
    ranges = []
    for row in combo_rows(combo_map):
        if row["bowl_order_left_to_right"].strip().lower() == layout_code:
            ranges.append((int(row["episode_start"]), int(row["episode_end"])))
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    hits = []
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep, fr = int(ep_val), int(fr_val)
        if frame_min <= fr <= frame_max and any(lo <= ep <= hi for lo, hi in ranges):
            hits.append(gi)
    if len(hits) <= n_frames:
        return hits
    stride = len(hits) / n_frames
    return [hits[int(round(i * stride))] for i in range(n_frames)]


def load_policy(label: str, path: str, device: str, base: str | None = None):
    if base:
        base_policy = SmolVLAPolicy.from_pretrained(base)
        policy = PeftModel.from_pretrained(base_policy, path, is_trainable=False)
    else:
        policy = SmolVLAPolicy.from_pretrained(path)
    policy = policy.to(device)
    policy.eval()
    return label, policy


@torch.no_grad()
def action_chunk(policy, preprocessor, ds, gi: int, prompt: str, device: str) -> torch.Tensor:
    from torch.utils.data import default_collate

    item = dict(ds[gi])
    item["task"] = prompt
    item.pop("task_index", None)
    for key in ds.meta.camera_keys:
        if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
            item[key] = item[key].float() / 255.0
    task = item.pop("task")
    batch = default_collate([item])
    batch["task"] = [task]
    batch = preprocessor(batch)
    policy.reset()
    return policy.predict_action_chunk(batch)[0].detach().cpu().float()


def cos(a, b):
    return float(F.cosine_similarity(a[None], b[None]).item())


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt().item())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combo_map = Path(args.combo_map)
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    tmp = SmolVLAPolicy.from_pretrained(args.baseline)
    delta_ts = resolve_delta_timestamps(tmp.config, ds_meta)
    del tmp
    ds = LeRobotDataset(args.dataset, root=args.dataset_root, delta_timestamps=delta_ts, video_backend="pyav")

    specs = [("stage2_baseline", args.baseline, None)]
    if args.stageA_adapter:
        specs.append(("stage2_plus_stageA_lora", args.stageA_adapter, args.baseline))
    for ckpt in args.ckpt:
        label = Path(ckpt).parent.name if "/checkpoints/" in ckpt else Path(ckpt).name
        specs.append((label, ckpt, None))

    baseline_actions = {}
    results = {"models": {}, "baseline": args.baseline}
    for label, path, base in specs:
        print(f"\n=== {label}: {path} ===", flush=True)
        _, policy = load_policy(label, path, device, base)
        preprocessor, _ = make_pre_post_processors(
            policy.config,
            pretrained_path=(base or path),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "normalizer_processor": {
                    "stats": ds.meta.stats,
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        model_out = {}
        for layout_code in args.layout:
            layout = [COLOR_FULL[ch] for ch in layout_code]
            gis = layout_frames(ds, combo_map, layout_code, args.n_frames, args.frame_min, args.frame_max)
            anchors = {}
            color_anchors = {}
            for si in range(3):
                chunks = [action_chunk(policy, preprocessor, ds, gi, ordinal_prompt(si), device) for gi in gis]
                anchors[SLOTS[si]] = torch.stack(chunks).mean(0)
            for color in layout:
                chunks = [action_chunk(policy, preprocessor, ds, gi, color_prompt(color), device) for gi in gis]
                color_anchors[color] = torch.stack(chunks).mean(0)
            rel_rows = []
            prompt_chunks = []
            for case in relative_cases(layout):
                chunks = [action_chunk(policy, preprocessor, ds, gi, case["prompt"], device) for gi in gis]
                mean = torch.stack(chunks).mean(0)
                prompt_chunks.append(mean)
                slot_sims = {slot: cos(flat(mean), flat(ref)) for slot, ref in anchors.items()}
                color_sims = {color: cos(flat(mean), flat(ref)) for color, ref in color_anchors.items()}
                pred_slot = max(slot_sims, key=slot_sims.get)
                pred_color = max(color_sims, key=color_sims.get)
                key = (layout_code, case["key"])
                if label == "stage2_baseline":
                    baseline_actions[key] = mean
                    delta = {"full": 0.0, "early_0_10": 0.0, "mid_10_25": 0.0, "late_25_50": 0.0}
                else:
                    base_act = baseline_actions.get(key)
                    delta = None
                    if base_act is not None:
                        delta = {
                            "full": rmse(mean, base_act),
                            "early_0_10": rmse(mean[:10], base_act[:10]),
                            "mid_10_25": rmse(mean[10:25], base_act[10:25]),
                            "late_25_50": rmse(mean[25:50], base_act[25:50]),
                        }
                row = {
                    **case,
                    "pred_slot_by_ordinal_anchor": pred_slot,
                    "pred_color_by_color_anchor": pred_color,
                    "slot_anchor_ok": pred_slot == case["expected_slot"],
                    "color_anchor_ok": pred_color == case["expected_color"],
                    "slot_sims": slot_sims,
                    "color_sims": color_sims,
                    "rmse_delta_vs_stage2_same_prompt": delta,
                }
                rel_rows.append(row)
                print(
                    f"{layout_code} {case['key']:16s} expected={case['expected_slot']}/{case['expected_color']} "
                    f"pred={pred_slot}/{pred_color} delta_vs_stage2={delta}",
                    flush=True,
                )
            sensitivity = {}
            if len(prompt_chunks) > 1:
                stack = torch.stack(prompt_chunks)
                sensitivity = {
                    "full_std_mean": float(stack.std(dim=0).mean().item()),
                    "early_0_10_std_mean": float(stack[:, :10].std(dim=0).mean().item()),
                    "mid_10_25_std_mean": float(stack[:, 10:25].std(dim=0).mean().item()),
                    "late_25_50_std_mean": float(stack[:, 25:50].std(dim=0).mean().item()),
                }
            model_out[layout_code] = {
                "frames": gis,
                "relative": rel_rows,
                "prompt_sensitivity": sensitivity,
            }
        results["models"][label] = model_out
        del policy
        torch.cuda.empty_cache()

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
