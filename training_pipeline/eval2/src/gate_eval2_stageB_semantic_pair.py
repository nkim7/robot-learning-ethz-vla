#!/usr/bin/env python3
"""Semantic gate for Stage B checkpoints.

Runs the Stage A pair-primary held-out evaluation on a trained Stage B policy.
This checks whether action co-training damaged the relational VLM/LoRA
representation.  The pair head is diagnostic only and is not part of inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/shadeform")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from train_eval2_stageA_head_relational_grounding import (
    first_frames,
    load_episode_truth,
)
from train_eval2_stageA_head_relational_grounding_v4 import (
    TargetHeadsV4,
    StageADatasetV4,
    build_cases_v4,
)
from train_eval2_stageA_head_relational_grounding_v5 import evaluate_v5


DEFAULT_DATASET = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_DATASET_ROOT = (
    "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--heads", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-amp", action="store_true", default=True)
    ap.add_argument("--train-reps", default="0,1,2,3")
    ap.add_argument("--val-reps", default="4,5")
    return ap.parse_args()


def pct(x):
    return 100.0 * x


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    heads_path = Path(args.heads) if args.heads else ckpt / "stageB_aux_target_heads.pt"

    print(f"SEMANTIC_GATE ckpt={ckpt}", flush=True)
    print(f"heads={heads_path}", flush=True)

    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy = policy.to(device)
    policy.eval()
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    heads = TargetHeadsV4(hidden_size, dropout=0.0).to(device)
    state = torch.load(heads_path, map_location=device)
    heads.load_state_dict(state, strict=False)
    heads.eval()

    ds = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    truths = load_episode_truth(args.dataset, Path(args.dataset_root))
    frames = first_frames(ds, truths)
    train_reps = {int(x) for x in args.train_reps.split(",") if x}
    val_reps = {int(x) for x in args.val_reps.split(",") if x}
    cases = build_cases_v4(frames, train_reps, val_reps)
    val_ds = StageADatasetV4(ds, frames, cases, split="val")

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
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

    metrics = evaluate_v5(policy, heads, preprocessor, val_ds, args, device)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))

    raw = metrics["by_family"]["raw_relative"]
    anti = metrics["raw_anti_shortcut"]
    print(
        f"raw_pair={pct(raw['pair_acc']):.1f}% raw_anti_pair={pct(anti['pair_acc']):.1f}%",
        flush=True,
    )
    print("raw_by_layout", {k: round(pct(v["pair_acc"]), 1) for k, v in metrics["raw_by_layout"].items()}, flush=True)
    print("raw_by_anchor", {k: round(pct(v["pair_acc"]), 1) for k, v in metrics["raw_by_anchor"].items()}, flush=True)
    print("raw_by_side", {k: round(pct(v["pair_acc"]), 1) for k, v in metrics["raw_by_side"].items()}, flush=True)
    print("SEMANTIC_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
