#!/usr/bin/env python3
"""Held-out gate for Eval2 Stage A head-first relational grounding."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from train_eval2_stageA_head_relational_grounding import (
    COLORS,
    DEFAULT_DATASET,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INIT,
    DEFAULT_OUT,
    ID_TO_COLOR,
    ID_TO_SLOT,
    LAYOUT_CODES,
    SLOTS,
    StageADataset,
    TargetHeads,
    build_cases,
    collate,
    first_frames,
    load_episode_truth,
    summarize_cases,
    vlm_last_text_hidden,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=f"{DEFAULT_OUT}/checkpoints/003000/pretrained_model")
    parser.add_argument("--base-policy", default=DEFAULT_INIT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--train-reps", default="0,1,2,3")
    parser.add_argument("--val-reps", default="4,5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-cases", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--raw-threshold", type=float, default=0.90)
    parser.add_argument("--anti-threshold", type=float, default=0.90)
    parser.add_argument("--layout-threshold", type=float, default=0.85)
    parser.add_argument("--anchor-threshold", type=float, default=0.85)
    return parser.parse_args()


def load_policy_and_heads(ckpt: Path, base_policy: str, device: str):
    if (ckpt / "adapter_config.json").exists():
        base = SmolVLAPolicy.from_pretrained(base_policy)
        base.config.device = device
        policy = PeftModel.from_pretrained(base, str(ckpt))
        processors_path = str(ckpt)
        print(f"Loaded Stage A adapter: {ckpt}")
        print(f"Adapter base policy: {base_policy}")
    else:
        policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        policy.config.device = device
        processors_path = str(ckpt)
        print(f"Loaded full Stage A policy: {ckpt}")
    policy = policy.to(device).eval()
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    hidden_size = raw.model.vlm_with_expert.config.text_config.hidden_size
    heads = TargetHeads(hidden_size).to(device)
    heads.load_state_dict(torch.load(ckpt / "stageA_target_heads.pt", map_location=device))
    heads.eval()
    return policy, heads, processors_path


def inc(bucket: Counter, slot_ok: bool, color_ok: bool) -> None:
    bucket["total"] += 1
    bucket["slot_ok"] += int(slot_ok)
    bucket["color_ok"] += int(color_ok)
    bucket["both_ok"] += int(slot_ok and color_ok)


def pack(bucket: Counter) -> dict:
    n = max(1, bucket["total"])
    return {
        "total": int(bucket["total"]),
        "slot_acc": bucket["slot_ok"] / n,
        "color_acc": bucket["color_ok"] / n,
        "both_acc": bucket["both_ok"] / n,
    }


@torch.no_grad()
def run_gate(policy, heads, preprocessor, dataset: StageADataset, args, device: str) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    overall = Counter()
    by_family = defaultdict(Counter)
    by_layout = defaultdict(Counter)
    by_anchor = defaultdict(Counter)
    by_side = defaultdict(Counter)
    by_target_slot = defaultdict(Counter)
    by_target_color = defaultdict(Counter)
    anti_by_family = defaultdict(Counter)
    anti = Counter()
    shortcut = defaultdict(Counter)
    rows = []

    for batch in loader:
        case_indices = batch.pop("_case_index").tolist()
        target_slot = batch.pop("target_slot_id").to(device)
        target_color = batch.pop("target_color_id").to(device)
        for key in list(dataset.base.meta.camera_keys):
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        batch = preprocessor(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp and device == "cuda"):
            hidden = vlm_last_text_hidden(policy, batch)
            slot_logits, color_logits = heads(hidden)
        pred_slot = slot_logits.argmax(dim=-1)
        pred_color = color_logits.argmax(dim=-1)
        for row_idx, ps, pc, ts, tc in zip(case_indices, pred_slot, pred_color, target_slot, target_color, strict=False):
            case = dataset.cases[int(row_idx)]
            ps_s = ID_TO_SLOT[int(ps.item())]
            pc_s = ID_TO_COLOR[int(pc.item())]
            slot_ok = ps.item() == ts.item()
            color_ok = pc.item() == tc.item()
            for bucket in (
                overall,
                by_family[case.family],
                by_layout[case.layout_code],
                by_target_slot[case.target_slot],
                by_target_color[case.target_color],
            ):
                inc(bucket, slot_ok, color_ok)
            if case.anchor_color:
                inc(by_anchor[case.anchor_color], slot_ok, color_ok)
            if case.side:
                inc(by_side[case.side], slot_ok, color_ok)
            if case.anti_shortcut:
                inc(anti, slot_ok, color_ok)
                inc(anti_by_family[case.family], slot_ok, color_ok)
            if case.family == "raw_relative":
                if case.side == "right_of":
                    shortcut["right_of_pred_rightmost"]["total"] += 1
                    shortcut["right_of_pred_rightmost"]["hits"] += int(ps_s == "right")
                if case.side == "left_of":
                    shortcut["left_of_pred_leftmost"]["total"] += 1
                    shortcut["left_of_pred_leftmost"]["hits"] += int(ps_s == "left")
                if pc_s == case.anchor_color:
                    shortcut["pred_anchor_color"]["hits"] += 1
                shortcut["pred_anchor_color"]["total"] += 1
            row = {
                "family": case.family,
                "layout": case.layout_code,
                "relation_key": case.relation_key,
                "anchor_color": case.anchor_color,
                "side": case.side,
                "anti_shortcut": case.anti_shortcut,
                "expected_slot": case.target_slot,
                "pred_slot": ps_s,
                "slot_ok": bool(slot_ok),
                "expected_color": case.target_color,
                "pred_color": pc_s,
                "color_ok": bool(color_ok),
                "prompt": case.prompt,
            }
            rows.append(row)
            if args.print_cases or case.family == "raw_relative":
                print(
                    f"{case.family}\t{case.layout_code}\t{case.relation_key}\t"
                    f"expected={case.target_slot}/{case.target_color}\t"
                    f"pred={ps_s}/{pc_s}\tslot_ok={slot_ok}\tcolor_ok={color_ok}"
                )

    shortcut_rates = {
        key: {
            "total": int(counter["total"]),
            "rate": counter["hits"] / max(1, counter["total"]),
        }
        for key, counter in shortcut.items()
    }
    metrics = {
        "overall": pack(overall),
        "by_family": {k: pack(v) for k, v in sorted(by_family.items())},
        "raw_relative": pack(by_family["raw_relative"]),
        "scaffolded_relative": pack(by_family["scaffolded_relative"]),
        "semi_resolved": pack(by_family["semi_resolved"]),
        "simple": pack(by_family["simple"]),
        "anti_shortcut": pack(anti),
        "anti_shortcut_by_family": {k: pack(v) for k, v in sorted(anti_by_family.items())},
        "by_layout": {k: pack(v) for k, v in sorted(by_layout.items())},
        "by_anchor": {k: pack(v) for k, v in sorted(by_anchor.items())},
        "by_side": {k: pack(v) for k, v in sorted(by_side.items())},
        "by_target_slot": {k: pack(v) for k, v in sorted(by_target_slot.items())},
        "by_target_color": {k: pack(v) for k, v in sorted(by_target_color.items())},
        "shortcut_rates": shortcut_rates,
        "rows": rows,
    }
    return metrics


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    train_reps = {int(x) for x in args.train_reps.split(",") if x}
    val_reps = {int(x) for x in args.val_reps.split(",") if x}

    truth = load_episode_truth(args.dataset, args.dataset_root)
    base_ds = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    frames = first_frames(base_ds, truth)
    cases = build_cases(frames, train_reps, val_reps)
    print("HELD_OUT_EVAL_SUMMARY")
    print(json.dumps(summarize_cases(frames, cases, "val"), indent=2))

    policy, heads, processors_path = load_policy_and_heads(ckpt, args.base_policy, device)
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=processors_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    val_ds = StageADataset(base_ds, frames, cases, "val")
    metrics = run_gate(policy, heads, preprocessor, val_ds, args, device)

    raw = metrics["raw_relative"]
    anti = metrics["anti_shortcut"]
    raw_ok = raw["slot_acc"] >= args.raw_threshold and raw["color_acc"] >= args.raw_threshold
    anti_ok = anti["slot_acc"] >= args.anti_threshold and anti["color_acc"] >= args.anti_threshold
    layout_ok = all(
        metrics["by_layout"].get(code, {"slot_acc": 0.0, "color_acc": 0.0})["slot_acc"] >= args.layout_threshold
        and metrics["by_layout"].get(code, {"slot_acc": 0.0, "color_acc": 0.0})["color_acc"] >= args.layout_threshold
        for code in LAYOUT_CODES
    )
    anchor_ok = all(
        metrics["by_anchor"].get(color, {"slot_acc": 0.0, "color_acc": 0.0})["slot_acc"] >= args.anchor_threshold
        and metrics["by_anchor"].get(color, {"slot_acc": 0.0, "color_acc": 0.0})["color_acc"] >= args.anchor_threshold
        for color in COLORS
    )
    no_middle_collapse = metrics["by_target_slot"].get("left", {"slot_acc": 0})["slot_acc"] >= args.layout_threshold and metrics[
        "by_target_slot"
    ].get("right", {"slot_acc": 0})["slot_acc"] >= args.layout_threshold
    gate_pass = raw_ok and anti_ok and layout_ok and anchor_ok and no_middle_collapse

    metrics["gate"] = {
        "raw_relative_slot_color_ge_threshold": raw_ok,
        "anti_shortcut_slot_color_ge_threshold": anti_ok,
        "every_layout_ge_threshold": layout_ok,
        "every_anchor_ge_threshold": anchor_ok,
        "no_middle_collapse_check": no_middle_collapse,
        "gate_pass": gate_pass,
    }
    print("\nSUMMARY")
    print(json.dumps({k: v for k, v in metrics.items() if k != "rows"}, indent=2))
    print(f"GATE_PASS={gate_pass}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
