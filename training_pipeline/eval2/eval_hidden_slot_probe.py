#!/usr/bin/env python3
"""Evaluate Stage 1B hidden target-slot probes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


SLOTS = ("left", "middle", "right")


class Probe(nn.Module):
    def __init__(self, in_dim: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, 3)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.05), nn.Linear(256, 3))
        else:
            raise ValueError(kind)
        self.register_buffer("mean", torch.zeros(1, in_dim))
        self.register_buffer("std", torch.ones(1, in_dim))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


@torch.inference_mode()
def predict(state_dict, x: torch.Tensor, kind: str) -> list[int]:
    model = Probe(x.shape[1], kind)
    model.load_state_dict(state_dict)
    return model(x).argmax(dim=-1).tolist()


def confusion(pred: list[int], y: torch.Tensor) -> list[list[int]]:
    mat = [[0, 0, 0] for _ in range(3)]
    for p, gt in zip(pred, y.tolist(), strict=True):
        mat[gt][p] += 1
    return mat


def acc(pred: list[int], y: torch.Tensor) -> dict:
    correct = sum(int(p == gt) for p, gt in zip(pred, y.tolist(), strict=True))
    return {"correct": correct, "total": int(y.numel()), "accuracy": correct / int(y.numel()) if y.numel() else 0.0}


def split_indices(metadata: list[dict], split: str) -> list[int]:
    return [i for i, row in enumerate(metadata) if row["split"] == split]


def breakdown(pred: list[int], y: torch.Tensor, metadata: list[dict]) -> dict:
    out = {}
    for field in ("prompt_type", "target_slot", "layout_code"):
        bucket = defaultdict(lambda: [0, 0])
        for p, gt, row in zip(pred, y.tolist(), metadata, strict=True):
            key = row[field]
            bucket[key][0] += int(p == gt)
            bucket[key][1] += 1
        out[field] = {
            key: {"correct": c, "total": t, "accuracy": c / t if t else 0.0}
            for key, (c, t) in sorted(bucket.items())
        }
    return out


def same_image_test(pred: list[int], y: torch.Tensor, metadata: list[dict], max_groups: int = 8) -> list[dict]:
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[row["episode_index"]].append(i)
    rows = []
    for _, indices in sorted(groups.items()):
        slots = {metadata[i]["target_slot"] for i in indices}
        if len(slots) < 2:
            continue
        sample = []
        for i in indices[:6]:
            sample.append(
                {
                    "prompt": metadata[i]["bare_prompt"],
                    "gt": metadata[i]["target_slot"],
                    "pred": SLOTS[pred[i]],
                    "ok": pred[i] == y[i].item(),
                }
            )
        rows.append(
            {
                "episode_index": metadata[indices[0]]["episode_index"],
                "layout": metadata[indices[0]]["layout_left_to_right"],
                "items": sample,
            }
        )
        if len(rows) >= max_groups:
            break
    return rows


def same_prompt_diff_layout_test(pred: list[int], y: torch.Tensor, metadata: list[dict], max_prompts: int = 6) -> list[dict]:
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[row["bare_prompt"]].append(i)
    rows = []
    for prompt, indices in sorted(groups.items()):
        layouts = {metadata[i]["layout_id"] for i in indices}
        slots = {metadata[i]["target_slot"] for i in indices}
        if len(layouts) < 2 or len(slots) < 2:
            continue
        sample = []
        seen = set()
        for i in indices:
            key = metadata[i]["layout_id"]
            if key in seen:
                continue
            seen.add(key)
            sample.append(
                {
                    "layout": metadata[i]["layout_left_to_right"],
                    "gt": metadata[i]["target_slot"],
                    "pred": SLOTS[pred[i]],
                    "ok": pred[i] == y[i].item(),
                }
            )
            if len(sample) >= 6:
                break
        rows.append({"prompt": prompt, "items": sample})
        if len(rows) >= max_prompts:
            break
    return rows


def table_lines(report: dict, kind: str) -> list[str]:
    lines = []
    for condition in ("scaffold", "bare"):
        if condition not in report["conditions"]:
            continue
        lines.append(f"Condition: {condition} ({kind})")
        lines.append("| Feature source | Train acc | Val acc | Test acc |")
        lines.append("|---|---:|---:|---:|")
        for feature, fdata in report["conditions"][condition].items():
            if feature in {"counts", "diagnostics"} or kind not in fdata:
                continue
            row = fdata[kind]
            lines.append(
                f"| {feature} | {row['train']['accuracy']*100:.1f}% | "
                f"{row['val']['accuracy']*100:.1f}% | {row['test']['accuracy']*100:.1f}% |"
            )
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", default="data/stage1b_probe/hidden_states.pt")
    parser.add_argument("--probes", default="data/stage1b_probe/probes.pt")
    parser.add_argument("--train-summary", default="data/stage1b_probe/probe_train_summary.json")
    parser.add_argument("--out", default="data/stage1b_probe/probe_eval_report.json")
    parser.add_argument("--markdown", default="data/stage1b_probe/probe_eval_report.md")
    args = parser.parse_args()

    hidden = torch.load(args.hidden, map_location="cpu", weights_only=False)
    probes = torch.load(args.probes, map_location="cpu", weights_only=False)
    train_summary = json.loads(Path(args.train_summary).read_text())
    report = train_summary

    for condition, cdata in hidden["conditions"].items():
        metadata_all = cdata["metadata"]
        idx = {split: split_indices(metadata_all, split) for split in ("train", "val", "test")}
        y_all = cdata["labels"]
        report["conditions"][condition].setdefault("diagnostics", {})
        for feature, x_all in cdata["features"].items():
            for kind, state in probes["probes"][condition][feature].items():
                feature_report = report["conditions"][condition][feature][kind]
                for split in ("train", "val", "test"):
                    split_idx = idx[split]
                    pred = predict(state, x_all[split_idx], kind)
                    y = y_all[split_idx]
                    meta = [metadata_all[i] for i in split_idx]
                    feature_report[split] = acc(pred, y)
                    feature_report[f"{split}_confusion"] = confusion(pred, y)
                    feature_report[f"{split}_breakdown"] = breakdown(pred, y, meta)
                test_pred = predict(state, x_all[idx["test"]], kind)
                test_y = y_all[idx["test"]]
                test_meta = [metadata_all[i] for i in idx["test"]]
                feature_report["same_image_different_prompt"] = same_image_test(test_pred, test_y, test_meta)
                feature_report["same_prompt_different_layout"] = same_prompt_diff_layout_test(test_pred, test_y, test_meta)

    Path(args.out).write_text(json.dumps(report, indent=2))
    md = []
    md.append("# Stage 1B Hidden Target-Slot Probe")
    md.append("")
    md.append(f"Model: `{report['model']}`")
    md.append("")
    md.extend(table_lines(report, "linear"))
    md.extend(table_lines(report, "mlp"))
    md.append("Confusion matrices are `[gt left/middle/right] x [pred left/middle/right]` in the JSON report.")
    Path(args.markdown).write_text("\n".join(md))
    print("\n".join(md))
    print(f"Saved {args.out}")
    print(f"Saved {args.markdown}")


if __name__ == "__main__":
    main()
