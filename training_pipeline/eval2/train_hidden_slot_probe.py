#!/usr/bin/env python3
"""Train linear/MLP hidden-state target-slot probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
def eval_probe(model: Probe, x: torch.Tensor, y: torch.Tensor) -> dict:
    pred = model(x).argmax(dim=-1)
    return {
        "correct": int(pred.eq(y).sum().item()),
        "total": int(y.numel()),
        "accuracy": float(pred.eq(y).float().mean().item()) if y.numel() else 0.0,
        "pred": pred.tolist(),
    }


def split_indices(metadata: list[dict], split: str) -> list[int]:
    return [i for i, row in enumerate(metadata) if row["split"] == split]


def train_one(x_train, y_train, x_val, y_val, kind: str, epochs: int, lr: float, seed: int) -> Probe:
    torch.manual_seed(seed)
    model = Probe(x_train.shape[1], kind)
    model.mean.copy_(x_train.mean(dim=0, keepdim=True))
    model.std.copy_(x_train.std(dim=0, keepdim=True).clamp(min=1e-6))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = eval_probe(model, x_val, y_val)["accuracy"]
        if val > best_val:
            best_val = val
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", default="data/stage1b_probe/hidden_states.pt")
    parser.add_argument("--out", default="data/stage1b_probe/probes.pt")
    parser.add_argument("--summary", default="data/stage1b_probe/probe_train_summary.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = torch.load(args.hidden, map_location="cpu", weights_only=False)
    result = {"model": payload["model"], "jsonl": payload["jsonl"], "conditions": {}}
    probes = {"model": payload["model"], "jsonl": payload["jsonl"], "probes": {}}
    for condition, cdata in payload["conditions"].items():
        probes["probes"][condition] = {}
        result["conditions"][condition] = {}
        metadata = cdata["metadata"]
        idx = {split: split_indices(metadata, split) for split in ("train", "val", "test")}
        y = cdata["labels"]
        for feature_name, x in cdata["features"].items():
            probes["probes"][condition][feature_name] = {}
            result["conditions"][condition][feature_name] = {}
            for kind in ("linear", "mlp"):
                model = train_one(
                    x[idx["train"]],
                    y[idx["train"]],
                    x[idx["val"]],
                    y[idx["val"]],
                    kind,
                    args.epochs,
                    args.lr,
                    args.seed,
                )
                train_acc = eval_probe(model, x[idx["train"]], y[idx["train"]])
                val_acc = eval_probe(model, x[idx["val"]], y[idx["val"]])
                test_acc = eval_probe(model, x[idx["test"]], y[idx["test"]])
                probes["probes"][condition][feature_name][kind] = model.state_dict()
                result["conditions"][condition][feature_name][kind] = {
                    "train": {k: v for k, v in train_acc.items() if k != "pred"},
                    "val": {k: v for k, v in val_acc.items() if k != "pred"},
                    "test": {k: v for k, v in test_acc.items() if k != "pred"},
                    "feature_dim": int(x.shape[1]),
                }
                print(
                    f"{condition:8s} {feature_name:24s} {kind:6s} "
                    f"train={train_acc['accuracy']*100:5.1f} "
                    f"val={val_acc['accuracy']*100:5.1f} "
                    f"test={test_acc['accuracy']*100:5.1f}"
                )
        result["conditions"][condition]["counts"] = {
            "split": dict(Counter(row["split"] for row in metadata)),
            "target_slot": dict(Counter(row["target_slot"] for row in metadata)),
            "prompt_type": dict(Counter(row["prompt_type"] for row in metadata)),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probes, out)
    Path(args.summary).write_text(json.dumps(result, indent=2))
    print(f"Saved probes to {out}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
