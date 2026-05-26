#!/usr/bin/env python3
"""Probe whether SmolVLM hidden states encode Eval2 target_slot.

This freezes a VLM, extracts prompt/image hidden representations, and trains a
small linear classifier for:

    hidden -> left / middle / right

It supports two input conditions:
  - scaffold: the VQA training wrapper that the strict model handles well
  - bare: only the original Eval2-style instruction text
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


DATASET_ID = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
MODEL_ID = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
COLOR_ABBR = {"b": "blue", "g": "green", "r": "red"}
SLOTS = ("left", "middle", "right")
SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}

SCAFFOLD_SYSTEM = (
    "You are looking at a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Given the image and the original robot instruction, identify the target bowl. "
    "Reply with exactly two short lines, not JSON and not prose. "
    "The exact format is: "
    "target color: green\n"
    "target slot: right"
)

PROMPT_VARIANTS = [
    ("ordinal", "Put the banana into the 2nd bowl from the left from the robot perspective"),
    ("ordinal", "Put the banana into the second bowl from the left from the robot perspective"),
    ("ordinal", "Put the banana into the middle bowl from the robot perspective"),
    ("ordinal", "Put the banana into the center bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl on the right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl to the right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl immediately right of the red bowl from the robot perspective"),
    ("relative", "Put the banana into the bowl just to the right of the red bowl from the robot perspective"),
    ("negation", "Put the banana into the bowl that is not green and not blue."),
    ("negation", "Put the banana into the bowl that is not blue and not green."),
    ("negation", "Put the banana into the bowl that is neither green nor blue."),
    ("negation", "Put the banana into the non-green and non-blue bowl."),
]


def split_for_replicate(rep: int) -> str:
    if rep <= 3:
        return "train"
    if rep == 4:
        return "val"
    return "test"


def resolve(prompt: str, layout: list[str]) -> tuple[str, str] | None:
    text = prompt.lower()
    if (
        "2nd bowl from the left" in text
        or "second bowl from the left" in text
        or "middle bowl" in text
        or "center bowl" in text
    ):
        return layout[1], "middle"
    if "right of the red bowl" in text:
        red_idx = layout.index("red")
        target_idx = red_idx + 1
        if target_idx >= len(layout):
            return None
        return layout[target_idx], SLOTS[target_idx]
    if (
        "not green and not blue" in text
        or "not blue and not green" in text
        or "neither green nor blue" in text
        or "non-green and non-blue" in text
    ):
        return "red", SLOTS[layout.index("red")]
    raise ValueError(prompt)


def load_episode_truth(dataset_id: str) -> dict[int, dict]:
    map_path = hf_hub_download(
        dataset_id,
        "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv",
        repo_type="dataset",
    )
    combo_df = pd.read_csv(map_path)
    truth = {}
    for _, row in combo_df.iterrows():
        layout = [COLOR_ABBR[ch] for ch in str(row["bowl_order_left_to_right"])]
        start = int(row["episode_start"])
        for episode_index in range(start, int(row["episode_end"]) + 1):
            truth[episode_index] = {
                "combo": str(row["combo"]),
                "layout_code": str(row["bowl_order_left_to_right"]),
                "layout_left_to_right": layout,
                "replicate_index": episode_index - start,
            }
    return truth


def to_pil_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy()
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return Image.fromarray(arr.astype("uint8")).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(value)}")


def build_prompt(processor, instruction: str, condition: str) -> str:
    if condition == "bare":
        text = instruction
    elif condition == "scaffold":
        text = f"{SCAFFOLD_SYSTEM}\n\nInstruction: {instruction}"
    else:
        raise ValueError(condition)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_rows(dataset_id: str, max_episodes: int = 0) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_id)
    truth = load_episode_truth(dataset_id)
    rows = []
    for ep in dataset.meta.episodes:
        episode_index = int(ep["episode_index"])
        if max_episodes and episode_index >= max_episodes:
            continue
        first_index = int(ep["dataset_from_index"])
        item = dataset[first_index]
        image = to_pil_image(item["observation.images.front"])
        info = truth[episode_index]
        for prompt_type, prompt in PROMPT_VARIANTS:
            resolved = resolve(prompt, info["layout_left_to_right"])
            if resolved is None:
                continue
            target_color, target_slot = resolved
            rows.append(
                {
                    "episode_index": episode_index,
                    "global_index": first_index,
                    "combo": info["combo"],
                    "layout_code": info["layout_code"],
                    "layout_left_to_right": info["layout_left_to_right"],
                    "replicate_index": info["replicate_index"],
                    "split": split_for_replicate(int(info["replicate_index"])),
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "target_color": target_color,
                    "target_slot": target_slot,
                    "image": image,
                }
            )
    return rows


@torch.inference_mode()
def extract_features(model, processor, rows: list[dict], condition: str, device: str, batch_size: int) -> dict:
    model.eval()
    features = []
    labels = []
    metadata = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [build_prompt(processor, row["prompt"], condition) for row in batch]
        images = [[row["image"]] for row in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = outputs.hidden_states[-1].float()
        mask = inputs["attention_mask"].bool()
        lengths = mask.long().sum(dim=1) - 1
        last = hidden[torch.arange(hidden.shape[0], device=device), lengths]
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        feat = torch.cat([last, mean], dim=-1).cpu()
        features.append(feat)
        labels.extend(SLOT_TO_ID[row["target_slot"]] for row in batch)
        for row in batch:
            metadata.append(
                {
                    "episode_index": row["episode_index"],
                    "combo": row["combo"],
                    "layout_code": row["layout_code"],
                    "split": row["split"],
                    "prompt_type": row["prompt_type"],
                    "prompt": row["prompt"],
                    "target_color": row["target_color"],
                    "target_slot": row["target_slot"],
                }
            )
        print(f"  extract {condition}: {min(start + batch_size, len(rows)):4d}/{len(rows)}", end="\r", flush=True)
    print()
    return {"x": torch.cat(features, dim=0), "y": torch.tensor(labels, dtype=torch.long), "metadata": metadata}


def train_probe(x_train, y_train, x_val, y_val, epochs: int, lr: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    model = nn.Linear(x_train.shape[1], 3)
    model.register_buffer("feature_mean", mean)
    model.register_buffer("feature_std", std)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x_train_n, y_train), batch_size=128, shuffle=True)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy(model, x_val_n, y_val)["accuracy"]
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.inference_mode()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    if hasattr(model, "feature_mean"):
        x = (x - model.feature_mean) / model.feature_std
    logits = model(x)
    pred = logits.argmax(dim=-1)
    correct = pred.eq(y)
    return {
        "correct": int(correct.sum().item()),
        "total": int(y.numel()),
        "accuracy": float(correct.float().mean().item()) if y.numel() else 0.0,
        "pred": pred.cpu().tolist(),
    }


def breakdown(pred: list[int], y: torch.Tensor, metadata: list[dict]) -> dict:
    buckets = {
        "prompt_type": defaultdict(lambda: [0, 0]),
        "target_slot": defaultdict(lambda: [0, 0]),
        "layout_code": defaultdict(lambda: [0, 0]),
    }
    y_list = y.cpu().tolist()
    for p, gt, meta in zip(pred, y_list, metadata, strict=True):
        ok = int(p == gt)
        for field, bucket in buckets.items():
            key = meta[field]
            bucket[key][0] += ok
            bucket[key][1] += 1
    return {
        field: {
            key: {"correct": c, "total": t, "accuracy": c / t if t else 0.0}
            for key, (c, t) in sorted(bucket.items())
        }
        for field, bucket in buckets.items()
    }


def hidden_separation(x: torch.Tensor, y: torch.Tensor, metadata: list[dict]) -> dict:
    """Measure same-image different-prompt separation with class centroids."""
    x_n = nn.functional.normalize(x, dim=-1)
    centroids = {}
    for slot, idx in SLOT_TO_ID.items():
        mask = y == idx
        centroids[slot] = x_n[mask].mean(dim=0)
        centroids[slot] = nn.functional.normalize(centroids[slot], dim=0)
    sims = torch.stack([x_n @ centroids[slot] for slot in SLOTS], dim=-1)
    margin = []
    for i, gt in enumerate(y.tolist()):
        gt_score = sims[i, gt].item()
        other = max(sims[i, j].item() for j in range(3) if j != gt)
        margin.append(gt_score - other)
    same_image_groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        same_image_groups[meta["episode_index"]].append(i)
    intra = []
    for indices in same_image_groups.values():
        for a in indices:
            for b in indices:
                if a < b and y[a] != y[b]:
                    intra.append(float(torch.dist(x_n[a], x_n[b]).item()))
    return {
        "centroid_margin_mean": float(np.mean(margin)) if margin else 0.0,
        "centroid_margin_min": float(np.min(margin)) if margin else 0.0,
        "same_image_diff_slot_distance_mean": float(np.mean(intra)) if intra else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--conditions", nargs="+", default=["scaffold", "bare"], choices=["scaffold", "bare"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="logs/vlm_target_slot_hidden_probe.json")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading rows from {args.dataset}")
    rows = build_rows(args.dataset, max_episodes=args.max_episodes)
    print(f"Rows: {len(rows)} counts={dict(Counter(row['split'] for row in rows))}")
    print(f"Loading model {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    report = {"model": args.model, "dataset": args.dataset, "conditions": {}}
    for condition in args.conditions:
        data = extract_features(model, processor, rows, condition, device, args.batch_size)
        split_idx = {split: [i for i, m in enumerate(data["metadata"]) if m["split"] == split] for split in ("train", "val", "test")}
        x_train, y_train = data["x"][split_idx["train"]], data["y"][split_idx["train"]]
        x_val, y_val = data["x"][split_idx["val"]], data["y"][split_idx["val"]]
        x_test, y_test = data["x"][split_idx["test"]], data["y"][split_idx["test"]]
        probe = train_probe(x_train, y_train, x_val, y_val, args.probe_epochs, args.probe_lr, args.seed)
        val_acc = accuracy(probe, x_val, y_val)
        test_acc = accuracy(probe, x_test, y_test)
        val_meta = [data["metadata"][i] for i in split_idx["val"]]
        test_meta = [data["metadata"][i] for i in split_idx["test"]]
        condition_report = {
            "feature_dim": int(data["x"].shape[1]),
            "counts": {
                "split": dict(Counter(m["split"] for m in data["metadata"])),
                "target_slot": dict(Counter(m["target_slot"] for m in data["metadata"])),
                "prompt_type": dict(Counter(m["prompt_type"] for m in data["metadata"])),
            },
            "val": {k: v for k, v in val_acc.items() if k != "pred"},
            "test": {k: v for k, v in test_acc.items() if k != "pred"},
            "val_breakdown": breakdown(val_acc["pred"], y_val, val_meta),
            "test_breakdown": breakdown(test_acc["pred"], y_test, test_meta),
            "separation_all": hidden_separation(data["x"], data["y"], data["metadata"]),
        }
        report["conditions"][condition] = condition_report
        print(
            f"{condition}: val={val_acc['accuracy']*100:.1f}% "
            f"test={test_acc['accuracy']*100:.1f}%"
        )
        print("  test by prompt type:", condition_report["test_breakdown"]["prompt_type"])
        print("  test by target slot:", condition_report["test_breakdown"]["target_slot"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
