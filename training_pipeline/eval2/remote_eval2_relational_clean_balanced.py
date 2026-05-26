#!/usr/bin/env python3
"""Clean Eval2 relational repair with fixed train/gate frame pools.

The sampler uses the position-only Eval2 dataset as the image/state source, then
overrides task text and trains a fresh q/v LoRA plus target-slot head with slot
loss only. A 360-example cycle gives the requested 60/20/15/5 family mix while
preserving exact slot/color/layout balance inside each family over every cycle.
For relative prompts, each layout's four valid prompts are expanded to six by
duplicating the edge-target cases; this changes the natural 1:2:1 target-slot
distribution into 2:2:2 without creating invalid relational prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

sys.path.insert(0, "/home/shadeform")

from eval2_layout_utils import COLOR_NAMES, SLOTS, normalize_layout
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLORS = ("blue", "green", "red")
SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}
DEFAULT_DATASET = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
DEFAULT_OUT = "/home/shadeform/outputs/eval2_stage2_relational_clean_60_20_15_5_fixedframe_balanced"


@dataclass(frozen=True)
class PromptCase:
    family: str
    layout_code: str
    prompt: str
    slot_id: int
    side: str = ""
    anchor: str = ""
    target_color: str = ""
    key: str = ""


@dataclass(frozen=True)
class TrainSample:
    frame_index: int
    case: PromptCase


def load_episode_layouts(dataset_root: Path) -> dict[int, str]:
    path = dataset_root / "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
    out: dict[int, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
                out[ep] = str(row["bowl_order_left_to_right"]).strip().lower()
    if len(out) != 108:
        raise RuntimeError(f"Expected 108 episode mappings, got {len(out)}")
    return out


def even_pick(values: list[int], n: int) -> list[int]:
    if len(values) < n:
        raise RuntimeError(f"Need {n} frames but only have {len(values)}")
    if n == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (n - 1))] for i in range(n)]


def build_or_load_split(
    ds: LeRobotDataset,
    dataset_root: Path,
    split_path: Path,
    train_per_layout: int,
    gate_per_layout: int,
    frame_min: int,
    frame_max: int,
) -> dict[str, dict[str, list[int]]]:
    if split_path.exists():
        with split_path.open() as f:
            split = json.load(f)
        print(f"Loaded fixed frame split: {split_path}")
        return split

    episode_to_layout = load_episode_layouts(dataset_root)
    candidates: dict[str, list[int]] = {code: [] for code in LAYOUT_CODES}
    ep_values = list(ds.hf_dataset["episode_index"])
    frame_values = list(ds.hf_dataset["frame_index"])
    for global_idx, (ep_val, fr_val) in enumerate(zip(ep_values, frame_values)):
        ep = int(ep_val)
        fr = int(fr_val)
        if frame_min <= fr <= frame_max:
            candidates[episode_to_layout[ep]].append(global_idx)

    split = {}
    for code in LAYOUT_CODES:
        chosen = even_pick(sorted(candidates[code]), train_per_layout + gate_per_layout)
        split[code] = {
            "train": chosen[:train_per_layout],
            "gate": chosen[train_per_layout:],
        }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    print(f"Wrote fixed frame split: {split_path}")
    return split


def make_cases_for_layout(code: str) -> dict[str, list[PromptCase]]:
    layout = normalize_layout(code)
    cases: dict[str, list[PromptCase]] = {k: [] for k in ("relative", "ordinal", "exclusion", "direct")}

    for ref_idx, anchor in enumerate(layout):
        if ref_idx > 0:
            cases["relative"].append(PromptCase(
                "relative", code,
                f"Put the banana into the bowl on the left of the {anchor} bowl from the robot perspective.",
                ref_idx - 1, side="left", anchor=anchor, target_color=layout[ref_idx - 1],
                key=f"left_of_{anchor}",
            ))
        if ref_idx < 2:
            cases["relative"].append(PromptCase(
                "relative", code,
                f"Put the banana into the bowl on the right of the {anchor} bowl from the robot perspective.",
                ref_idx + 1, side="right", anchor=anchor, target_color=layout[ref_idx + 1],
                key=f"right_of_{anchor}",
            ))

    for slot_id, ord_text in enumerate(("1st", "2nd", "3rd")):
        cases["ordinal"].append(PromptCase(
            "ordinal", code,
            f"Put the banana into the {ord_text} bowl from the left from the robot perspective.",
            slot_id, target_color=layout[slot_id], key=f"ord_{SLOTS[slot_id]}",
        ))

    for target in COLORS:
        excluded = [c for c in COLORS if c != target]
        cases["exclusion"].append(PromptCase(
            "exclusion", code,
            f"Put the banana into the bowl that is not {excluded[0]} and not {excluded[1]}.",
            layout.index(target), target_color=target, key=f"not_{excluded[0]}_not_{excluded[1]}",
        ))
        cases["direct"].append(PromptCase(
            "direct", code,
            f"Put the banana into the {target} bowl.",
            layout.index(target), target_color=target, key=f"direct_{target}",
        ))

    assert len(cases["relative"]) == 4
    return cases


ALL_CASES = {code: make_cases_for_layout(code) for code in LAYOUT_CODES}


class BalancedCycleSampler:
    def __init__(self, train_frames: dict[str, list[int]], seed: int):
        self.train_frames = train_frames
        self.rng = random.Random(seed)
        self.ptr = {code: 0 for code in LAYOUT_CODES}
        self.deck: list[TrainSample] = []

    def _frame(self, code: str) -> int:
        frames = self.train_frames[code]
        idx = self.ptr[code] % len(frames)
        self.ptr[code] += 1
        return frames[idx]

    def refill(self) -> None:
        deck: list[TrainSample] = []
        # 216 relative = 6 repeats × 6 layouts × 6 balanced prompt instances.
        # The four valid prompts per layout naturally produce target slots 1:2:1.
        # Duplicate edge-target cases so each layout contributes 2 left, 2 middle,
        # and 2 right targets while preserving the same-frame four-prompt set.
        for _ in range(6):
            for code in LAYOUT_CODES:
                frame = self._frame(code)
                for case in ALL_CASES[code]["relative"]:
                    deck.append(TrainSample(frame, case))
                    if case.slot_id in (SLOT_TO_ID["left"], SLOT_TO_ID["right"]):
                        deck.append(TrainSample(frame, case))
        # 72 ordinal = 4 repeats × 6 layouts × 3 slots.
        for _ in range(4):
            for code in LAYOUT_CODES:
                frame = self._frame(code)
                for case in ALL_CASES[code]["ordinal"]:
                    deck.append(TrainSample(frame, case))
        # 54 exclusion = 3 repeats × 6 layouts × 3 target colors.
        for _ in range(3):
            for code in LAYOUT_CODES:
                frame = self._frame(code)
                for case in ALL_CASES[code]["exclusion"]:
                    deck.append(TrainSample(frame, case))
        # 18 direct = 1 repeat × 6 layouts × 3 target colors.
        for code in LAYOUT_CODES:
            frame = self._frame(code)
            for case in ALL_CASES[code]["direct"]:
                deck.append(TrainSample(frame, case))
        assert len(deck) == 360
        self.rng.shuffle(deck)
        self.deck.extend(deck)

    def next(self, n: int) -> list[TrainSample]:
        while len(self.deck) < n:
            self.refill()
        out = self.deck[:n]
        del self.deck[:n]
        return out


def _fetch_item(ds: LeRobotDataset, sample: TrainSample) -> dict:
    item = dict(ds[sample.frame_index])
    item["task"] = sample.case.prompt
    item.pop("task_index", None)
    return item


def collate_batch(items: list[dict]) -> dict:
    from torch.utils.data import default_collate
    tasks = [item.pop("task") for item in items]
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


def configure_trainability(policy, train_connector: bool) -> None:
    raw = policy.get_base_model()
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    if train_connector:
        connector = raw.model.vlm_with_expert.get_vlm_model().connector
        for p in connector.parameters():
            p.requires_grad = True
        print(f"connector unfrozen: {sum(p.numel() for p in connector.parameters()):,} params")
    raw.model.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.model.vlm_with_expert.lm_expert.eval()


def add_slot_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model()

    def hook(_module, inputs, _output):
        storage["slot_logits"] = slot_head(inputs[0].float().mean(dim=1))

    return raw.model.action_out_proj.register_forward_hook(hook)


def copy_processors(src: Path, dst: Path) -> None:
    for name in (
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "config.json",
    ):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def save_ckpt(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> None:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    print(f"saved {ckpt}", flush=True)


def summarize_samples(samples: list[TrainSample]) -> dict:
    out = {
        "family": Counter(s.case.family for s in samples),
        "slot": Counter(SLOTS[s.case.slot_id] for s in samples),
        "relative_side": Counter(s.case.side for s in samples if s.case.family == "relative"),
        "relative_anchor": Counter(s.case.anchor for s in samples if s.case.family == "relative"),
        "layout": Counter(s.case.layout_code for s in samples),
    }
    return {k: dict(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=DEFAULT_INIT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--split-path", default="/home/shadeform/eval3_repair_work/eval2_clean_fixed_frame_split.json")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--train-connector", action="store_true")
    ap.add_argument("--connector-lr", type=float, default=1e-6)
    ap.add_argument("--save-steps", default="500,1000,1500,2000,3000,5000")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lerobot")
    ap.add_argument("--wandb-name", default="eval2-relational-clean")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    print(f"Loading dataset {args.dataset}")
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(SmolVLAPolicy.from_pretrained(args.init).config, ds_meta)
    ds = LeRobotDataset(args.dataset, root=args.dataset_root, delta_timestamps=delta_ts, video_backend="pyav")
    split = build_or_load_split(ds, Path(args.dataset_root), Path(args.split_path), 30, 10, 5, 80)
    train_frames = {code: split[code]["train"] for code in LAYOUT_CODES}
    with (out_dir / "fixed_frame_split.json").open("w") as f:
        json.dump(split, f, indent=2, sort_keys=True)

    sampler = BalancedCycleSampler(train_frames, args.seed)
    probe_samples = sampler.next(360)
    print("One-cycle composition:", json.dumps(summarize_samples(probe_samples), sort_keys=True))
    sampler.deck = []
    sampler.ptr = {code: 0 for code in LAYOUT_CODES}
    if args.dry_run:
        return

    print(f"Loading policy from {args.init}")
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=(r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj"),
        lora_dropout=0.05,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    configure_trainability(policy, train_connector=args.train_connector)
    policy = policy.to(device)
    raw = policy.get_base_model()
    slot_head = nn.Linear(raw.model.vlm_with_expert.expert_hidden_size, len(SLOTS)).to(device)
    storage: dict = {}
    hook = add_slot_hook(policy, slot_head, storage)

    lora_params = [p for p in policy.parameters() if p.requires_grad]
    trainable = lora_params + list(slot_head.parameters())
    if args.train_connector:
        conn_params = [p for p in raw.model.vlm_with_expert.get_vlm_model().connector.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": lora_params + list(slot_head.parameters()), "lr": args.lr},
                {"params": conn_params, "lr": min(args.connector_lr, 1e-6)},
            ],
            weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8,
        )
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
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

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    policy.train()
    slot_head.train()
    raw.model.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.model.vlm_with_expert.lm_expert.eval()
    if not args.train_connector:
        raw.model.vlm_with_expert.get_vlm_model().connector.eval()

    print(f"Training {args.steps} steps, batch={args.batch_size}, slot loss only")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        samples = sampler.next(args.batch_size)
        items = [_fetch_item(ds, s) for s in samples]
        for item in items:
            for key in ds.meta.camera_keys:
                if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                    item[key] = item[key].float() / 255.0
        batch = collate_batch(items)
        targets = torch.tensor([s.case.slot_id for s in samples], dtype=torch.long, device=device)
        storage.clear()
        batch = preprocessor(batch)
        _, _ = policy.forward(batch)
        if "slot_logits" not in storage:
            raise RuntimeError("Slot hook did not fire")
        loss = F.cross_entropy(storage["slot_logits"], targets)
        acc = (storage["slot_logits"].argmax(dim=-1) == targets).float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            fam = Counter(s.case.family for s in samples)
            metrics = {
                "slot_loss": float(loss.item()),
                "slot_acc": float(acc.item()),
                "grad_norm": float(grad.item()),
                "steps_per_s": step / max(time.time() - t0, 1e-6),
            }
            print(
                f"step={step:5d} slot={metrics['slot_loss']:.4f} acc={metrics['slot_acc']*100:5.1f}% "
                f"grad={metrics['grad_norm']:.2f} fam={dict(fam)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log(metrics, step=step)
        if step in save_steps or step == args.steps:
            save_ckpt(out_dir, step, policy, slot_head, init_path)

    hook.remove()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
