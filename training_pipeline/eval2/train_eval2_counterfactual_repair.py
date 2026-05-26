#!/usr/bin/env python3
"""Eval2 Stage 2C: Counterfactual paired relational repair.

Key design vs Stage 2B:
  - SAME frame appears with ALL 4 relative prompt variants for its layout.
    The model cannot rely on episode-level image features; it MUST route by
    the anchor color + direction in the prompt.
  - Blue anchor oversampled 2x early in training (gates showed 0% blue accuracy).
  - Slot loss ONLY — no action loss.  Action loss conflicted because the original
    demo actions target a different slot than the counterfactual prompt.
  - All other choices preserved: LoRA rank-16 on VLM q/v, slot head on
    action_out_proj input, connector and action expert frozen.

Batch structure (default --batch-base=24, --blue-oversample=2):
  6 layouts × 4 prompts = 24 base samples (same frame within each layout quad)
  + 6 layouts × n_blue_per_layout × (oversample-1) extra blue-anchor samples
  Total ≈ 32 per step.

Save at --save-every (default 500). Gate is run externally by run_stage2c.sh.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

sys.path.insert(0, "/home/shadeform")
sys.path.insert(0, "/home/shadeform/lerobot/src")

from probe_eval2_relational_diagnosis import (
    augmented_episode_info, DATASET_REPO,
)
from eval2_layout_utils import SLOTS, normalize_layout
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ── constants ──────────────────────────────────────────────────────────────────
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}
LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")

DEFAULT_INIT = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = (
    "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented"
)
DEFAULT_OUT = "/home/shadeform/outputs/eval2_stage2c_counterfactual_v2"


# ── build relative-prompt quads ────────────────────────────────────────────────
def _build_quads() -> dict[str, list[dict]]:
    """For each layout, return the 4 relative prompt cases (matching gate exactly)."""
    quads: dict[str, list[dict]] = {}
    for code in LAYOUT_CODES:
        layout = normalize_layout(code)   # e.g. ["blue","green","red"]
        cases = []
        for ref_idx, ref_color in enumerate(layout):
            if ref_idx > 0:  # left_of_ref exists
                cases.append(dict(
                    key=f"left_of_{ref_color}",
                    prompt=(f"Put the banana into the bowl on the left of the "
                            f"{ref_color} bowl from the robot perspective."),
                    slot_id=ref_idx - 1,
                    anchor=ref_color,
                ))
            if ref_idx < 2:  # right_of_ref exists
                cases.append(dict(
                    key=f"right_of_{ref_color}",
                    prompt=(f"Put the banana into the bowl on the right of the "
                            f"{ref_color} bowl from the robot perspective."),
                    slot_id=ref_idx + 1,
                    anchor=ref_color,
                ))
        assert len(cases) == 4, f"Expected 4 cases for {code}, got {len(cases)}"
        quads[code] = cases
    return quads


RELATIVE_QUADS = _build_quads()


# ── frame pool ─────────────────────────────────────────────────────────────────
def build_frame_pool(ds: LeRobotDataset) -> dict[str, list[int]]:
    """Map layout_code -> sorted list of global frame indices in ds."""
    pool: dict[str, list[int]] = {c: [] for c in LAYOUT_CODES}
    ep_col = list(ds.hf_dataset["episode_index"])
    for global_idx, ep_val in enumerate(ep_col):
        layout, _lid, _color, _slot, _orig = augmented_episode_info(int(ep_val))
        code = "".join(c[0] for c in layout)
        pool[code].append(global_idx)
    sizes = {c: len(v) for c, v in pool.items()}
    print(f"Frame pool sizes: {sizes}")
    assert all(v > 0 for v in sizes.values()), f"Empty pool for some layout: {sizes}"
    return pool


# ── batch generation ───────────────────────────────────────────────────────────
def make_batch(
    ds: LeRobotDataset,
    frame_pool: dict[str, list[int]],
    rng: random.Random,
    blue_oversample: int,
    camera_keys: list[str],
    task_lookup: dict[int, str] | None,
) -> tuple[list[torch.Tensor | str], list[int]]:
    """Build one counterfactual batch: 24 base + extra blue samples.

    Returns (raw_items, target_slot_ids) where raw_items are dicts ready
    for collation.
    """
    raw_items: list[dict] = []
    slot_ids:  list[int]  = []

    for code in LAYOUT_CODES:
        frames = frame_pool[code]
        f1 = rng.choice(frames)

        for case in RELATIVE_QUADS[code]:
            item = _fetch_item(ds, f1, case["prompt"], camera_keys)
            raw_items.append(item)
            slot_ids.append(case["slot_id"])

        if blue_oversample > 1:
            f2 = rng.choice(frames)
            for case in RELATIVE_QUADS[code]:
                if case["anchor"] == "blue":
                    for _ in range(blue_oversample - 1):
                        item = _fetch_item(ds, f2, case["prompt"], camera_keys)
                        raw_items.append(item)
                        slot_ids.append(case["slot_id"])

    # Shuffle together
    combined = list(zip(raw_items, slot_ids))
    rng.shuffle(combined)
    raw_items, slot_ids = zip(*combined)
    return list(raw_items), list(slot_ids)


def _fetch_item(
    ds: LeRobotDataset,
    global_idx: int,
    task_text: str,
    camera_keys: list[str],
) -> dict:
    """Fetch frame from dataset and override the task with our prompt."""
    item = dict(ds[global_idx])
    # Override task (str key used by SmolVLA preprocessor)
    item["task"] = task_text
    item.pop("task_index", None)   # remove integer key to avoid conflict
    return item


def collate_batch(items: list[dict], device: str) -> dict:
    """Stack items into a batched dict (handles uint8 images and string tasks)."""
    from torch.utils.data import default_collate
    tasks = [it.pop("task") for it in items]
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


# ── LoRA + slot head ───────────────────────────────────────────────────────────
def configure_trainability(policy, train_connector: bool = False) -> None:
    """Freeze all; unfreeze LoRA params and optionally the VLM connector.

    Frozen always:
      - vision encoder
      - base VLM weights (except LoRA deltas)
      - action expert (lm_expert)

    Trainable always:
      - VLM q/v LoRA parameters
      - target slot head (managed separately)

    Trainable when train_connector=True:
      - vlm_with_expert.vlm connector (visual-text bridge)
        Enables the model to update how visual features map into the text
        embedding space, which helps ground anchor-color positions in the
        image when learning left_of/right_of routing.
    """
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
        n = sum(p.numel() for p in connector.parameters())
        print(f"  connector unfrozen: {n:,} params")
    # Keep vision encoder and action expert in eval mode (affects dropout/BN)
    raw.model.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.model.vlm_with_expert.lm_expert.eval()


def add_slot_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model()

    def hook(_module, inputs, _output):
        h = inputs[0].float()
        storage["slot_logits"] = slot_head(h.mean(dim=1))

    return raw.model.action_out_proj.register_forward_hook(hook)


# ── checkpoint helpers ─────────────────────────────────────────────────────────
def copy_processors(src: Path, dst: Path) -> None:
    for name in [
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "config.json",
    ]:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def save_ckpt(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    print(f"  saved → {ckpt}", flush=True)
    return ckpt


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=DEFAULT_INIT,
                        help="Stage 2 base checkpoint to repair")
    parser.add_argument("--dataset", default=DATASET_REPO)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-base", type=int, default=24,
                        help="6 layouts × 4 prompts = 24; must be divisible by 24")
    parser.add_argument("--blue-oversample", type=int, default=2,
                        help="Duplicate blue-anchor cases N-1 extra times per layout")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak LR (lowered from v1 1e-4 to reduce grad-norm spikes)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--train-connector", action="store_true",
                        help="Unfreeze the VLM visual-text connector for better "
                             "anchor-color grounding")
    parser.add_argument("--slot-weights", type=float, nargs=3,
                        default=[2.0, 1.0, 2.0],
                        metavar=("W_LEFT", "W_MID", "W_RIGHT"),
                        help="Cross-entropy class weights for [left, middle, right]. "
                             "Default 2:1:2 upweights left/right to break middle-default.")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-name", default="eval2-stage2c-counterfactual-v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print batch stats and trainable params, then exit")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ────────────────────────────────────────────────────────────────
    print(f"Loading dataset {args.dataset} …")
    ds_meta  = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(
        SmolVLAPolicy.from_pretrained(args.init).config, ds_meta
    )
    base_ds = LeRobotDataset(
        args.dataset, root=args.dataset_root, delta_timestamps=delta_ts,
        video_backend="pyav",
    )
    frame_pool   = build_frame_pool(base_ds)
    camera_keys  = list(base_ds.meta.camera_keys)

    # ── policy ─────────────────────────────────────────────────────────────────
    print(f"Loading policy from {args.init} …")
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=(
            r"model\.vlm_with_expert\.vlm\.model\.text_model"
            r"\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        ),
        lora_dropout=0.05,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    configure_trainability(policy, train_connector=args.train_connector)
    policy = policy.to(device)

    raw = policy.get_base_model()
    expert_hidden = raw.model.vlm_with_expert.expert_hidden_size
    slot_head = nn.Linear(expert_hidden, len(SLOTS)).to(device)
    storage: dict = {}
    hook = add_slot_hook(policy, slot_head, storage)

    trainable_params = [p for p in policy.parameters() if p.requires_grad] + \
                       list(slot_head.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_trainable:,} "
          f"(LoRA rank-{args.lora_rank} + slot head"
          f"{' + connector' if args.train_connector else ''})")

    # Class weights for slot loss — break the middle-default shortcut
    slot_class_weights = torch.tensor(args.slot_weights, dtype=torch.float32, device=device)
    print(f"Slot class weights:   left={args.slot_weights[0]}  "
          f"middle={args.slot_weights[1]}  right={args.slot_weights[2]}")

    # Verify quad coverage
    blue_per_batch = sum(
        1 for code in LAYOUT_CODES
        for c in RELATIVE_QUADS[code] if c["anchor"] == "blue"
    )
    total_per_batch = 24 + blue_per_batch * (args.blue_oversample - 1)
    print(f"Batch composition: 24 base + {blue_per_batch*(args.blue_oversample-1)} "
          f"blue-extras = {total_per_batch} per step")
    print(f"Blue fraction: {(blue_per_batch*args.blue_oversample)/total_per_batch*100:.1f}%")

    if args.dry_run:
        print("\nDRY RUN — printing 1 batch slot distribution then exiting.")
        items, sids = make_batch(base_ds, frame_pool, rng, args.blue_oversample,
                                 camera_keys, None)
        from collections import Counter
        slot_dist  = Counter(SLOTS[sid] for sid in sids)
        anchor_dist = Counter()
        for code in LAYOUT_CODES:
            for c in RELATIVE_QUADS[code]:
                if c["anchor"] == "blue":
                    anchor_dist["blue"] += args.blue_oversample
                else:
                    anchor_dist[c["anchor"]] += 1
        print(f"  Slot distribution: {dict(slot_dist)}")
        print(f"  Anchor distribution: {dict(anchor_dist)}")
        hook.remove()
        return

    # ── preprocessor ───────────────────────────────────────────────────────────
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats":    base_ds.meta.stats,
                "features": {**policy.config.input_features,
                             **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # ── optimizer ──────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.95), eps=1e-8,
    )

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_name,
            config=vars(args),
        )

    # ── training loop ──────────────────────────────────────────────────────────
    policy.train()
    slot_head.train()
    # Frozen modules stay in eval mode regardless of policy.train()
    raw.model.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.model.vlm_with_expert.lm_expert.eval()
    # If connector is trainable, leave it in train mode (already set by policy.train())
    # If connector is frozen, put it in eval mode to disable any dropout
    if not args.train_connector:
        raw.model.vlm_with_expert.get_vlm_model().connector.eval()

    print(f"\nTraining for {args.steps} steps …")
    print(f"  init:            {args.init}")
    print(f"  out_dir:         {out_dir}")
    print(f"  LR:              {args.lr}   LoRA rank: {args.lora_rank}")
    print(f"  blue_oversample: {args.blue_oversample}x")
    print(f"  train_connector: {args.train_connector}")
    print(f"  slot weights:    left={args.slot_weights[0]}  "
          f"middle={args.slot_weights[1]}  right={args.slot_weights[2]}")
    print(f"  slot loss ONLY (no action loss)")
    print()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Build counterfactual batch
        raw_items, slot_ids_list = make_batch(
            base_ds, frame_pool, rng, args.blue_oversample, camera_keys, None
        )

        # Convert images from uint8 to float in each item
        for item in raw_items:
            for key in camera_keys:
                if key in item and isinstance(item[key], torch.Tensor) \
                        and item[key].dtype == torch.uint8:
                    item[key] = item[key].float() / 255.0

        # Collate
        batch = collate_batch(raw_items, device)
        target_slot_ids = torch.tensor(slot_ids_list, dtype=torch.long, device=device)

        # Preprocess (normalise state, etc.)
        storage.clear()
        batch = preprocessor(batch)

        # Forward pass — action_loss is computed but NOT added to the gradient
        _, _ = policy.forward(batch)

        if "slot_logits" not in storage:
            raise RuntimeError("Slot head hook did not fire during forward pass")

        # Slot loss only — weighted to break middle-default shortcut
        slot_loss = F.cross_entropy(
            storage["slot_logits"], target_slot_ids,
            weight=slot_class_weights,
        )
        slot_acc  = (storage["slot_logits"].argmax(dim=-1) == target_slot_ids).float().mean()

        opt.zero_grad(set_to_none=True)
        slot_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            # Per-anchor accuracy for diagnostic logging
            logits_np = storage["slot_logits"].detach().cpu()
            preds = logits_np.argmax(dim=-1).tolist()
            per_anchor: dict[str, list[bool]] = {c: [] for c in ("blue", "green", "red")}
            base_cases = [(c, case) for c in LAYOUT_CODES for case in RELATIVE_QUADS[c]]
            for i_case, (_, case) in enumerate(base_cases[:len(preds)]):
                ok = preds[i_case] == case["slot_id"]
                per_anchor[case["anchor"]].append(ok)
            anchor_acc = {
                c: (sum(v) / len(v) * 100 if v else float("nan"))
                for c, v in per_anchor.items()
            }

            metrics = {
                "slot_loss":  float(slot_loss.item()),
                "slot_acc":   float(slot_acc.item()),
                "grad_norm":  float(grad_norm.item()),
                "steps_per_s": step / max(time.time() - t0, 1e-6),
                "blue_acc":   anchor_acc.get("blue", float("nan")),
                "green_acc":  anchor_acc.get("green", float("nan")),
                "red_acc":    anchor_acc.get("red", float("nan")),
            }
            print(
                f"step={step:5d}  slot={metrics['slot_loss']:.4f}  "
                f"acc={metrics['slot_acc']*100:5.1f}%  "
                f"blue={metrics['blue_acc']:5.1f}%  "
                f"green={metrics['green_acc']:5.1f}%  "
                f"red={metrics['red_acc']:5.1f}%  "
                f"grad={metrics['grad_norm']:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log(metrics, step=step)

        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out_dir, step, policy, slot_head, init_path)

    hook.remove()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"\nDone. Last checkpoint: {out_dir}/checkpoints/{args.steps:06d}/pretrained_model")


if __name__ == "__main__":
    main()
