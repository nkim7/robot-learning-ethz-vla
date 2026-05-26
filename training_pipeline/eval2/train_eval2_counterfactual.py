#!/usr/bin/env python3
"""Stage 2 counterfactual same-observation VLA training for Eval2.

The core idea: for each layout, use the SAME canonical early-frame
observation (image + state, from a left-slot episode at lf=0-4) paired
with ALL THREE target prompts and their corresponding action labels.
This forces the model to learn prompt-conditioned routing rather than
state-conditioned routing, because the state signal is identical across
the three prompt/action pairs.

Architecture:
  - Backbone : smolvla-eval2-strict10-eval1-init (eval1 expert weights)
  - LoRA r=16 : text_model q/v projections only  (opens text pathway)
  - modules_to_save : action projections, expert, state_proj
  - Target-slot aux head : nn.Linear(expert_hidden=720, 3)
  - Lambda aux : 0.05–0.10 (default 0.075)

Data composition per batch:
  - 75% counterfactual triplets  (N_triplets×3, same group = same canonical obs)
  - 25% full-trajectory replay   (uniform sampling from base dataset)

Run name: eval2_strict10_counterfactual_initial_lora_aux_v1
Ablation:  eval2_strict10_lora_expert_full_v2  (ordinary BC, no counterfactuals)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION


SLOTS = ("left", "middle", "right")
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}

# Ordinal prompts indexed by SLOT position (0=left,1=middle,2=right)
ORDINAL_PROMPTS = {
    "left":   "Put the banana in the leftmost bowl.",
    "middle": "Put the banana in the middle bowl.",
    "right":  "Put the banana in the rightmost bowl.",
}


# ---------------------------------------------------------------------------
# Counterfactual anchor specification
# ---------------------------------------------------------------------------

class CfSpec(NamedTuple):
    """A single counterfactual training sample specification."""
    canonical_global_idx: int   # global idx for image + state (always same within group)
    action_global_start: int    # global idx for start of action chunk
    action_ep_len: int          # episode length (for padding boundary)
    prompt: str                 # task/prompt string (overrides canonical item's task)
    target_slot_id: int         # 0=left, 1=middle, 2=right
    layout_id: str
    group_id: int               # anchors sharing same canonical_global_idx get same group_id


def build_ep_starts(meta: LeRobotDatasetMetadata) -> dict[int, int]:
    """Return {episode_index: first_global_frame_index}."""
    return {row["episode_index"]: row["dataset_from_index"] for row in meta.episodes}


def build_ep_lengths(meta: LeRobotDatasetMetadata) -> dict[int, int]:
    """Return {episode_index: n_frames}."""
    return {row["episode_index"]: row["length"] for row in meta.episodes}


def build_counterfactual_specs(
    target_df: pd.DataFrame,
    ep_starts: dict[int, int],
    ep_lengths: dict[int, int],
    n_anchor_frames: int = 5,
    include_ordinal: bool = True,
) -> list[CfSpec]:
    """Build counterfactual sample specifications.

    For each layout:
      - Pair episodes by within-slot index: (ep_left[i], ep_middle[i], ep_right[i])
      - For each lf in [0 .. n_anchor_frames-1]:
          canonical obs = ep_left[i][lf]   (home position, same across all slots)
          left   sample: prompt=left_color,   action from ep_left[i][lf]
          middle sample: prompt=middle_color, action from ep_middle[i][lf]
          right  sample: prompt=right_color,  action from ep_right[i][lf]
      - Optionally repeat with canonical from ep_middle[i] and ep_right[i] for
        more diversity (canonical_rotation=True).

    Returns list of CfSpec, sorted by group_id so triplets are consecutive.
    """
    # Per-layout, per-slot: sorted list of episode indices
    layout_slot_eps: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    # Per-(layout, episode): (slot, color, prompt)
    ep_meta: dict[int, dict] = {}

    for _, row in target_df.drop_duplicates("episode_index").iterrows():
        ep = int(row["episode_index"])
        lid = str(row["layout_id"])
        slot = str(row["target_slot"])
        layout_slot_eps[lid][slot].append(ep)
        ep_meta[ep] = {
            "slot": slot,
            "color": str(row["target_color"]),
            "prompt_color": str(row["prompt"]),
            "layout_id": lid,
        }

    # Sort episode lists within each slot
    for lid in layout_slot_eps:
        for slot in layout_slot_eps[lid]:
            layout_slot_eps[lid][slot].sort()

    specs: list[CfSpec] = []
    group_id = 0

    for lid, slot_eps in layout_slot_eps.items():
        # Require all 3 slots present
        if not all(s in slot_eps for s in SLOTS):
            print(f"[warn] layout {lid} missing slots, skipping")
            continue

        n_pairs = min(len(slot_eps[s]) for s in SLOTS)
        for pair_i in range(n_pairs):
            ep_per_slot = {s: slot_eps[s][pair_i] for s in SLOTS}

            # Rotate canonical: use each slot's episode as canonical once
            # (all start at home, so image/state should be ~identical)
            canonical_slots = ["left"]  # conservative: only left-slot as canonical

            for canonical_slot in canonical_slots:
                ep_canonical = ep_per_slot[canonical_slot]
                for lf in range(n_anchor_frames):
                    canonical_global_idx = ep_starts[ep_canonical] + lf
                    color_group_specs = []
                    for target_slot in SLOTS:
                        ep_target = ep_per_slot[target_slot]
                        action_global_start = ep_starts[ep_target] + lf
                        color_group_specs.append(CfSpec(
                            canonical_global_idx=canonical_global_idx,
                            action_global_start=action_global_start,
                            action_ep_len=ep_lengths[ep_target],
                            prompt=ep_meta[ep_target]["prompt_color"],
                            target_slot_id=SLOT_TO_ID[target_slot],
                            layout_id=lid,
                            group_id=group_id,
                        ))
                    specs.extend(color_group_specs)
                    group_id += 1

                    if include_ordinal:
                        ordinal_group_specs = []
                        for target_slot in SLOTS:
                            ep_target = ep_per_slot[target_slot]
                            action_global_start = ep_starts[ep_target] + lf
                            ordinal_group_specs.append(CfSpec(
                                canonical_global_idx=canonical_global_idx,
                                action_global_start=action_global_start,
                                action_ep_len=ep_lengths[ep_target],
                                prompt=ORDINAL_PROMPTS[target_slot],
                                target_slot_id=SLOT_TO_ID[target_slot],
                                layout_id=lid,
                                group_id=group_id,
                            ))
                        specs.extend(ordinal_group_specs)
                        group_id += 1

    return specs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CounterfactualDataset(Dataset):
    """Combines counterfactual anchor samples with full-trajectory replay.

    Items [0 .. n_cf-1] are counterfactual samples.
    Items [n_cf .. n_cf+n_base-1] are full-trajectory samples from base dataset.
    """

    def __init__(
        self,
        base: LeRobotDataset,
        specs: list[CfSpec],
        ep_starts: dict[int, int],
        ep_lengths: dict[int, int],
        chunk_size: int = 50,
    ):
        self.base = base
        self.specs = specs
        self.n_cf = len(specs)
        self.chunk_size = chunk_size

        # Pre-load all actions from Arrow for fast counterfactual construction.
        # Shape: [total_frames, 6]
        print(f"Pre-loading actions ({base.meta.total_frames} frames)...", flush=True)
        self.all_actions = np.array(base.hf_dataset["action"])  # [N, 6]
        print(f"  action array: {self.all_actions.shape}", flush=True)

        # Per-episode slot labels for full-trajectory samples (aux loss)
        # Build from specs: ep → slot_id
        self.ep_slot: dict[int, int] = {}
        for spec in specs:
            ep_idx = spec.action_global_start - ep_starts.get(
                self._global_to_ep(spec.action_global_start, ep_starts, ep_lengths), 0
            )
            # Simpler: iterate target_df externally; for now build from spec
        # Build ep_slot from specs (action_global_start maps to episode)
        self._ep_starts = ep_starts
        self._ep_lengths = ep_lengths
        self._build_ep_slot(specs, ep_starts)

    def _global_to_ep(
        self,
        global_idx: int,
        ep_starts: dict[int, int],
        ep_lengths: dict[int, int],
    ) -> int:
        """Find episode index for a global frame index."""
        for ep, start in ep_starts.items():
            if start <= global_idx < start + ep_lengths[ep]:
                return ep
        raise ValueError(f"global_idx {global_idx} not in any episode")

    def _build_ep_slot(self, specs: list[CfSpec], ep_starts: dict[int, int]) -> None:
        """Build {episode_index: slot_id} from specs for full-trajectory tagging."""
        # Use action_global_start to determine episode and slot
        start_to_ep = {v: k for k, v in ep_starts.items()}
        for spec in specs:
            # Derive episode from action_global_start (which is ep_start[ep_target] + lf)
            ep_start = spec.action_global_start - (
                spec.action_global_start % 1  # subtract lf; we do a range check below
            )
        # Faster: build from target_branching_index externally by passing ep_slot dict
        # For now, build by scanning: for each spec, ep_start[ep_target] = spec.action_global_start - lf
        # We don't know lf directly, but we can search for the episode containing action_global_start
        checked = set()
        for spec in specs:
            g = spec.action_global_start
            if g in checked:
                continue
            checked.add(g)
            for ep, start in ep_starts.items():
                length = self._ep_lengths[ep]
                if start <= g < start + length:
                    self.ep_slot[ep] = spec.target_slot_id
                    break

    def __len__(self) -> int:
        return self.n_cf + len(self.base)

    def _get_action_chunk(self, action_global_start: int, action_ep_len: int) -> torch.Tensor:
        """Build a [chunk_size, 6] action tensor, padding at episode end."""
        # Episode boundary: episodes are contiguous in all_actions
        # Find episode start for the given action_global_start
        ep_start_g = None
        for ep, st in self._ep_starts.items():
            length = self._ep_lengths[ep]
            if st <= action_global_start < st + length:
                ep_start_g = st
                ep_end_g = st + length
                break
        if ep_start_g is None:
            raise ValueError(f"action_global_start={action_global_start} not in any episode")

        end = min(action_global_start + self.chunk_size, ep_end_g)
        chunk = self.all_actions[action_global_start:end]  # [k, 6], k <= chunk_size
        if len(chunk) < self.chunk_size:
            pad = np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return torch.tensor(chunk, dtype=torch.float32)  # [chunk_size, 6]

    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_cf:
            spec = self.specs[idx]
            # Load canonical observation (image + state + metadata)
            item = dict(self.base[spec.canonical_global_idx])
            # Override action with target-slot action chunk
            item[ACTION] = self._get_action_chunk(
                spec.action_global_start, spec.action_ep_len
            )
            # Override task/prompt with target-slot prompt
            item["task"] = spec.prompt
            # Add slot label and metadata
            item["target_slot_id"] = torch.tensor(spec.target_slot_id, dtype=torch.long)
            item["stage2_sample_type"] = "counterfactual"
            item["stage2_layout_id"] = spec.layout_id
            item["stage2_group_id"] = spec.group_id
            return item
        else:
            base_idx = idx - self.n_cf
            item = dict(self.base[base_idx])
            ep = int(item["episode_index"])
            slot_id = self.ep_slot.get(ep, 0)
            item["target_slot_id"] = torch.tensor(slot_id, dtype=torch.long)
            item["stage2_sample_type"] = "full"
            item["stage2_layout_id"] = ""
            item["stage2_group_id"] = -1
            return item


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

class CounterfactualBatchSampler(Sampler[list[int]]):
    """Samples batches with guaranteed triplet completeness.

    Each batch contains:
      - n_triplets × 3  counterfactual items (same group_id within triplet)
      - full_per_batch  full-trajectory items
    """

    def __init__(
        self,
        dataset: CounterfactualDataset,
        batch_size: int,
        cf_fraction: float = 0.75,
        steps: int = 20000,
        seed: int = 42,
    ) -> None:
        if batch_size < 3:
            raise ValueError("batch_size must be >= 3")
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.rng = random.Random(seed)

        # Build group_id → list of spec indices
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx, spec in enumerate(dataset.specs):
            self.groups[spec.group_id].append(idx)
        self.group_ids = sorted(
            gid for gid, idxs in self.groups.items() if len(idxs) == 3
        )
        if not self.group_ids:
            raise ValueError("No complete triplet groups found (need exactly 3 items per group_id)")

        # How many complete triplets per batch
        n_triplets = max(1, round(batch_size * cf_fraction / 3))
        self.cf_per_batch = min(batch_size, n_triplets * 3)
        self.full_per_batch = batch_size - self.cf_per_batch
        self.n_triplets = self.cf_per_batch // 3
        self.n_base = len(dataset.base)

        print(
            f"Sampler: {self.n_triplets} triplets × 3 = {self.cf_per_batch} cf + "
            f"{self.full_per_batch} full per batch, {len(self.group_ids)} unique groups",
            flush=True,
        )

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        cf_offset = 0  # counterfactual items are at [0..n_cf-1]
        full_offset = self.dataset.n_cf  # full-trajectory items at [n_cf..]
        for _ in range(self.steps):
            batch: list[int] = []
            # Sample n_triplets complete groups
            chosen_groups = [
                self.rng.choice(self.group_ids) for _ in range(self.n_triplets)
            ]
            for gid in chosen_groups:
                batch.extend(self.groups[gid])
            # Fill with full-trajectory samples
            for _ in range(self.full_per_batch):
                batch.append(full_offset + self.rng.randrange(self.n_base))
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]


# ---------------------------------------------------------------------------
# Utilities (mirrors Stage2 training script)
# ---------------------------------------------------------------------------

def collate_with_metadata(items: list[dict]) -> dict:
    meta = {
        "target_slot_id": torch.stack([item.pop("target_slot_id") for item in items]),
        "stage2_sample_type": [item.pop("stage2_sample_type") for item in items],
        "stage2_layout_id": [item.pop("stage2_layout_id") for item in items],
        "stage2_group_id": [item.pop("stage2_group_id") for item in items],
    }
    batch = torch.utils.data.default_collate(items)
    batch.update(meta)
    return batch


def add_slot_head_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    action_out = raw.model.action_out_proj

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    return action_out.register_forward_hook(hook)


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


def save_checkpoint(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual same-observation Stage2 training"
    )
    parser.add_argument("--init", default="/home/shadeform/outputs/smolvla-eval2-strict10-eval1-init")
    parser.add_argument("--dataset", default="robot-learning-group47/eval2_fixed_bowl_fixed_banana_108")
    parser.add_argument("--dataset-root", default="/home/shadeform/.cache/lerobot/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108")
    parser.add_argument("--target-index", default="/home/shadeform/data/eval2_fixed_target_branching/target_branching_index.csv")
    parser.add_argument("--out-dir", default="/home/shadeform/outputs/eval2_strict10_counterfactual_initial_lora_aux_v1")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cf-fraction", type=float, default=0.75,
                        help="Fraction of each batch that is counterfactual samples")
    parser.add_argument("--n-anchor-frames", type=int, default=5,
                        help="Number of early local frames per episode to use as anchors (lf=0..n-1)")
    parser.add_argument("--no-ordinal", action="store_true",
                        help="Disable ordinal (left/middle/rightmost) prompt variants")
    parser.add_argument("--lambda-slot", type=float, default=0.075)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-name", default="eval2_strict10_counterfactual_initial_lora_aux_v1")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # -----------------------------------------------------------------------
    # Load policy + dataset
    # -----------------------------------------------------------------------
    print("Loading policy...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device

    print("Loading dataset metadata...", flush=True)
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy.config, ds_meta)

    print("Loading dataset...", flush=True)
    base_ds = LeRobotDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_ts,
        video_backend="pyav",
        tolerance_s=1e-4,
    )

    # -----------------------------------------------------------------------
    # Build counterfactual specs
    # -----------------------------------------------------------------------
    print("Building counterfactual specs...", flush=True)
    ep_starts = build_ep_starts(ds_meta)
    ep_lengths = build_ep_lengths(ds_meta)
    target_df = pd.read_csv(args.target_index)
    specs = build_counterfactual_specs(
        target_df,
        ep_starts,
        ep_lengths,
        n_anchor_frames=args.n_anchor_frames,
        include_ordinal=not args.no_ordinal,
    )
    n_groups = len(set(s.group_id for s in specs))
    print(
        f"Counterfactual specs: {len(specs)} total, {n_groups} groups, "
        f"{len(specs) // 3 if n_groups else 0} triplets",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build dataset + sampler
    # -----------------------------------------------------------------------
    dataset = CounterfactualDataset(base_ds, specs, ep_starts, ep_lengths, chunk_size=50)
    sampler = CounterfactualBatchSampler(
        dataset,
        batch_size=args.batch_size,
        cf_fraction=args.cf_fraction,
        steps=args.steps,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        collate_fn=collate_with_metadata,
    )

    # -----------------------------------------------------------------------
    # Preprocessor
    # -----------------------------------------------------------------------
    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {
                "rename_map": {"observation.images.front": "observation.images.camera1"}
            },
        },
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    }
    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        **processor_kwargs,
    )

    # -----------------------------------------------------------------------
    # LoRA + slot head (identical to Stage2 training)
    # -----------------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=16,
        target_modules=r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj",
        modules_to_save=[
            "lm_expert",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ],
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg).to(device)
    raw = policy.get_base_model()
    slot_head = nn.Linear(raw.model.vlm_with_expert.expert_hidden_size, 3).to(device)
    storage: dict = {}
    hook_handle = add_slot_head_hook(policy, slot_head, storage)

    params = [p for p in policy.parameters() if p.requires_grad] + list(slot_head.parameters())
    opt = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=1e-10, betas=(0.9, 0.95), eps=1e-8
    )
    total_trainable = sum(p.numel() for p in params)
    print(f"Trainable params incl slot head: {total_trainable:,}", flush=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    policy.train()
    slot_head.train()
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        # Normalise uint8 images
        for key in list(base_ds.meta.camera_keys):
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0

        target_slot_id = batch.pop("target_slot_id").to(device)
        sample_types = batch.pop("stage2_sample_type")
        batch.pop("stage2_layout_id")
        group_ids = batch.pop("stage2_group_id")

        batch = preprocessor(batch)
        storage.clear()
        action_loss, output = policy.forward(batch)

        if "slot_logits" not in storage:
            raise RuntimeError("Slot hook did not fire; check action_out_proj hook")

        slot_logits = storage["slot_logits"]
        slot_loss = F.cross_entropy(slot_logits, target_slot_id)
        slot_acc = (slot_logits.argmax(dim=-1) == target_slot_id).float().mean()
        loss = action_loss + args.lambda_slot * slot_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            cf_mask = [t == "counterfactual" for t in sample_types]
            cf_count = sum(cf_mask)
            elapsed = time.time() - start
            metrics = {
                "loss": float(loss.item()),
                "action_loss": float(action_loss.item()),
                "slot_loss": float(slot_loss.item()),
                "slot_acc": float(slot_acc.item()),
                "grad_norm": float(grad_norm.item()),
                "cf_count": cf_count,
                "full_count": len(sample_types) - cf_count,
                "steps_per_s": step / max(elapsed, 1e-6),
            }
            print(
                f"step={step} loss={metrics['loss']:.4f} "
                f"action={metrics['action_loss']:.4f} "
                f"slot={metrics['slot_loss']:.4f} slot_acc={metrics['slot_acc']*100:.1f}% "
                f"cf/full={metrics['cf_count']}/{metrics['full_count']} "
                f"grad={metrics['grad_norm']:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log(metrics, step=step)

        if step % args.save_every == 0 or step == args.steps:
            ckpt = save_checkpoint(out_dir, step, policy, slot_head, init_path)
            print(f"saved {ckpt}", flush=True)

        if step >= args.steps:
            break

    hook_handle.remove()
    if wandb_run is not None:
        wandb_run.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
