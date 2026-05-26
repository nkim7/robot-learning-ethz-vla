#!/usr/bin/env python3
"""Eval2 Stage B v1: conservative action co-training from Stage2 + Stage A LoRA.

This is one-policy training, not a resolver:
  image + raw prompt -> SmolVLA -> action chunk

Stage A's relational LoRA is inserted into the Stage2-20k motion policy and
co-trained with action loss.  The Stage A pair head is used only as an auxiliary
training loss and is saved separately for diagnostics; it is not needed for
inference.

Loss:
  relative / clockwise scaffold:
      L_action_demo + pair_weight * L_pair
  non-relative preservation:
      L_action_demo + pair_weight * L_pair + beta_distill * L_stage2_distill

All prompts are action-consistent with the recorded demo target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

sys.path.insert(0, "/home/shadeform")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from train_eval2_stageA_head_relational_grounding import (
    COLORS,
    SLOTS,
    vlm_last_text_hidden,
)
from train_eval2_stageA_head_relational_grounding_v4 import (
    PAIR_TO_ID,
    TargetHeadsV4,
)


COLOR_FULL = {"b": "blue", "g": "green", "r": "red"}
SLOT_NAMES = ("left", "middle", "right")
ORDINALS = ("1st", "2nd", "3rd")
COMBO_CSV = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50

DEFAULT_DATASET = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_DATASET_ROOT = (
    "/home/shadeform/lerobot_cache/"
    "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
)
DEFAULT_STAGE2_BASE = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/"
    "checkpoints/020000/pretrained_model"
)
DEFAULT_STAGEA = (
    "/home/shadeform/outputs/eval2_stageA_head_relational_grounding_v6_clockwise_from_v5_300/"
    "checkpoints/000100/pretrained_model"
)
DEFAULT_OUT = "/home/shadeform/outputs/eval2_stageB_pair_aux_cotrain_v1_from_v6_100"
DEFAULT_COMBO_MAP = (
    "/home/shadeform/.cache/lerobot/robot-learning-group47/"
    "eval2_fixed_bowl_fixed_banana_108/combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
)

FAMILY_WEIGHTS = {
    "relative": 0.50,
    "direct": 0.10,
    "ordinal": 0.10,
    "exclusion": 0.10,
    "slot_color": 0.10,
    "clockwise_scaffold": 0.10,
}
DISTILL_FAMILIES = {"direct", "ordinal", "exclusion", "slot_color"}


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    layout: tuple[str, str, str]
    layout_code: str
    target_color: str
    target_slot: str
    target_slot_idx: int


@dataclass
class FrameRecord:
    global_idx: int
    episode_index: int
    frame_index: int
    meta: EpisodeMeta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Eval2 Stage B pair-aux action co-training")
    ap.add_argument("--stage2-base", default=DEFAULT_STAGE2_BASE)
    ap.add_argument("--stageA", default=DEFAULT_STAGEA)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--combo-map", default=DEFAULT_COMBO_MAP)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-lr", type=float, default=1e-6)
    ap.add_argument("--expert-lr", type=float, default=5e-7)
    ap.add_argument("--proj-lr", type=float, default=5e-7)
    ap.add_argument("--pair-head-lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-scale", type=float, default=0.1)
    ap.add_argument("--pair-weight", type=float, default=0.2)
    ap.add_argument("--beta-distill", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-10)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--frame-min", type=int, default=5)
    ap.add_argument("--frame-max", type=int, default=80)
    ap.add_argument("--save-steps", default="100,200,300,500")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--use-amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="use_amp")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--dry-inspect", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lerobot")
    ap.add_argument("--wandb-name", default="eval2-stageB-pair-aux-cotrain-v1")
    return ap.parse_args()


def load_episode_meta(dataset_root: Path, combo_map: str | None = None) -> dict[int, EpisodeMeta]:
    candidates = []
    if combo_map:
        candidates.append(Path(combo_map))
    candidates.append(dataset_root / COMBO_CSV)
    candidates.append(
        Path("/home/shadeform/.cache/lerobot/robot-learning-group47/")
        / "eval2_fixed_bowl_fixed_banana_108"
        / COMBO_CSV
    )
    csv_path = next((p for p in candidates if p.exists()), candidates[0])
    out: dict[int, EpisodeMeta] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find combo map; tried: {[str(p) for p in candidates]}")
    print(f"Using combo map: {csv_path}", flush=True)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            layout_code = row["bowl_order_left_to_right"].strip().lower()
            layout = tuple(COLOR_FULL[ch] for ch in layout_code)
            target_color = row["target_bowl_color"].strip().lower()
            target_slot_idx = layout.index(target_color)
            meta_kwargs = dict(
                layout=layout,
                layout_code=layout_code,
                target_color=target_color,
                target_slot=SLOT_NAMES[target_slot_idx],
                target_slot_idx=target_slot_idx,
            )
            for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
                out[ep] = EpisodeMeta(episode_index=ep, **meta_kwargs)
    return out


def valid_relative_prompts(meta: EpisodeMeta) -> list[str]:
    prompts = []
    ti = meta.target_slot_idx
    if ti > 0:
        anchor = meta.layout[ti - 1]
        prompts.append(
            f"Put the banana into the bowl on the right of the {anchor} bowl "
            f"from the robot perspective."
        )
    if ti < 2:
        anchor = meta.layout[ti + 1]
        prompts.append(
            f"Put the banana into the bowl on the left of the {anchor} bowl "
            f"from the robot perspective."
        )
    return prompts


def clockwise_scaffold_prompts(meta: EpisodeMeta) -> list[str]:
    prompts = []
    order = " -> ".join(meta.layout)
    ti = meta.target_slot_idx
    if ti > 0:
        anchor = meta.layout[ti - 1]
        prompts.extend([
            (
                f"Treat right_of as the next bowl clockwise from the anchor bowl. "
                f"Clockwise bowl order is {order}. Put the banana into the bowl "
                f"on the right of the {anchor} bowl."
            ),
            (
                f"Read the bowls in clockwise order around the arc. Put the banana "
                f"into the bowl on the right of the {anchor} bowl."
            ),
            (
                f"Use clockwise bowl order, not the visually rightmost bowl. Put "
                f"the banana into the bowl on the right of the {anchor} bowl."
            ),
        ])
    if ti < 2:
        anchor = meta.layout[ti + 1]
        prompts.extend([
            (
                f"Treat left_of as the previous bowl in clockwise order from the "
                f"anchor bowl. Clockwise bowl order is {order}. Put the banana "
                f"into the bowl on the left of the {anchor} bowl."
            ),
            (
                f"Read the bowls in clockwise order around the arc. Put the banana "
                f"into the bowl on the left of the {anchor} bowl."
            ),
            (
                f"Use clockwise bowl order, not horizontal image position. Put "
                f"the banana into the bowl on the left of the {anchor} bowl."
            ),
        ])
    return prompts


def direct_prompt(meta: EpisodeMeta) -> str:
    return f"Put the banana into the {meta.target_color} bowl."


def ordinal_prompt(meta: EpisodeMeta) -> str:
    return (
        f"Put the banana into the {ORDINALS[meta.target_slot_idx]} bowl "
        f"from the left from the robot perspective."
    )


def exclusion_prompt(meta: EpisodeMeta) -> str:
    excluded = [c for c in COLORS if c != meta.target_color]
    return f"Put the banana into the bowl that is not {excluded[0]} and not {excluded[1]}."


def slot_color_prompt(meta: EpisodeMeta) -> str:
    return f"Put the banana into the {meta.target_slot} {meta.target_color} bowl."


def sample_prompt(meta: EpisodeMeta, rng: random.Random) -> tuple[str, str]:
    roll = rng.random()
    acc = 0.0
    family = "clockwise_scaffold"
    for fam, weight in FAMILY_WEIGHTS.items():
        acc += weight
        if roll < acc:
            family = fam
            break
    if family == "relative":
        return family, rng.choice(valid_relative_prompts(meta))
    if family == "direct":
        return family, direct_prompt(meta)
    if family == "ordinal":
        return family, ordinal_prompt(meta)
    if family == "exclusion":
        return family, exclusion_prompt(meta)
    if family == "slot_color":
        return family, slot_color_prompt(meta)
    return family, rng.choice(clockwise_scaffold_prompts(meta))


def build_frame_pool(
    ds: LeRobotDataset,
    ep_meta: dict[int, EpisodeMeta],
    frame_min: int,
    frame_max: int,
) -> dict[str, list[FrameRecord]]:
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    pool: dict[str, list[FrameRecord]] = {}
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep = int(ep_val)
        fr = int(fr_val)
        if fr < frame_min or fr > frame_max:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        key = f"{meta.layout_code}_{meta.target_color[0]}"
        pool.setdefault(key, []).append(FrameRecord(gi, ep, fr, meta))
    return pool


class BalancedSampler:
    def __init__(self, pool: dict[str, list[FrameRecord]], seed: int):
        self.pool = {k: list(v) for k, v in pool.items() if v}
        self.keys = sorted(self.pool)
        self.ptrs = {k: 0 for k in self.keys}
        self.rng = random.Random(seed)
        for key in self.keys:
            self.rng.shuffle(self.pool[key])

    def next_batch(self, n: int) -> list[tuple[FrameRecord, str, str]]:
        keys = [self.keys[i % len(self.keys)] for i in range(n)]
        self.rng.shuffle(keys)
        out = []
        for key in keys:
            frames = self.pool[key]
            rec = frames[self.ptrs[key] % len(frames)]
            self.ptrs[key] += 1
            family, prompt = sample_prompt(rec.meta, self.rng)
            out.append((rec, family, prompt))
        return out


def fetch_item(ds: LeRobotDataset, rec: FrameRecord, prompt: str) -> dict:
    item = dict(ds[rec.global_idx])
    item["task"] = prompt
    item.pop("task_index", None)
    return item


def collate_batch(items: list[dict]) -> dict:
    from torch.utils.data import default_collate

    tasks = [item.pop("task") for item in items]
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


def scale_images(items: list[dict], camera_keys: list[str]) -> None:
    for item in items:
        for key in camera_keys:
            if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                item[key] = item[key].float() / 255.0


class VelocityCapture:
    def __init__(self):
        self.vel: torch.Tensor | None = None
        self._handle = None

    def register(self, module: torch.nn.Module) -> None:
        def _hook(_mod, _inp, output):
            self.vel = output
        self._handle = module.register_forward_hook(_hook)

    def clear(self) -> None:
        self.vel = None

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()


def configure_trainability(policy, heads: TargetHeadsV4) -> None:
    raw = policy.get_base_model().model
    for p in policy.parameters():
        p.requires_grad = False
    for p in heads.parameters():
        p.requires_grad = False
    for p in heads.pair_head.parameters():
        p.requires_grad = True
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    for p in raw.vlm_with_expert.lm_expert.parameters():
        p.requires_grad = True
    for sub in (
        raw.state_proj,
        raw.action_in_proj,
        raw.action_out_proj,
        raw.action_time_mlp_in,
        raw.action_time_mlp_out,
    ):
        for p in sub.parameters():
            p.requires_grad = True
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def build_param_groups(policy, heads: TargetHeadsV4, args: argparse.Namespace) -> list[dict]:
    raw = policy.get_base_model().model
    lora = [(n, p) for n, p in policy.named_parameters() if p.requires_grad and "lora_" in n]
    expert_ids = {id(p) for p in raw.vlm_with_expert.lm_expert.parameters()}
    expert = [(n, p) for n, p in policy.named_parameters() if p.requires_grad and id(p) in expert_ids]
    proj_modules = (
        raw.state_proj,
        raw.action_in_proj,
        raw.action_out_proj,
        raw.action_time_mlp_in,
        raw.action_time_mlp_out,
    )
    proj_ids = {id(p) for module in proj_modules for p in module.parameters()}
    proj = [(n, p) for n, p in policy.named_parameters() if p.requires_grad and id(p) in proj_ids]
    head = [(f"target_heads.{n}", p) for n, p in heads.named_parameters() if p.requires_grad]

    def count(named):
        return sum(p.numel() for _, p in named)

    print("TRAINABLE_PARAMETER_SUMMARY", flush=True)
    print(f"  lora:       {count(lora):,} lr={args.lora_lr}", flush=True)
    print(f"  lm_expert:  {count(expert):,} lr={args.expert_lr}", flush=True)
    print(f"  action_proj:{count(proj):,} lr={args.proj_lr}", flush=True)
    print(f"  pair_heads: {count(head):,} lr={args.pair_head_lr}", flush=True)
    print(f"  total:      {count(lora)+count(expert)+count(proj)+count(head):,}", flush=True)
    print("TRAINABLE_PARAMETER_NAMES", flush=True)
    for named in (lora, expert, proj, head):
        for name, p in named:
            print(f"  {name} ({p.numel():,})", flush=True)

    return [
        {"params": [p for _, p in lora], "lr": args.lora_lr, "name": "lora"},
        {"params": [p for _, p in expert], "lr": args.expert_lr, "name": "lm_expert"},
        {"params": [p for _, p in proj], "lr": args.proj_lr, "name": "action_proj"},
        {"params": [p for _, p in head], "lr": args.pair_head_lr, "name": "pair_head"},
    ]


def set_lrs(optimizer, step: int, args: argparse.Namespace) -> None:
    bases = [args.lora_lr, args.expert_lr, args.proj_lr, args.pair_head_lr]
    if step <= args.warmup_steps:
        scale = max(step, 1) / max(args.warmup_steps, 1)
    else:
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        scale = args.min_lr_scale + 0.5 * (1.0 - args.min_lr_scale) * (1.0 + math.cos(math.pi * progress))
    for group, base in zip(optimizer.param_groups, bases, strict=True):
        group["lr"] = base * scale


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


def save_ckpt(out_dir: Path, step: int, policy, heads: TargetHeadsV4, stage2_base: Path) -> None:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy_copy = copy.deepcopy(policy)
    merged = policy_copy.merge_and_unload()
    merged.save_pretrained(ckpt)
    if hasattr(merged, "config"):
        merged.config.save_pretrained(ckpt)
    copy_processors(stage2_base, ckpt)
    torch.save(heads.state_dict(), ckpt / "stageB_aux_target_heads.pt")
    del policy_copy, merged
    print(f"CHECKPOINT step={step} path={ckpt}", flush=True)


def pair_targets(samples: list[tuple[FrameRecord, str, str]], device: str) -> torch.Tensor:
    ids = [
        PAIR_TO_ID[f"{rec.meta.target_slot}_{rec.meta.target_color}"]
        for rec, _, _ in samples
    ]
    return torch.tensor(ids, dtype=torch.long, device=device)


def distill_mask(samples: list[tuple[FrameRecord, str, str]], device: str) -> torch.Tensor:
    vals = [fam in DISTILL_FAMILIES for _, fam, _ in samples]
    return torch.tensor(vals, dtype=torch.bool, device=device)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage2_base = Path(args.stage2_base)
    stagea_path = Path(args.stageA)
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    print("STAGE_B_SELECTION", flush=True)
    print(f"  stage2_base: {stage2_base}", flush=True)
    print(f"  stageA_semantic: {stagea_path}", flush=True)
    print("  stageA_reason: v6@100 is the best pair-primary gate checkpoint in logs; v6@150/200 regress bgr.", flush=True)
    print("  inference_contract: one SmolVLA policy; no resolver, no prompt rewriting, no pair_head at inference.", flush=True)

    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    tmp = SmolVLAPolicy.from_pretrained(str(stage2_base))
    delta_ts = resolve_delta_timestamps(tmp.config, ds_meta)
    del tmp
    ds = LeRobotDataset(args.dataset, root=args.dataset_root, delta_timestamps=delta_ts, video_backend="pyav")
    ep_meta = load_episode_meta(Path(args.dataset_root), args.combo_map)
    pool = build_frame_pool(ds, ep_meta, args.frame_min, args.frame_max)
    sampler = BalancedSampler(pool, args.seed)

    probe = sampler.next_batch(1800)
    fam_cnt = Counter(f for _, f, _ in probe)
    layout_cnt = Counter(r.meta.layout_code for r, _, _ in probe)
    target_cnt = Counter(f"{r.meta.target_slot}_{r.meta.target_color}" for r, _, _ in probe)
    consistency_errors = 0
    for rec, fam, prompt in probe:
        allowed = {
            direct_prompt(rec.meta),
            ordinal_prompt(rec.meta),
            exclusion_prompt(rec.meta),
            slot_color_prompt(rec.meta),
            *valid_relative_prompts(rec.meta),
            *clockwise_scaffold_prompts(rec.meta),
        }
        consistency_errors += int(prompt not in allowed)
    report = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "combo_map": args.combo_map,
        "frames_in_pool": sum(len(v) for v in pool.values()),
        "combo_keys": {k: len(v) for k, v in sorted(pool.items())},
        "prompt_mix_target": FAMILY_WEIGHTS,
        "dry_prompt_counts_1800": dict(fam_cnt),
        "dry_layout_counts_1800": dict(layout_cnt),
        "dry_target_pair_counts_1800": dict(target_cnt),
        "action_consistency_errors_in_dry_probe": consistency_errors,
        "counterfactual_action_mismatched_samples": "excluded by construction",
        "distill_families": sorted(DISTILL_FAMILIES),
        "frozen": [
            "vision encoder",
            "connector",
            "base VLM weights outside LoRA",
        ],
        "trainable": [
            "VLM q/v LoRA",
            "lm_expert/action expert",
            "state/action/time/action_out projections",
            "target pair head for auxiliary loss only",
        ],
        "loss": {
            "action_demo_weight": 1.0,
            "pair_semantic_weight": args.pair_weight,
            "stage2_distill_weight_non_relative_only": args.beta_distill,
        },
        "output_dir": str(out_dir),
        "wandb_name": args.wandb_name,
    }
    print("STAGE_B_DATA_REPORT", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    (out_dir / "stageB_data_report.json").write_text(json.dumps(report, indent=2))

    if args.dry_inspect:
        print("DRY_INSPECT_DATA_ONLY_DONE", flush=True)

    print(f"Loading Stage2 base policy: {stage2_base}", flush=True)
    base_policy = SmolVLAPolicy.from_pretrained(str(stage2_base))
    base_policy.config.device = device
    base_policy.config.train_expert_only = True
    base_policy.config.freeze_vision_encoder = True

    print(f"Loading Stage A adapter into Stage2 base: {stagea_path}", flush=True)
    policy = PeftModel.from_pretrained(base_policy, str(stagea_path), is_trainable=True)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    raw = policy.get_base_model().model
    hidden_size = raw.vlm_with_expert.config.text_config.hidden_size

    heads = TargetHeadsV4(hidden_size, dropout=0.05)
    heads_path = stagea_path / "stageA_v4_target_heads.pt"
    state = torch.load(heads_path, map_location="cpu")
    missing, unexpected = heads.load_state_dict(state, strict=False)
    print(f"Loaded Stage A aux heads from {heads_path}; missing={missing} unexpected={unexpected}", flush=True)

    configure_trainability(policy, heads)
    policy = policy.to(device)
    heads = heads.to(device)
    raw = policy.get_base_model().model

    param_groups = build_param_groups(policy, heads, args)
    if args.dry_inspect:
        print("DRY_INSPECT_MODEL_DONE", flush=True)
        return

    teacher_policy = None
    teacher_raw = None
    teacher_cap = VelocityCapture()
    if args.beta_distill > 0:
        print(f"Loading frozen Stage2 teacher: {stage2_base}", flush=True)
        teacher_policy = SmolVLAPolicy.from_pretrained(str(stage2_base)).to(device)
        for p in teacher_policy.parameters():
            p.requires_grad = False
        teacher_policy.eval()
        teacher_raw = teacher_policy.model
        teacher_cap.register(teacher_raw.action_out_proj)

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(stage2_base),
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

    student_cap = VelocityCapture()
    student_cap.register(raw.action_out_proj)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    policy.train()
    heads.train()
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()
    use_bf16 = args.use_amp and device == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 else torch.amp.autocast("cuda", enabled=False)
    )
    trainable_params = [p for p in list(policy.parameters()) + list(heads.parameters()) if p.requires_grad]
    print(f"TRAINING_START steps={args.steps} batch={args.batch_size} amp={use_bf16} out={out_dir}", flush=True)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        set_lrs(optimizer, step, args)
        samples = sampler.next_batch(args.batch_size)
        items = [fetch_item(ds, rec, prompt) for rec, _, prompt in samples]
        scale_images(items, ds.meta.camera_keys)
        batch = collate_batch(items)
        batch = preprocessor(batch)
        target_pair = pair_targets(samples, device)
        nonrel_mask = distill_mask(samples, device)

        noise = torch.randn(args.batch_size, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
        time_ = torch.rand(args.batch_size, device=device)

        student_cap.clear()
        with autocast_ctx:
            loss_action, _ = policy.forward(batch, noise=noise, time=time_)
            hidden = vlm_last_text_hidden(policy, batch)
            _, _, pair_logits = heads(hidden)
            loss_pair = F.cross_entropy(pair_logits.float(), target_pair)

        loss_distill = torch.tensor(0.0, device=device)
        if teacher_policy is not None and nonrel_mask.any():
            teacher_cap.clear()
            with torch.no_grad(), autocast_ctx:
                teacher_policy.forward(batch, noise=noise, time=time_)
            if student_cap.vel is not None and teacher_cap.vel is not None:
                per_sample = (student_cap.vel.float() - teacher_cap.vel.float().detach()).pow(2).mean(dim=(1, 2))
                loss_distill = per_sample[nonrel_mask].mean()

        loss = loss_action + args.pair_weight * loss_pair + args.beta_distill * loss_distill

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            fam = Counter(f for _, f, _ in samples)
            pair_acc = (pair_logits.argmax(dim=-1) == target_pair).float().mean().item()
            elapsed = time.time() - t0
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            print(
                f"step={step} loss={float(loss):.4f} action={float(loss_action):.4f} "
                f"pair={float(loss_pair):.4f} distill={float(loss_distill):.5f} "
                f"pair_acc={100*pair_acc:.1f}% grad={float(grad_norm):.2f} "
                f"lrs={lrs} fam={dict(fam)} sps={step/max(elapsed,1e-6):.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "loss/total": float(loss),
                    "loss/action_demo": float(loss_action),
                    "loss/pair_semantic": float(loss_pair),
                    "loss/stage2_distill": float(loss_distill),
                    "train/pair_acc": pair_acc,
                    "grad_norm": float(grad_norm),
                    **{f"lr/{k}": v for k, v in lrs.items()},
                    **{f"family/{k}": v for k, v in fam.items()},
                }, step=step)

        if step in save_steps or step == args.steps:
            save_ckpt(out_dir, step, policy, heads, stage2_base)

    student_cap.remove()
    teacher_cap.remove()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"TRAINING_DONE out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
