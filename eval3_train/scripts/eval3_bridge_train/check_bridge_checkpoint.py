#!/usr/bin/env python3
"""Inspect a bridge-trained SmolVLA checkpoint.

Prints:
  - spatial_head weights (should differ from zero-init after training)
  - Slot prediction on dummy inputs (sanity check)
  - spatial_acc from training logs if available
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = sorted((output_dir / "checkpoints").glob("*/pretrained_model"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pretrained_model dir.  Defaults to latest in --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Bridge training output dir (used when --checkpoint is omitted).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.output_dir:
        ckpt_path = _find_latest_checkpoint(Path(args.output_dir))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {args.output_dir}")
    else:
        parser.error("Provide --checkpoint or --output-dir")

    print(f"Checkpoint: {ckpt_path}")

    # ── Load weights without full model init ────────────────────────────────
    import torch
    from safetensors.torch import load_file

    model_file = ckpt_path / "model.safetensors"
    if not model_file.exists():
        # Try split shards
        shards = sorted(ckpt_path.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors found in {ckpt_path}")
        state = {}
        for s in shards:
            state.update(load_file(s))
    else:
        state = load_file(model_file)

    # ── spatial_head ────────────────────────────────────────────────────────
    sh_w = state.get("model.spatial_head.weight")
    sh_b = state.get("model.spatial_head.bias")

    if sh_w is None:
        print("WARNING: spatial_head.weight not found in checkpoint!")
        print("  Keys starting with 'model.spatial': ",
              [k for k in state if "spatial" in k])
    else:
        print(f"\nspatial_head.weight  shape={tuple(sh_w.shape)}")
        print(f"  row norms: {sh_w.norm(dim=1).tolist()}")
        print(f"  bias:      {sh_b.tolist() if sh_b is not None else 'None'}")
        # Argmax of each row — which slot each row most resembles
        import torch
        print(f"  Argmax per slot row: {sh_w.argmax(dim=1).tolist()}  (0=left 1=middle 2=right)")

    # ── Training log spatial_acc ────────────────────────────────────────────
    if args.output_dir:
        log_file = Path(args.output_dir) / "train_log.json"
        if not log_file.exists():
            log_file = Path(args.output_dir) / "metrics.json"
        if log_file.exists():
            try:
                log = json.loads(log_file.read_text())
                # Extract spatial_acc entries
                accs = [(e.get("step", i), e["spatial_acc"])
                        for i, e in enumerate(log)
                        if "spatial_acc" in e]
                if accs:
                    print(f"\nspatial_acc from log (first/last 5):")
                    for step, acc in accs[:5]:
                        print(f"  step={step:6d}  spatial_acc={acc:.3f}")
                    if len(accs) > 5:
                        print("  ...")
                        for step, acc in accs[-5:]:
                            print(f"  step={step:6d}  spatial_acc={acc:.3f}")
            except Exception as e:
                print(f"  (could not parse log: {e})")

    # ── Other key weight counts ─────────────────────────────────────────────
    total = sum(v.numel() for v in state.values())
    expert_keys = [k for k in state if "lm_expert" in k]
    expert_params = sum(state[k].numel() for k in expert_keys)
    vlm_lora_keys = [k for k in state if "vlm" in k and "lora_" in k]
    print(f"\nTotal params in checkpoint : {total:,}")
    print(f"Action expert params       : {expert_params:,}  ({len(expert_keys)} tensors)")
    print(f"VLM LoRA keys              : {len(vlm_lora_keys)}")


if __name__ == "__main__":
    main()
