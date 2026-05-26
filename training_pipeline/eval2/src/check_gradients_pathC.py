#!/usr/bin/env python3
"""
Path C smoke-test verifier.

Run this AFTER a ~200-step smoke run on the merged checkpoint:

    uv run python ~/check_gradients_pathC.py \
        --pathC_log /data/lerobot_outputs/train/pathC_smoke/... \
        --pathA_log /data/lerobot_outputs/train/pathA_smoke/...   # optional, for loss comparison

What it checks:
  1. VLM backbone params have non-None gradients    → co-adaptation is real
  2. lm_expert params have gradients                 → expert is training
  3. Path C step-0 loss < Path A step-0 loss         → eval1 expert warm-start is working
     (only if --pathA_log is provided)

Alternatively, use as a callback inside a custom training loop (see use_as_hook()).
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Hook-style usage: paste this into your training loop after loss.backward()
# and before optimizer.step()
# ---------------------------------------------------------------------------
INLINE_SNIPPET = """
# ── paste after loss.backward(), before optimizer.step() ─────────────────
backbone_grad_found = False
expert_grad_found   = False
for name, p in policy.named_parameters():
    if not p.requires_grad or p.grad is None:
        continue
    gnorm = p.grad.norm().item()
    if "vlm_with_expert.vlm" in name and not backbone_grad_found:
        print(f"✅ BACKBONE GRAD: {name}: {gnorm:.6f}")
        backbone_grad_found = True
    if "lm_expert" in name and not expert_grad_found:
        print(f"✅ EXPERT GRAD:   {name}: {gnorm:.6f}")
        expert_grad_found = True
    if backbone_grad_found and expert_grad_found:
        break

if not backbone_grad_found:
    print("❌ NO BACKBONE GRADIENT — train_expert_only=false did not take!")
    raise RuntimeError("Backbone has no gradients. Check policy config.")
if not expert_grad_found:
    print("❌ NO EXPERT GRADIENT — expert is frozen!")
    raise RuntimeError("Expert has no gradients.")
# ─────────────────────────────────────────────────────────────────────────
"""


def check_from_checkpoint(checkpoint_dir: Path) -> None:
    """
    Load the policy from a checkpoint and inspect gradient status after
    one synthetic forward+backward pass.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"\nLoading policy from: {checkpoint_dir}")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
    policy.train()

    # Check which params are trainable
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    frozen    = {n: p for n, p in policy.named_parameters() if not p.requires_grad}
    print(f"\nTrainable params: {sum(p.numel() for p in trainable.values()):,}")
    print(f"Frozen    params: {sum(p.numel() for p in frozen.values()):,}")

    backbone_trainable = [n for n in trainable if "vlm_with_expert.vlm" in n]
    expert_trainable   = [n for n in trainable if "lm_expert" in n]
    print(f"\nBackbone trainable layers: {len(backbone_trainable)}")
    print(f"Expert trainable layers:   {len(expert_trainable)}")

    if len(backbone_trainable) == 0:
        print("❌ Backbone is frozen! train_expert_only=false was not applied.")
        sys.exit(1)
    if len(expert_trainable) == 0:
        print("❌ Expert is frozen!")
        sys.exit(1)

    print("\n✅ Trainability check passed.")
    print("\nFirst 3 trainable backbone params:")
    for n in backbone_trainable[:3]:
        print(f"  {n}")
    print("\nFirst 3 trainable expert params:")
    for n in expert_trainable[:3]:
        print(f"  {n}")


def compare_initial_losses(pathC_log: Path, pathA_log: Path) -> None:
    """
    Compare step-0 (or earliest available) loss from two W&B-style JSON logs
    or lerobot CSV logs.
    """
    def read_first_loss(log_path: Path) -> float | None:
        # Try lerobot's train_stats.jsonl format
        jsonl = log_path / "train_stats.jsonl"
        if jsonl.exists():
            with open(jsonl) as f:
                for line in f:
                    row = json.loads(line)
                    if "loss" in row:
                        return float(row["loss"])
        # Try any .json in the dir with a 'loss' key
        for jf in log_path.glob("*.json"):
            try:
                data = json.loads(jf.read_text())
                if "loss" in data:
                    return float(data["loss"])
            except Exception:
                pass
        return None

    loss_C = read_first_loss(pathC_log)
    loss_A = read_first_loss(pathA_log)

    print(f"\n── Initial loss comparison ──")
    print(f"  Path C (merged init):     {loss_C}")
    print(f"  Path A (fresh expert):    {loss_A}")

    if loss_C is None or loss_A is None:
        print("  ⚠️  Could not read losses automatically.")
        print("     Check your W&B dashboard under project 'lerobot'.")
        print("     Path C step-0 loss SHOULD be lower than Path A step-0 loss.")
        print("     If they are equal, the merge may have silently failed.")
        return

    if loss_C < loss_A:
        delta = loss_A - loss_C
        print(f"  ✅ Path C starts lower by {delta:.4f} — eval1 expert warm-start is working.")
    else:
        delta = loss_C - loss_A
        print(f"  ⚠️  Path C loss ({loss_C:.4f}) ≥ Path A loss ({loss_A:.4f}) by {delta:.4f}.")
        print("     This suggests the merged checkpoint may not have loaded correctly.")
        print("     Re-run merge_vqa_backbone_into_eval1.py --dry-run and check key counts.")


def main():
    print("=" * 70)
    print("Path C verification: gradient check + loss baseline comparison")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a saved checkpoint dir to check trainability")
    parser.add_argument("--pathC_log", type=Path, default=None,
                        help="Path to Path C output dir (for initial loss check)")
    parser.add_argument("--pathA_log", type=Path, default=None,
                        help="Path to Path A output dir (for comparison baseline)")
    parser.add_argument("--print-snippet", action="store_true",
                        help="Print the inline gradient-check snippet and exit")
    args = parser.parse_args()

    if args.print_snippet:
        print("\nInline snippet to paste after loss.backward():\n")
        print(INLINE_SNIPPET)
        return

    if args.checkpoint:
        check_from_checkpoint(args.checkpoint)

    if args.pathC_log and args.pathA_log:
        compare_initial_losses(args.pathC_log, args.pathA_log)
    elif args.pathC_log or args.pathA_log:
        print("  ⚠️  Provide both --pathC_log and --pathA_log for loss comparison.")

    if not args.checkpoint and not args.pathC_log:
        print("\nNo arguments given. Use --print-snippet to get the inline gradient check.")
        print("Or pass --checkpoint <dir> to verify a saved checkpoint.")
        parser.print_help()


if __name__ == "__main__":
    main()
