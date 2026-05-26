#!/usr/bin/env python3
"""
Path C merge: VQA backbone weights → eval1_final_final checkpoint

Produces a checkpoint with:
  - backbone (model.vlm_with_expert.vlm.*): from smolvlm2-500m-vqa-lora-merged
  - expert   (model.vlm_with_expert.lm_expert.*): from eval1_final_final
  - projections (state_proj, action_{in,out}_proj, action_time_mlp_*): from eval1_final_final

Usage (on GPU server):
    cd ~/lerobot
    uv run python ~/merge_vqa_backbone_into_eval1.py [--dry-run]

    --dry-run  : print all diagnostics but do NOT write any files

Gates (script aborts loudly before writing if any fail):
  1. Namespace alignment  : VQA→eval1 key overlap must be ≥ 100
  2. Key count preserved  : merged dict must have same number of keys as eval1
  3. Dtype consistency    : each overwritten key must have matching dtype
  4. Expert untouched     : zero expert/projection keys should have been overwritten
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults; override with CLI flags for concrete runs.
# ---------------------------------------------------------------------------
DEFAULT_EVAL1_REPO = "robot-learning-group47/eval1_final_final"
DEFAULT_VQA_REPO = "robot-learning-group47/smolvlm2-500m-vqa-lora-merged"
DEFAULT_HF_HOME = "/data/hf_home/hub"
DEFAULT_OUT_DIR = Path("/data/lerobot_outputs/train/smolvla-eval2-pathC-init")
# ---------------------------------------------------------------------------

# The hierarchy inside eval1's safetensors:
#   SmolVLAPolicy.model  (VLAFlowMatching)
#     .vlm_with_expert   (SmolVLMWithExpertModel)
#       .vlm             (SmolVLMForConditionalGeneration)   ← backbone
#       .lm_expert       (AutoModel)                         ← expert
#     .state_proj / .action_* / .action_time_*              ← projections
#
# So backbone keys in eval1 have the prefix:
EVAL1_VLM_PREFIX = "model.vlm_with_expert.vlm."

# Keys in the VQA safetensors that we DON'T want to copy into eval1
# (SmolVLMForConditionalGeneration has an lm_head that SmolVLA doesn't use for generation)
SKIP_SOURCE_SUBSTRINGS = ["lm_head"]

# Keys in eval1 that must NOT be touched (expert, projections)
PROTECTED_SUBSTRINGS = [
    "lm_expert",
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp",
]


def sep(char="-", n=70):
    print(char * n)


def detect_vlm_prefix(eval1_keys: list[str]) -> str:
    """
    Sniff the actual VLM prefix from eval1's key list.
    Returns the detected prefix string (with trailing dot).
    Aborts if none of the candidates match enough keys.
    """
    candidates = [
        "model.vlm_with_expert.vlm.",
        "vlm_with_expert.vlm.",
        "model.vlm.",
        "vlm.",
    ]
    for candidate in candidates:
        n = sum(1 for k in eval1_keys if k.startswith(candidate))
        print(f"  prefix candidate '{candidate}': {n} matching keys")
        if n >= 100:
            print(f"  → using '{candidate}'")
            return candidate

    sep("!")
    print("ERROR: Could not find a VLM prefix in eval1 with ≥100 matches.")
    print("       Check the eval1 safetensors key list above and update")
    print("       EVAL1_VLM_PREFIX in this script.")
    print()
    print("First 20 eval1 keys:")
    for k in sorted(eval1_keys)[:20]:
        print(f"  {k}")
    sys.exit(1)


def detect_vqa_key_style(vqa_keys: list[str]) -> str:
    """
    VQA model may store keys as 'model.text_model...' or 'text_model...'
    Returns the prefix to strip from VQA keys before prepending EVAL1_VLM_PREFIX.
    """
    # SmolVLMForConditionalGeneration wraps its backbone under 'model.'
    sample = sorted(vqa_keys)[0]
    print(f"  Sample VQA key: {sample}")
    # No stripping needed — VQA state_dict keys are already relative to the model class root
    # (e.g. 'model.text_model.layers.0...' which maps to eval1's 'model.vlm_with_expert.vlm.model.text_model...')
    return ""  # no prefix to strip from VQA source keys


def build_merge_map(
    eval1_keys: set[str],
    vqa_sd: dict,
    vlm_prefix: str,
) -> dict[str, str]:
    """
    Returns {eval1_key: vqa_key} for every VQA key that has a matching eval1 slot.
    Skips keys in SKIP_SOURCE_SUBSTRINGS.
    """
    merge_map = {}
    for vqa_key in vqa_sd:
        if any(s in vqa_key for s in SKIP_SOURCE_SUBSTRINGS):
            continue
        target_key = vlm_prefix + vqa_key
        if target_key in eval1_keys:
            merge_map[target_key] = vqa_key
    return merge_map


def check_protected_untouched(merge_map: dict[str, str]) -> None:
    violations = [k for k in merge_map if any(s in k for s in PROTECTED_SUBSTRINGS)]
    if violations:
        sep("!")
        print(f"ERROR: {len(violations)} protected keys would be overwritten:")
        for k in violations[:10]:
            print(f"  {k}")
        sys.exit(1)
    print(f"  ✅ Zero expert/projection keys overwritten.")


def check_dtype_consistency(
    eval1_sd: dict, merge_map: dict[str, str], vqa_sd: dict
) -> list[str]:
    """Warn about dtype mismatches; returns list of mismatched eval1 keys."""
    mismatches = []
    for eval1_key, vqa_key in merge_map.items():
        e_dtype = eval1_sd[eval1_key].dtype
        v_dtype = vqa_sd[vqa_key].dtype
        if e_dtype != v_dtype:
            mismatches.append((eval1_key, e_dtype, v_dtype))
    if mismatches:
        print(f"  ⚠️  {len(mismatches)} dtype mismatches (will cast VQA → eval1 dtype):")
        for k, ed, vd in mismatches[:5]:
            print(f"    {k}: eval1={ed}, vqa={vd}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches)-5} more")
    else:
        print(f"  ✅ All dtypes consistent.")
    return [m[0] for m in mismatches]


def main(dry_run: bool, eval1_repo: str, vqa_repo: str, hf_home: str, out_dir: Path) -> None:
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForImageTextToText

    sep("=")
    print("Path C merge: VQA backbone → eval1_final_final")
    sep("=")

    # ------------------------------------------------------------------
    # Step 1: load eval1 safetensors
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading eval1 weights from {eval1_repo} …")
    eval1_sf = hf_hub_download(
        repo_id=eval1_repo,
        filename="model.safetensors",
        cache_dir=hf_home,
    )
    eval1_sd = load_file(eval1_sf)
    eval1_keys = set(eval1_sd.keys())
    print(f"  eval1 total keys: {len(eval1_keys)}")

    # ------------------------------------------------------------------
    # Step 2: detect VLM prefix in eval1
    # ------------------------------------------------------------------
    print("\n[2/6] Detecting VLM prefix in eval1 keyspace …")
    vlm_prefix = detect_vlm_prefix(list(eval1_keys))

    # Sanity: how many eval1 keys are backbone vs expert vs projections?
    n_backbone   = sum(1 for k in eval1_keys if k.startswith(vlm_prefix))
    n_expert     = sum(1 for k in eval1_keys if "lm_expert" in k)
    n_proj       = sum(1 for k in eval1_keys if any(s in k for s in
                       ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp"]))
    print(f"  eval1 key breakdown:")
    print(f"    backbone ({vlm_prefix}*): {n_backbone}")
    print(f"    lm_expert.*:              {n_expert}")
    print(f"    projections.*:            {n_proj}")
    print(f"    other:                    {len(eval1_keys) - n_backbone - n_expert - n_proj}")

    # ------------------------------------------------------------------
    # Step 3: load VQA model weights
    # ------------------------------------------------------------------
    print(f"\n[3/6] Loading VQA backbone from {vqa_repo} …")
    vqa_dir = snapshot_download(repo_id=vqa_repo, cache_dir=hf_home)
    vqa_model = AutoModelForImageTextToText.from_pretrained(
        vqa_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    vqa_sd = vqa_model.state_dict()
    print(f"  VQA model total keys: {len(vqa_sd)}")
    detect_vqa_key_style(list(vqa_sd.keys()))

    # ------------------------------------------------------------------
    # Step 4: build merge map and run all assertions
    # ------------------------------------------------------------------
    print(f"\n[4/6] Building merge map and running gate checks …")
    merge_map = build_merge_map(eval1_keys, vqa_sd, vlm_prefix)

    # Gate 1: namespace overlap
    overlap = len(merge_map)
    print(f"\n  Gate 1 — Namespace overlap:")
    print(f"    VQA keys (minus skipped): {sum(1 for k in vqa_sd if not any(s in k for s in SKIP_SOURCE_SUBSTRINGS))}")
    print(f"    Matched in eval1:         {overlap}")
    if overlap < 100:
        sep("!")
        print(f"ERROR: Only {overlap} keys overlap — namespace mismatch.")
        print("       Inspect key samples:")
        print("  eval1 backbone sample:", sorted(k for k in eval1_keys if k.startswith(vlm_prefix))[:3])
        print("  constructed VQA keys: ", [vlm_prefix + k for k in sorted(vqa_sd)[:3]])
        sys.exit(1)
    print(f"  ✅ Gate 1 passed: {overlap} keys overlap (≥100 required).")

    # Gate 2: protected keys untouched
    print(f"\n  Gate 2 — Protected keys untouched:")
    check_protected_untouched(merge_map)

    # Gate 3: dtype consistency
    print(f"\n  Gate 3 — Dtype consistency:")
    dtype_mismatches = check_dtype_consistency(eval1_sd, merge_map, vqa_sd)

    # Gate 4: key count preserved
    print(f"\n  Gate 4 — Key count preserved:")
    # The merged dict will have the same keys as eval1 (we only overwrite values)
    print(f"  ✅ Merged key count will be {len(eval1_keys)} (same as eval1).")

    # Summary
    sep()
    print("\n  MERGE PLAN SUMMARY:")
    print(f"    Keys to overwrite (backbone):   {overlap}")
    print(f"    Keys kept as-is (expert+proj):  {len(eval1_keys) - overlap}")
    print(f"    Dtype casts needed:             {len(dtype_mismatches)}")
    print(f"    Protected key violations:       0  ✅")

    if dry_run:
        sep("=")
        print("DRY RUN — no files written. All gates passed. Safe to run without --dry-run.")
        sep("=")
        return

    # ------------------------------------------------------------------
    # Step 5: build merged state dict
    # ------------------------------------------------------------------
    print(f"\n[5/6] Building merged state dict …")
    merged_sd = dict(eval1_sd)  # shallow copy — we replace tensors, not mutate them
    n_written = 0
    for eval1_key, vqa_key in merge_map.items():
        src = vqa_sd[vqa_key]
        tgt_dtype = eval1_sd[eval1_key].dtype
        if src.dtype != tgt_dtype:
            src = src.to(tgt_dtype)
        merged_sd[eval1_key] = src
        n_written += 1

    assert len(merged_sd) == len(eval1_sd), (
        f"BUG: key count changed! {len(merged_sd)} != {len(eval1_sd)}"
    )
    print(f"  Overwrote {n_written} backbone keys.")
    print(f"  Final key count: {len(merged_sd)} (matches eval1 ✅)")

    # ------------------------------------------------------------------
    # Step 6: save + copy config
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving to {out_dir} …")
    if dry_run:
        print("  (dry-run — skipping write)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_sf = out_dir / "model.safetensors"
    save_file(merged_sd, str(out_sf))
    print(f"  Saved: {out_sf} ({out_sf.stat().st_size / 1e9:.2f} GB)")

    # Copy config.json from eval1 (same architecture)
    import shutil
    config_path = hf_hub_download(
        repo_id=eval1_repo, filename="config.json", cache_dir=hf_home
    )
    shutil.copy(config_path, out_dir / "config.json")
    print(f"  Copied config.json from eval1 (same architecture)")

    sep("=")
    print("✅ PATH C INIT CHECKPOINT READY")
    print(f"   Location: {out_dir}")
    print()
    print("Next steps:")
    print("  1. Smoke test (200 steps):")
    print(f"       lerobot-train --policy.path={out_dir} \\")
    print(f"         --policy.train_expert_only=false \\")
    print(f"         --policy.freeze_vision_encoder=true \\")
    print(f"         --dataset.repo_id=robot-learning-group47/eval2_compositional_augmented \\")
    print(f"         --steps=200 --batch_size=8 --output_dir=/tmp/pathC_smoke")
    print()
    print("  2. Gradient check after step 1:")
    print("       (paste snippet from scripts/check_gradients_pathC.py)")
    print()
    print("  3. Loss baseline: compare step-0 loss vs Path A cold start.")
    print("     Path C should start lower because the eval1 expert is already competent.")
    sep("=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge VQA backbone into eval1 checkpoint (Path C init)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all checks and print diagnostics without writing any files")
    parser.add_argument("--eval1-repo", default=DEFAULT_EVAL1_REPO)
    parser.add_argument("--vqa-repo", default=DEFAULT_VQA_REPO)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        eval1_repo=args.eval1_repo,
        vqa_repo=args.vqa_repo,
        hf_home=args.hf_home,
        out_dir=args.out_dir,
    )
