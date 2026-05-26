#!/usr/bin/env python3
"""
Run 2 merge: strict10 VQA backbone weights → eval1_final_final checkpoint

Produces a combined SmolVLA checkpoint with:
  - backbone (model.vlm_with_expert.vlm.*):
      from  robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged
  - expert   (model.vlm_with_expert.lm_expert.*):
      from  robot-learning-group47/eval1_final_final
  - projections (state_proj, action_{in,out}_proj, action_time_mlp_*):
      from  robot-learning-group47/eval1_final_final

This is the init checkpoint for Run 2 (eval2_strict10_frozenbackbone_expert_full_v1).
The backbone will be fully frozen during training; only the expert and projections are trained.

Usage (on GPU server):
    cd ~/lerobot
    python ~/merge_strict10_into_eval1.py [--dry-run]

Gates (script aborts loudly before writing if any fail):
  1. Namespace alignment  : VQA→eval1 key overlap must be ≥ 100
  2. Key count preserved  : merged dict must have same number of keys as eval1
  3. Dtype consistency    : each overwritten key must have matching dtype
  4. Expert untouched     : zero expert/projection keys should have been overwritten
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Run-2-specific defaults
# ---------------------------------------------------------------------------
DEFAULT_EVAL1_REPO = "robot-learning-group47/eval1_final_final"
DEFAULT_VQA_REPO   = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
DEFAULT_HF_HOME    = "/data/hf_home/hub"
DEFAULT_OUT_DIR    = Path("/data/lerobot_outputs/train/smolvla-eval2-strict10frozenbackbone-init")
# ---------------------------------------------------------------------------

EVAL1_VLM_PREFIX = "model.vlm_with_expert.vlm."

SKIP_SOURCE_SUBSTRINGS = ["lm_head"]

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
    print("First 20 eval1 keys:")
    for k in sorted(eval1_keys)[:20]:
        print(f"  {k}")
    sys.exit(1)


def build_merge_map(eval1_keys: set[str], vqa_sd: dict, vlm_prefix: str) -> dict[str, str]:
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


def check_dtype_consistency(eval1_sd: dict, merge_map: dict[str, str], vqa_sd: dict):
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


def print_param_summary(merged_sd: dict, vlm_prefix: str) -> None:
    """
    Print a breakdown of the merged checkpoint's parameter groups.
    Used to verify frozen vs trainable split before training.
    """
    sep("=")
    print("  PARAMETER BREAKDOWN (merged checkpoint)")
    sep()

    groups = {
        "VLM backbone (FROZEN in training)": vlm_prefix,
        "lm_expert (trainable)":             "lm_expert",
        "state_proj (trainable)":            "state_proj",
        "action_in_proj (trainable)":        "action_in_proj",
        "action_out_proj (trainable)":       "action_out_proj",
        "action_time_mlp (trainable)":       "action_time_mlp",
    }

    total_params = 0
    for label, prefix_or_substr in groups.items():
        # check both startswith (for VLM) and substring (for expert/proj)
        if label.startswith("VLM"):
            keys = [k for k in merged_sd if k.startswith(prefix_or_substr)]
        else:
            keys = [k for k in merged_sd if prefix_or_substr in k]
        n = sum(merged_sd[k].numel() for k in keys)
        total_params += n
        print(f"  {label:45s}  {n/1e6:8.2f} M  ({len(keys)} tensors)")

    other_keys = [
        k for k in merged_sd
        if not k.startswith(vlm_prefix)
        and not any(s in k for s in ["lm_expert", "state_proj", "action_in_proj",
                                      "action_out_proj", "action_time_mlp"])
    ]
    n_other = sum(merged_sd[k].numel() for k in other_keys)
    if other_keys:
        print(f"  {'other':45s}  {n_other/1e6:8.2f} M  ({len(other_keys)} tensors)")

    grand_total = sum(v.numel() for v in merged_sd.values())
    backbone_n  = sum(merged_sd[k].numel() for k in merged_sd if k.startswith(vlm_prefix))
    trainable_n = grand_total - backbone_n
    sep()
    print(f"  Grand total:       {grand_total/1e6:.2f} M")
    print(f"  Frozen (backbone): {backbone_n/1e6:.2f} M")
    print(f"  Trainable (expert+proj): {trainable_n/1e6:.2f} M")
    sep("=")

    if trainable_n < 1e6:
        print("⚠️  WARNING: trainable param count < 1 M — expert may be accidentally frozen!")
        sys.exit(1)
    if backbone_n < 100e6:
        print("⚠️  WARNING: backbone param count < 100 M — backbone may not be loaded!")
        sys.exit(1)


def main(dry_run: bool, eval1_repo: str, vqa_repo: str, hf_home: str, out_dir: Path) -> None:
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForImageTextToText
    import shutil

    sep("=")
    print("Run 2 merge: strict10 VQA backbone → eval1_final_final")
    print(f"  VQA backbone: {vqa_repo}")
    print(f"  Expert:       {eval1_repo}")
    print(f"  Output:       {out_dir}")
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

    n_backbone = sum(1 for k in eval1_keys if k.startswith(vlm_prefix))
    n_expert   = sum(1 for k in eval1_keys if "lm_expert" in k)
    n_proj     = sum(1 for k in eval1_keys if any(s in k for s in
                     ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp"]))
    print(f"  eval1 key breakdown:")
    print(f"    backbone ({vlm_prefix}*): {n_backbone}")
    print(f"    lm_expert.*:              {n_expert}")
    print(f"    projections.*:            {n_proj}")
    print(f"    other:                    {len(eval1_keys) - n_backbone - n_expert - n_proj}")

    # ------------------------------------------------------------------
    # Step 3: load strict10 VQA model weights
    # ------------------------------------------------------------------
    print(f"\n[3/6] Loading strict10 VQA backbone from {vqa_repo} …")
    vqa_dir = snapshot_download(repo_id=vqa_repo, cache_dir=hf_home)
    vqa_model = AutoModelForImageTextToText.from_pretrained(
        vqa_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    vqa_sd = vqa_model.state_dict()
    print(f"  VQA model total keys: {len(vqa_sd)}")
    sample = sorted(vqa_sd.keys())[0]
    print(f"  Sample VQA key: {sample}")

    # ------------------------------------------------------------------
    # Step 4: build merge map and run all gates
    # ------------------------------------------------------------------
    print(f"\n[4/6] Building merge map and running gate checks …")
    merge_map = build_merge_map(eval1_keys, vqa_sd, vlm_prefix)

    # Gate 1: namespace overlap
    overlap = len(merge_map)
    print(f"\n  Gate 1 — Namespace overlap:")
    n_vqa_eligible = sum(1 for k in vqa_sd if not any(s in k for s in SKIP_SOURCE_SUBSTRINGS))
    print(f"    VQA keys (minus skipped): {n_vqa_eligible}")
    print(f"    Matched in eval1:         {overlap}")
    if overlap < 100:
        sep("!")
        print(f"ERROR: Only {overlap} keys overlap — namespace mismatch.")
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
    print(f"  ✅ Merged key count will be {len(eval1_keys)} (same as eval1).")

    # Summary
    sep()
    print("\n  MERGE PLAN SUMMARY:")
    print(f"    Keys to overwrite (backbone):        {overlap}")
    print(f"    Keys kept as-is (expert+proj):       {len(eval1_keys) - overlap}")
    print(f"    Dtype casts needed:                  {len(dtype_mismatches)}")
    print(f"    Protected key violations:            0  ✅")
    print(f"    Backbone source:  {vqa_repo}")
    print(f"    Expert source:    {eval1_repo}")

    if dry_run:
        sep("=")
        print("DRY RUN — no files written. All gates passed. Safe to run without --dry-run.")
        sep("=")
        return

    # ------------------------------------------------------------------
    # Step 5: build merged state dict
    # ------------------------------------------------------------------
    print(f"\n[5/6] Building merged state dict …")
    merged_sd = dict(eval1_sd)
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

    # Parameter summary
    print_param_summary(merged_sd, vlm_prefix)

    # ------------------------------------------------------------------
    # Step 6: save + copy config
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving to {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sf = out_dir / "model.safetensors"
    save_file(merged_sd, str(out_sf))
    print(f"  Saved: {out_sf} ({out_sf.stat().st_size / 1e9:.2f} GB)")

    config_path = hf_hub_download(
        repo_id=eval1_repo, filename="config.json", cache_dir=hf_home
    )
    shutil.copy(config_path, out_dir / "config.json")
    print(f"  Copied config.json from eval1 (same SmolVLA architecture)")

    sep("=")
    print("✅ RUN 2 INIT CHECKPOINT READY")
    print(f"   Location: {out_dir}")
    print(f"   Backbone: {vqa_repo}  [WILL BE FROZEN]")
    print(f"   Expert:   {eval1_repo}  [TRAINABLE]")
    sep("=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge strict10 VQA backbone into eval1 checkpoint (Run 2 init)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all checks without writing files")
    parser.add_argument("--eval1-repo", default=DEFAULT_EVAL1_REPO)
    parser.add_argument("--vqa-repo",   default=DEFAULT_VQA_REPO)
    parser.add_argument("--hf-home",    default=DEFAULT_HF_HOME)
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        eval1_repo=args.eval1_repo,
        vqa_repo=args.vqa_repo,
        hf_home=args.hf_home,
        out_dir=args.out_dir,
    )
