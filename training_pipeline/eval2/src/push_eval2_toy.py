"""
Push Phase 1 and Phase 2 checkpoints to HuggingFace.
  Phase 1 -> nkim7/eval2_phase1_TOY
  Phase 2 -> nkim7/eval2_phase2_TOY
"""
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
user = "nkim7"

# ── Phase 1 ──────────────────────────────────────────────────────────────────
p1_repo = f"{user}/eval2_phase1_TOY"
p1_dir  = Path("./checkpoints/phase1_faces/final")

print(f"Creating {p1_repo} ...")
api.create_repo(p1_repo, repo_type="model", exist_ok=True, private=False)
print(f"Uploading {p1_dir} ...")
api.upload_folder(
    folder_path=str(p1_dir),
    repo_id=p1_repo,
    repo_type="model",
    commit_message="Phase 1 face-recognition LoRA (Taylor Swift / Barack Obama / Yann LeCun) — 100% benchmark",
)
print(f"Done: https://huggingface.co/{p1_repo}")

# ── Phase 2 ──────────────────────────────────────────────────────────────────
p2_repo = f"{user}/eval2_phase2_TOY"
p2_dir  = Path("./checkpoints/phase2_scenes/epoch_6")

print(f"\nCreating {p2_repo} ...")
api.create_repo(p2_repo, repo_type="model", exist_ok=True, private=False)
print(f"Uploading {p2_dir} ...")
api.upload_folder(
    folder_path=str(p2_dir),
    repo_id=p2_repo,
    repo_type="model",
    commit_message="Phase 2 scene-position LoRA (LEFT/CENTER/RIGHT card recognition) — 99.44% benchmark (179/180)",
)
print(f"Done: https://huggingface.co/{p2_repo}")
