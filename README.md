# Robot Learning — ETH Zürich · Group 47

> Language-conditioned robot manipulation using a VLA model on a SO-101 robot arm.  
> *Robot Learning course — ETH Zürich, Spring 2026.*

---

## Project Overview

This project explores language-conditioned manipulation with a Vision-Language-Action model on a real robot. We used [SmolVLA 0.5B](https://huggingface.co/HuggingFaceTB/SmolVLA-500M) (`HuggingFaceTB/SmolVLA-500M`) — built on SmolVLM2 — as our VLA backbone, training it on three progressively harder tasks on a SO-101 6-DOF follower arm.

| | Task | Language |
|---|---|---|
| **Eval 1** | Color bowl | *"Put the banana in the blue bowl."* |
| **Eval 2** | Relational bowl | *"Put the banana into the bowl to the right of the red bowl."* |
| **Eval 3** | Open-vocabulary placement | *"Place the coke on Taylor Swift."* |

---

## Tasks

### Eval 1 — Color-conditioned placement
The robot picks up a banana and places it into a target bowl identified by color (red, green, or blue). Three bowls are always present; the language prompt specifies which one. We collected 60 teleoperated demonstrations across fixed bowl positions and trained SmolVLA 0.5B on this data.

### Eval 2 — Relational and ordinal language grounding
A harder generalization test: three bowls appear in varying color-and-order permutations (18 combinations in total), and the prompt uses positional (*"1st from the left"*), relational (*"to the right of the red bowl"*), or negation (*"not red and not blue"*) language. We collected 108 demonstrations across all combinations and trained SmolVLA 0.5B to ground these richer descriptions into motor actions.

### Eval 3 — Open-vocabulary object placement
The robot must pick up a coke can and place it precisely on top of a small celebrity photo card lying flat on the table — identified only by the person's name in the prompt. Solved with SmolVLA 0.5B.

---

## Setup

**Requirements:** macOS or Linux · Python 3.12+ · `ffmpeg` · `uv`

```bash
# Clone with submodule
git clone --recursive https://github.com/lorefrigo/robot-learning-ethz-vla.git
cd robot-learning-ethz-vla

# Install system dependencies
brew install ffmpeg uv                  # macOS
# sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux

# Create virtual environment
uv sync && source .venv/bin/activate

# Find your robot ports, then export them
lerobot-find-port
export FOLLOWER_PORT=/dev/tty.usbmodemXXXX
export LEADER_PORT=/dev/tty.usbmodemYYYY
```

Models and the camera patch are downloaded automatically on first run. Add the `export` lines to `~/.zshrc` or `~/.bashrc` to avoid re-typing each session.

---

## Running

### Eval 1
```bash
bash run_eval_1.sh
```

### Eval 2
```bash
bash run_eval_2.sh
```

### Eval 3 — TOY figures
```bash
TASK="Place the coke on Taylor Swift." bash run_eval_3_TOY.sh
```

### Eval 3 — Celebrity cards
```bash
TASK="Place the coke on Rihanna." bash run_eval_3_celeb.sh
```

---

## Models

All three evals use **SmolVLA 0.5B** ([`HuggingFaceTB/SmolVLA-500M`](https://huggingface.co/HuggingFaceTB/SmolVLA-500M)) as the VLA backbone.

Weights are **not stored in git** — they live on HuggingFace under `robot-learning-group47/` and are downloaded automatically on first run. To pre-fetch:

```bash
bash scripts/download_models.sh
```

```
models/
├── eval1/       ← SmolVLA 0.5B checkpoint
├── eval2/       ← SmolVLA 0.5B checkpoint
└── eval3/
    └── runtime/ ← inference code (git-tracked)
```

---

## Datasets

Hosted on HuggingFace under `robot-learning-group47/`. Download with:

```bash
bash scripts/download_datasets.sh          # all three
bash scripts/download_datasets.sh eval1    # single eval
```

Downloaded data lands in `training_pipeline/eval*/data/`.

---

## Teleoperation

```bash
source .venv/bin/activate
python robot/teleop.py   # requires FOLLOWER_PORT and LEADER_PORT
```

---

## Repository Structure

```
robot-learning-ethz-vla/
│
├── run_eval_1.sh                          — Eval 1 one-command runner
├── run_eval_2.sh                          — Eval 2 one-command runner
├── run_eval_3_TOY.sh                      — Eval 3 TOY figures runner
├── run_eval_3_celeb.sh                    — Eval 3 celebrity cards runner
│
├── robot/
│   ├── calibration/so101_group47/
│   │   ├── follower/                      — Follower arm calibration JSON
│   │   └── leader/                        — Leader arm calibration JSON
│   ├── config/
│   │   ├── poses/eval1_follower_initial_pose.yaml  — Reset pose for all evals
│   │   └── robot/                         — Robot config JSONs
│   ├── patches/lerobot_opencv_camera_no_crash.patch
│   ├── teleop.py                          — Teleoperation script
│   ├── calibrate.py                       — Calibration helper
│   ├── go_eval1_both_initial.py           — Move both arms to initial pose
│   ├── reset_from_eval1_pose.py           — Reset follower to initial pose
│   ├── replay_all.sh                      — Replay recorded episodes on robot
│   ├── apply_lerobot_camera_patch.sh      — Apply OpenCV camera stability patch
│   └── setup_teleop_keyword.sh            — Keyboard shortcut setup for teleop
│
├── models/                                — Checkpoints (gitignored, auto-downloaded)
│   ├── eval1/                             — SmolVLA 0.5B checkpoint
│   ├── eval2/                             — SmolVLA 0.5B checkpoint
│   └── eval3/runtime/                     — Eval 3 inference code (git-tracked)
│
├── training_pipeline/
│   │
│   ├── eval1/
│   │   ├── shells/
│   │   │   ├── record_eval1_interactive_resume.sh  — Interactive recording loop
│   │   │   ├── run_train_eval1_fixed_bowl.sh        — Main training run
│   │   │   ├── run_train_eval1_position.sh          — Position-only training
│   │   │   └── eval_stage1.sh                       — Evaluation script
│   │   └── src/
│   │       ├── eval_stage1_quick.py                 — On-robot evaluation (3 prompts)
│   │       ├── run_eval1_checkpoint.py              — Run specific HF checkpoint
│   │       ├── run_eval1_inference.py               — Low-level inference runner
│   │       ├── delete_latest_eval1_episode.py       — Dataset episode management
│   │       ├── relabel_eval1_position_dataset.py
│   │       ├── inspect_eval1_grasp_quality.py
│   │       ├── merge_vqa_backbone_into_eval1.py
│   │       └── build_eval1final_groundedvlm_hybrid.py
│   │
│   ├── eval2/
│   │   ├── shells/
│   │   │   ├── eval2_record_combinations.sh         — Record all 18 bowl combos
│   │   │   ├── run_eval2_pathC_20prompts.sh          — 20-prompt sweep on robot
│   │   │   ├── run_eval2_resolvedcolor.sh            — Color-resolved eval
│   │   │   ├── run_stage1_position_bootstrap.sh
│   │   │   ├── run_stage2_compositional.sh
│   │   │   └── run_train_*.sh                        — Various training experiments
│   │   └── src/
│   │       ├── eval_stage2_prompt_loop.py            — Interactive on-robot eval loop
│   │       ├── run_eval2_pathC_20prompts.py          — 20-prompt sweep script
│   │       ├── run_eval2_resolver_fullrollout.py
│   │       ├── build_eval2_*.py                      — Dataset construction scripts
│   │       ├── create_eval2_*.py                     — Prompt augmentation & VQA datasets
│   │       ├── train_eval2_*.py                      — Training experiment scripts
│   │       └── probe_*.py                            — Diagnostic & probing scripts
│   │
│   ├── eval3/
│   │   ├── shells/
│   │   │   ├── eval3_record.sh                      — Interactive recording loop
│   │   │   ├── run_eval3_cotrain_cycle06_action.sh  — Main training run
│   │   │   ├── run_phase2_pipeline.sh
│   │   │   └── init.sh
│   │   └── src/
│   │       ├── vla/                                  — VLA action model training
│   │       │   ├── scripts/eval3_run1_action_train/  — Run 1 training scripts
│   │       │   ├── scripts/eval3_run2_lmr210_action_train/
│   │       │   ├── scripts/eval3_cotrain/            — Co-training experiments
│   │       │   ├── scripts/eval3_vlm_lora/           — VLM LoRA experiments
│   │       │   ├── scripts/eval3_bridge_train/
│   │       │   ├── scripts/eval3_alignment/
│   │       │   └── scripts/eval3_runtime_common.sh
│   │       └── vlm/                                  — VLM scene understanding scripts
│   │           ├── train_eval3_phase1_lora.py
│   │           ├── train_eval3_phase2_*.py
│   │           ├── eval3_*.py                        — Evaluation & probing
│   │           └── build_*/gen_*/synthesize_*.py     — Dataset generation
│   │
│   └── logs/                                         — Experiment logs & probe results
│
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
│
└── lerobot/                                          — LeRobot submodule (v0.5.2)
```

---

## Hardware Notes

- Calibration files: `robot/calibration/so101_group47/`
- Ports are auto-detected from `/dev/tty.usbmodem*` if env vars are not set

---

## Calibration (new robot only)

```bash
source .venv/bin/activate
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=my_follower \
  --robot.calibration_dir=robot/calibration/so101_group47/follower
```
