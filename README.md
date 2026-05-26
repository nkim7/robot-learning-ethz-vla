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
A harder generalization test: three bowls appear in varying color-and-order permutations (18 combinations in total), and the prompt uses positional (*"1st from the left"*), relational (*"to the right of the red bowl"*), or negation (*"not red and not blue"*) language. We collected 108 demonstrations across all combinations and trained SmolVLA 0.5B.

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
├── run_eval_1.sh                 — Eval 1 one-command runner
├── run_eval_2.sh                 — Eval 2 one-command runner
├── run_eval_3_TOY.sh             — Eval 3 TOY figures runner
├── run_eval_3_celeb.sh           — Eval 3 celebrity cards runner
│
├── robot/                        — Hardware scripts, calibration & config
│   ├── calibration/              — Arm calibration files for group 47
│   ├── config/                   — Robot config and initial pose YAML
│   └── patches/                  — OpenCV camera stability patch
│
├── models/                       — Checkpoints (gitignored, auto-downloaded)
│   ├── eval1/                    — SmolVLA 0.5B checkpoint
│   ├── eval2/                    — SmolVLA 0.5B checkpoint
│   └── eval3/runtime/            — Eval 3 inference code (git-tracked)
│
├── training_pipeline/
│   ├── eval1/
│   │   ├── data/                 — Training datasets (gitignored)
│   │   ├── shells/               — Recording, training & evaluation shell scripts
│   │   └── src/                  — Training, evaluation & dataset scripts
│   ├── eval2/
│   │   ├── data/                 — Training datasets & image assets (gitignored)
│   │   ├── shells/               — Recording, training & evaluation shell scripts
│   │   └── src/                  — Training, probing & dataset construction scripts
│   ├── eval3/
│   │   ├── data/                 — Training datasets & image assets (gitignored)
│   │   ├── shells/               — Recording & training shell scripts
│   │   └── src/
│   │       ├── vla/              — VLA action model training scripts
│   │       └── vlm/              — VLM scene understanding training scripts
│   └── logs/                     — Experiment logs & probe results
│
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
│
└── lerobot/                      — LeRobot submodule (v0.5.2)
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
