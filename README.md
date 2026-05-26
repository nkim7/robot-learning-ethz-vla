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
The robot picks up a banana and places it into a target bowl identified by color (red, green, or blue). Three bowls are always present; the language prompt specifies which one. We collected around 200 teleoperated demonstrations across fixed bowl positions and fine-tuned SmolVLA 0.5B from scratch on this data.

### Eval 2 — Relational and ordinal language grounding
A harder generalization test: three bowls appear in varying color-and-order permutations (18 combinations in total), and the prompt uses positional (*"1st from the left"*), relational (*"to the right of the red bowl"*), or negation (*"not red and not blue"*) language. We collected around 1 080 demonstrations across all combinations and fine-tuned SmolVLA 0.5B to ground these richer descriptions into motor actions.

### Eval 3 — Open-vocabulary object placement
The robot places a coke can on a celebrity face card identified only by name in the prompt. This task is open-vocabulary — the set of valid targets is not fixed at training time. We solve it with a two-stage pipeline: a **VLM LoRA** (SmolVLM2-2.2B) reads the scene and resolves the target card position; that position is then passed as context to **SmolVLA 0.5B** which executes the placement.

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

All three evals use **SmolVLA 0.5B** ([`HuggingFaceTB/SmolVLA-500M`](https://huggingface.co/HuggingFaceTB/SmolVLA-500M)) as the VLA backbone. Eval 3 additionally uses a VLM LoRA on SmolVLM2-2.2B for scene understanding.

Weights are **not stored in git** — they live on HuggingFace under `robot-learning-group47/` and are downloaded automatically on first run. To pre-fetch:

```bash
bash scripts/download_models.sh
```

```
models/
├── eval1/       ← SmolVLA 0.5B fine-tune
├── eval2/       ← SmolVLA 0.5B fine-tune
└── eval3/
    ├── TOY/     ← SmolVLM2-2.2B VLM LoRA (TOY figures)
    ├── celeb/   ← SmolVLM2-2.2B VLM LoRA (celebrity cards)
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
├── run_eval_*.sh              — One-command eval runners
│
├── robot/                     — Hardware layer
│   ├── calibration/           — Calibration files for group 47 arms
│   ├── config/poses/          — Initial pose YAML
│   ├── teleop.py
│   ├── go_eval1_both_initial.py
│   └── reset_from_eval1_pose.py
│
├── models/                    — Checkpoints (gitignored, auto-downloaded)
│
├── training_pipeline/
│   ├── eval1/  ─┐
│   ├── eval2/   ├─  data/ · src/ · shells/
│   └── eval3/  ─┘  (src/ splits into vlm/ and vla/ for eval3)
│
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
│
└── lerobot/                   — LeRobot submodule (v0.5.2)
```

---

## Hardware Notes

- **Follower arm charger:** green LED — **Leader arm charger:** blue LED
- Connect the 12V power adapter to the follower before running
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
