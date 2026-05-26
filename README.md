# Robot Learning — ETH Zürich, Group 47

**Vision-Language-Action policy experiments on a SO-101 robot arm.**  
Course project for *Robot Learning* (ETH Zürich, Spring 2026).

We fine-tuned [SmolVLA 0.5B](https://huggingface.co/HuggingFaceTB/SmolVLA-500M) — a compact Vision-Language-Action model built on SmolVLM2 — on three increasingly challenging manipulation tasks using a SO-101 6-DOF follower arm.

---

## Tasks

### Eval 1 — Color-conditioned bowl placement
The robot places a banana into a specified colored bowl (red, green, or blue).  
Language prompt: *"Put the banana in the blue colored bowl."*  
Policy fine-tuned on ~200 teleoperated demonstrations across three fixed bowl positions.

### Eval 2 — Relational and ordinal language grounding
The robot interprets spatial and ordinal prompts referring to one of three bowls with varying layout and color permutations.  
Language prompts include positional (*"1st from the left"*), relational (*"to the right of the red bowl"*), and negation forms (*"not red and not blue"*).  
Policy fine-tuned on ~1080 demonstrations across 18 bowl-order × color combinations.

### Eval 3 — Open-vocabulary object placement
The robot places a coke can on a target image identified by a natural-language celebrity name.  
Language prompt: *"Place the coke on Taylor Swift."*  
Built on a two-stage pipeline: a VLM LoRA (SmolVLM2-2.2B) resolves the target card position, which is then passed to the SmolVLA 0.5B action policy.

---

## Setup

**Requirements:** macOS or Linux · Python 3.12+ · `ffmpeg` · `uv`

```bash
# 1. Clone (with submodule)
git clone --recursive https://github.com/lorefrigo/robot-learning-ethz-vla.git
cd robot-learning-ethz-vla

# 2. Install system dependencies
brew install ffmpeg uv                  # macOS
# sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux

# 3. Create virtual environment
uv sync && source .venv/bin/activate

# 4. Find your robot ports
lerobot-find-port

# 5. Export port variables (add to ~/.zshrc or ~/.bashrc to persist)
export FOLLOWER_PORT=/dev/tty.usbmodemXXXX
export LEADER_PORT=/dev/tty.usbmodemYYYY
```

Everything else (model download, camera patch) is handled automatically on first run.

---

## Running Evals

Set your port variables, then:

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

Weights are **not stored in git** — they are downloaded automatically on first `bash run_eval_*.sh` from `robot-learning-group47/` on HuggingFace. To pre-download:

```bash
bash scripts/download_models.sh
```

All three evals use **SmolVLA 0.5B** (`HuggingFaceTB/SmolVLA-500M`) as the base action policy, fine-tuned from the same checkpoint.

```
models/
├── eval1/          ← SmolVLA checkpoint for Eval 1
├── eval2/          ← SmolVLA checkpoint for Eval 2
└── eval3/
    ├── TOY/        ← VLM LoRA (SmolVLM2-2.2B) for TOY figures
    ├── celeb/      ← VLM LoRA (SmolVLM2-2.2B) for celebrity cards
    └── runtime/    ← Eval 3 inference code (git-tracked)
```

---

## Datasets

Training datasets are hosted on HuggingFace under `robot-learning-group47/`.

```bash
bash scripts/download_datasets.sh          # all three evals
bash scripts/download_datasets.sh eval1    # only eval 1
bash scripts/download_datasets.sh eval2
bash scripts/download_datasets.sh eval3
```

Downloads land in `training_pipeline/eval*/data/`.

---

## Teleoperation

```bash
source .venv/bin/activate
python robot/teleop.py
# Requires FOLLOWER_PORT and LEADER_PORT to be set
```

---

## Repository Structure

```
robot-learning-ethz-vla/
├── run_eval_1.sh              — Eval 1 runner
├── run_eval_2.sh              — Eval 2 runner
├── run_eval_3_TOY.sh          — Eval 3 TOY figures runner
├── run_eval_3_celeb.sh        — Eval 3 celebrity cards runner
│
├── robot/                     — Hardware scripts & calibration
│   ├── calibration/           — Calibration files for group 47 arms
│   ├── config/poses/          — Initial pose YAML
│   ├── teleop.py              — Teleoperation
│   ├── go_eval1_both_initial.py
│   └── reset_from_eval1_pose.py
│
├── models/                    — Policy checkpoints (gitignored, auto-downloaded)
│
├── training_pipeline/
│   ├── eval1/
│   │   ├── data/              — Datasets and config
│   │   ├── src/               — Training & evaluation scripts
│   │   └── shells/            — Shell scripts (train, record, eval)
│   ├── eval2/
│   │   ├── data/
│   │   ├── src/
│   │   └── shells/
│   └── eval3/
│       ├── data/
│       ├── src/
│       │   ├── vlm/           — VLM LoRA training (SmolVLM2-2.2B)
│       │   └── vla/           — VLA action model training & server
│       └── shells/
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
