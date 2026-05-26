# Robot Learning — Group 47

SO-101 robot arm with SmolVLA policies. Three evaluation tasks.

---

## Setup

**Requirements:** macOS or Linux, Python 3.12+

```bash
# 1. Clone
git clone --recursive https://github.com/lorefrigo/robot-learning-ethz-vla.git
cd robot-learning-ethz-vla

# 2. Install dependencies
brew install ffmpeg uv          # macOS
# sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux

# 3. Find and save your robot port
source .venv/bin/activate 2>/dev/null || { uv sync && source .venv/bin/activate; }
lerobot-find-port
echo "export FOLLOWER_PORT=/dev/tty.usbmodemXXXX" >> .env   # replace with your port
```

Everything else (Python env, camera patch, model download) is handled automatically on first run.

---

## Running Evals

### Eval 1 — *"Put the banana in the blue colored bowl."*
```bash
bash run_eval_1.sh
```

### Eval 2 — *"Put the banana into the bowl to the right of the blue bowl."*
```bash
bash run_eval_2.sh
```

### Eval 3 — TOY figures (tasks 1 & 2)
```bash
TASK="Place the coke on Taylor Swift." bash run_eval_3_TOY.sh
# Other supported figures: Obama, Yann LeCun, …
```

### Eval 3 — Celebrity cards (task 3)
```bash
TASK="Place the coke on Rihanna." bash run_eval_3_celeb.sh
# Other supported celebrities: Elon Musk, …
```

---

## Teleoperation

```bash
[[ -f .env ]] && source .env
source .venv/bin/activate
python robot/teleop.py
```

Requires both `FOLLOWER_PORT` and `LEADER_PORT` set in `.env`.

---

## How Eval 3 Works

Eval 3 uses a two-stage pipeline:

1. **VLM LoRA** — on the first camera frame, a fine-tuned SmolVLM2-2.2B identifies which slot (left / middle / right) the target person is in.
2. **Instruction rewrite** — the task `"Place the coke on Taylor Swift."` is rewritten to `"Put the coke on left."` so the action expert receives the position-conditioned instruction it was trained on.
3. **Action expert** — SmolVLA takes the rewritten instruction + camera image and outputs joint commands every frame.

Two LoRA adapters cover the two visual domains:
- `models/eval3/TOY/` — plastic figurines
- `models/eval3/celeb/` — printed celebrity cards

---

## Models

Model weights are **not stored in git** — they live on HuggingFace and are downloaded automatically on first `bash run_eval_*.sh`. To download manually upfront:

```bash
bash scripts/download_models.sh
```

This fetches from `robot-learning-group47/` on HuggingFace into:

```
models/
├── eval1/          ← weights for run_eval_1.sh
├── eval2/          ← weights for run_eval_2.sh
└── eval3/
    ├── TOY/        ← VLM LoRA for run_eval_3_TOY.sh
    ├── celeb/      ← VLM LoRA for run_eval_3_celeb.sh
    └── runtime/    ← inference code (git-tracked, no download needed)
```

---

## Datasets

To download the training datasets used for each eval:

```bash
bash scripts/download_datasets.sh          # all three
bash scripts/download_datasets.sh eval1    # only eval1
bash scripts/download_datasets.sh eval2
bash scripts/download_datasets.sh eval3
```

Downloads into `training_pipeline/eval*/data/`.

---

## Hardware Notes

- Follower arm charger: **green** light — Leader arm charger: **blue** light
- Connect the 12V power adapter to the follower arm before running
- Calibration files: `robot/calibration/so101_group47/`

---

## Repository Structure

```
robot-learning-ethz-vla/
├── run_eval_1.sh              — Eval 1 one-shot runner
├── run_eval_2.sh              — Eval 2 one-shot runner
├── run_eval_3_TOY.sh          — Eval 3 TOY figures runner
├── run_eval_3_celeb.sh        — Eval 3 celebrity cards runner
│
├── robot/                     — Hardware scripts & calibration
│   ├── calibration/           — Calibration files for group 47 arms
│   ├── patches/               — LeRobot camera stability patch
│   ├── teleop.py              — Teleoperation
│   ├── calibrate.py           — Calibration helper
│   ├── go_eval1_both_initial.py
│   └── reset_from_eval1_pose.py
│
├── models/                    — Policy checkpoints (gitignored, auto-downloaded)
│   ├── eval1/
│   ├── eval2/
│   └── eval3/
│       ├── TOY/               — VLM LoRA for TOY figures
│       ├── celeb/             — VLM LoRA for celebrity cards
│       └── runtime/           — Eval 3 inference runner (git-tracked)
│
├── training_pipeline/
│   ├── eval1/
│   │   ├── data/              — Datasets and config files
│   │   ├── src/               — Python training & analysis scripts
│   │   └── shells/            — Shell scripts (train, record, eval)
│   ├── eval2/
│   │   ├── data/              — Datasets, previews, VQA assets
│   │   ├── src/               — Python training & analysis scripts
│   │   └── shells/            — Shell scripts (train, record, eval)
│   └── eval3/
│       ├── data/              — Datasets and generated assets
│       ├── src/
│       │   ├── vlm/           — VLM LoRA training scripts
│       │   └── vla/           — VLA action model training scripts & server code
│       └── shells/            — Shell scripts (train, record, eval)
│
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
└── lerobot/                   — LeRobot submodule (v0.5.2)
```

---

## Calibration (new robot only)

```bash
source .venv/bin/activate
python robot/calibrate.py \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=my_follower \
  --robot.calibration_dir=robot/calibration/so101_group47/follower
```
