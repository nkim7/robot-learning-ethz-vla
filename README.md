# Robot Learning — Group 47

SO-101 robot arm with SmolVLA policies. Three evaluation tasks.

---

## Repository Structure

```
robot-learning-ethz-vla/
├── run_eval_1.sh              — Eval 1: fixed-bowl grasping
├── run_eval_2.sh              — Eval 2: relational reasoning
├── run_eval_3_TOY.sh          — Eval 3 tasks 1/2: TOY figures (Taylor Swift, Obama, LeCun …)
├── run_eval_3_celeb.sh        — Eval 3 task 3: celebrities (Rihanna, Elon Musk …)
│
├── robot/                     — Hardware scripts and configuration
│   ├── calibration/so101_group47/   — Arm calibration files (follower + leader)
│   ├── patches/               — LeRobot camera stability patch
│   ├── apply_lerobot_camera_patch.sh
│   ├── teleop.py              — Teleoperation
│   └── calibrate.py           — Calibration script
│
├── models/                    — Policy checkpoints (gitignored, auto-downloaded on first run)
│   ├── eval1/                 — Eval 1 final policy
│   ├── eval2/                 — Eval 2 final policy
│   └── eval3/
│       ├── model.safetensors + config   — Eval 3 action model
│       ├── TOY/               — VLM LoRA for TOY figures
│       ├── celeb/             — VLM LoRA for celebrities
│       └── runtime/           — Inference code (git-tracked)
│           ├── eval_3_runner.py
│           ├── run1_lora_patch.py
│           ├── run1_common.py
│           ├── run2_policy_server_launcher.py
│           └── run2_common.py
│
├── training_pipeline/
│   ├── eval1/                 — Training scripts for Eval 1
│   ├── eval2/                 — Training scripts for Eval 2
│   └── eval3/
│       ├── vlm/               — VLM LoRA training (face recognition, position QA)
│       └── vla/               — VLA action model training (cotrain, bridge, run1/run2)
│
├── scripts/
│   ├── download_models.sh     — Download final policies from HuggingFace
│   └── download_datasets.sh   — Download training datasets from HuggingFace
├── lerobot/                   — LeRobot library (git submodule)
├── pyproject.toml
└── uv.lock
```

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
uv sync

# 3. Find and save your robot port
source .venv/bin/activate
lerobot-find-port
echo "export FOLLOWER_PORT=/dev/tty.usbmodemXXXX" >> .env   # replace with your port

# 4. Download models
bash scripts/download_models.sh
```

---

## Running Evals

Each script auto-installs dependencies and reads `FOLLOWER_PORT` from `.env`.

**Eval 1** — *"Put the banana in the blue colored bowl."*
```bash
bash run_eval_1.sh
```

**Eval 2** — *"Put the banana into the bowl to the right of the blue bowl."*
```bash
bash run_eval_2.sh
```

**Eval 3 — TOY** (tasks 1/2) — TOY figures on table, model identifies by face
```bash
TASK="Place the coke on Taylor Swift." bash run_eval_3_TOY.sh
# also: Obama, Yann LeCun, etc.
```

**Eval 3 — Celeb** (task 3) — real celebrity cards
```bash
TASK="Place the coke on Rihanna." bash run_eval_3_celeb.sh
# also: Elon Musk, etc.
```

---

## Hardware Notes

- Follower arm charger: **green** light — Leader arm charger: **blue** light
- Connect the 12V power adapter to the follower arm before running
- Calibration files for this robot are in `robot/calibration/so101_group47/`

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

---

## Data Collection

```bash
source .venv/bin/activate
lerobot-record \
  --robot.type=so101_follower \
  --robot.id=my_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.calibration_dir=robot/calibration/so101_group47/follower \
  --dataset.repo_id=robot-learning-group47/your-dataset \
  --dataset.single_task="Your task." \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=20 \
  --dataset.fps=15 \
  --dataset.push_to_hub=true
```
