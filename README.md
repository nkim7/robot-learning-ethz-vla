# Project 1: Robot Learning (Group 47)

This repository contains the codebase for our Robot Learning project. We use **LeRobot** as our core library, managed as a Git submodule to ensure version consistency across the team while allowing for local modifications.

---

## 📂 Repository Structure

* `configs/`: Configuration files for models, robots, and training.
* `data/`: Local storage for datasets and processed data.
* `lerobot/`: The LeRobot library (Git submodule). Core logic for control and training.
* `src/`: Custom project code.
    * `data_utils/`: Utilities for data processing.
    * `models/`: Custom model architectures.
    * `scripts/`: Python scripts for calibration, training, and evaluation.
* `requirements.txt` & `pyproject.toml`: Dependency management files.

---

## 1. Prerequisites

### System Dependency: FFmpeg
LeRobot requires FFmpeg for video processing. **Install this before setting up the Python environment.**

* **macOS**: `brew install ffmpeg`
* **Ubuntu/Linux**: `sudo apt update && sudo apt install ffmpeg`
* **Windows**: `winget install ffmpeg`

### Python 3.12
This project is optimized for **Python 3.12**. Using `uv` is recommended for automated version management.

---

## 2. Installation & Setup

### Step 1: Clone Recursively
To ensure the `lerobot/` submodule is not empty, you must clone with the `--recursive` flag:

```bash
git clone --recursive https://github.com/lorefrigo/robot-learning-ethz-vla.git
cd project1_group_47
```
*If already cloned, run: `git submodule update --init --recursive`*

### Step 2: Environment Setup
We recommend using `uv` for dependency resolution, but we execute scripts via the standard virtual environment.

#### Option A: Using `uv` (Recommended)
1.  **Sync the project**:
    ```bash
    uv sync
    ```
    *This creates a `.venv` and installs all dependencies, including LeRobot in editable mode.*
2.  **Activate & Use**:
    ```bash
    source .venv/bin/activate
    python --version  # Should be 3.12.x
    ```

#### Option B: Using `pip`
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Hardware & Calibration

### Finding Robot Ports
USB port addresses are specific to your machine and the physical port used. Identify them before starting:

```bash
cd lerobot
lerobot-find-port
```

### Local Environment Variables (`.env`)
To avoid re-entering port addresses, save them in a local `.env` file at the project root. **This file is ignored by Git.**

1.  **Create the file**:
    ```bash
    echo "export LEADER_PORT=/dev/tty.your_port_here" >> .env
    echo "export FOLLOWER_PORT=/dev/tty.your_port_here" >> .env
    ```
2.  **Load the ports**:
    ```bash
    source .env
    ```

### Calibration Workflow
Always run calibration from the **root directory**. Ensure the **12V power adapter** is connected to the follower arm to avoid voltage errors.

* **Leader Arm**:
    ```bash
    python src/scripts/calibrate.py --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=leader_arm
    ```
* **Follower Arm**:
    ```bash
    python src/scripts/calibrate.py --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=follower_arm
    ```

---

## 🛠️ Working with the `lerobot` Submodule

Since `lerobot` is a separate repository (a fork), follow these protocols to keep the team in sync.

### Daily Workflow (Pulling)
When the library is updated, you must update the submodule files after pulling the main repo:
```bash
git pull --recurse-submodules
# OR
git pull origin main && git submodule update --recursive
```

### Modifying the Library (Pushing)
If you edit code *inside* the `lerobot/` folder:
1.  **Commit inside `lerobot`**:
    ```bash
    cd lerobot
    git checkout main
    git add . && git commit -m "Your library fix"
    git push origin main
    ```
2.  **Update the Main Project "Bookmark"**:
    ```bash
    cd ..
    git add lerobot
    git commit -m "chore: update lerobot reference"
    git push origin main
    ```

---

> [!IMPORTANT]
> **Collaborator Rule**: Never commit your physical port paths to Git. Always use the `.env` method described in the Hardware section.