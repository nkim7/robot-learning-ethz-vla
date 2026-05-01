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

## Data Collection & Collaboration

This section outlines the workflow for obtaining credentials, setting up your environment, and recording teleoperation data for the **SO-101** robot. We use a shared **Hugging Face Organization** to centralize all training data.

---

### 1. Obtain a Hugging Face Access Token
Before you can upload data, you need a personal "Write" token from Hugging Face.

1.  Log in to [Hugging Face](https://huggingface.co/).
2.  Navigate to **Settings > Access Tokens** (or go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3.  Click **New token**.
4.  Set the **Token type** to **Write**.
5.  Give it a name (e.g., `robot-learning-group47-yourname`) and click **Generate a token**.
6.  **Copy the token immediately**—you will not be able to see the full string again once you leave the page.

---

### 2. Configure Environment Variables (`.env`)
To avoid entering your token every time you run a command, store it in a local environment file.

1.  Create a file named `.env` in the root of this project.
2.  Paste your token into the file using these two variables (the same token for both):
    ```env
    HUGGINGFACE_TOKEN=hf_your_token_here
    HF_TOKEN=hf_your_token_here
    ```
3.  **Security Check:** Ensure your `.gitignore` file contains a line for `.env` so you do not accidentally share your token with the team or the public.

To load these variables into your current terminal session, run:
```bash
# Activate the virtual environment
source .venv/bin/activate

# Load the tokens
export $(grep -v '^#' .env | xargs)
```

---

### 3. Initialize the Dataset Repository
The first time the team records data for a specific project, a repository needs to be created under the Organization namespace. Running the script with the `--push-to-hub=true` flag handles this automatically.

**First-time Authentication:**
```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

**Run a test recording to create the repo:**
```bash
python lerobot/scripts/control_robot.py record \
  --robot.type=so101 \
  --repo-id=robot-learning-group47/so101-teleoperation \
  --push-to-hub=true \
  --num-episodes=1
```


### 4. Recording Teleoperation Data
Once the environment is set up, use the following workflow to record training episodes.

**Recording Workflow:**
```bash
# 1. Activate and Load
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

# 2. Record Session
python lerobot/scripts/control_robot.py record \
  --robot.type=so101 \
  --repo-id=your-org-name/your-dataset-name \
  --push-to-hub=true \
  --fps=30 \
  --warmup-time-s=5 \
  --episode-time-s=30 \
  --reset-time-s=5 \
  --num-episodes=10
```

### Script Parameter Reference

| Parameter | Description |
| :--- | :--- |
| **`--robot.type`** | Must be `so101` for our hardware. |
| **`--repo-id`** | The Organization path (e.g., `org/dataset-name`). |
| **`--push-to-hub`** | Automatically uploads data to the cloud after recording. |
| **`--num-episodes`** | Number of successful demonstrations to capture. |
| **`--episode-time-s`** | Max time for one demonstration (default 30s). |
| **`--fps`** | The frequency of the control loop (standard is 30Hz). |
```
---

> [!IMPORTANT]
> **Collaborator Rule**: Never commit your physical port paths to Git. Always use the `.env` method described in the Hardware section.
> **About the robots**: Each robot works with its own bus driver and its own charger. The charger for the follower arm has a green light, the one for the leader arm is blue.