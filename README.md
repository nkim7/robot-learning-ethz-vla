# Project 1: Robot Learning (Group 47)

## 📂 Repository Structure

- `configs/`: Configuration files for models, robots, and training.
- `data/`: Local storage for datasets and processed data.
- `lerobot/`: The LeRobot library (Git submodule). Contains core logic for robot control and training.
- `src/`: Custom project code.
    - `data_utils/`: Utilities for data processing and dataset management.
    - `models/`: Custom model architectures.
    - `scripts/`: Python scripts for calibration, training, and evaluation.
- `requirements.txt` & `pyproject.toml`: Dependency management files.

---

This repository contains the codebase for our Robot Learning project. We use **LeRobot** as our core library, managed as a Git submodule to allow for version consistency and local modifications.

---

## 1. Prerequisites

### FFmpeg (System Dependency)
LeRobot requires FFmpeg for processing and streaming video datasets. **You must install this on your operating system before setting up the Python environment.**

* **macOS**: `brew install ffmpeg`
* **Ubuntu/Linux**: `sudo apt update && sudo apt install ffmpeg`
* **Windows**: `winget install ffmpeg` (or download from [ffmpeg.org](https://ffmpeg.org))

### Python 3.12
This project is optimized for **Python 3.12**. We recommend using `uv` to manage Python versions automatically.

---

## 2. Setup Instructions

### Step 1: Clone the Repository
Because this project uses a Git submodule for LeRobot, a standard clone will leave the `lerobot/` folder empty. We must clone recursively:

```bash
git clone --recursive <your-repo-url>
cd project1_group_47
```

If you already cloned it without the flag, run:
```bash
git submodule update --init --recursive
```
### Step 2: Environment Installation

#### Option A: Using `uv`
`uv` is significantly faster and handles the Python version and `uv.lock` file automatically.

1.  **Install uv**: [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
2.  **Sync the project**:
    ```bash
    uv sync
    ```
    *This will automatically download Python 3.12 (if missing), create a `.venv`, and install all dependencies—including LeRobot in editable mode.*

---

#### Option B: Using standard `pip`
1.  **Create and activate a virtual environment**:
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will automatically link the local ./lerobot folder in editable mode.*

---

### 3. How to Manage the Environment

#### Activating the Environment
We do **not** use `uv run` for daily tasks. Instead, use the standard activation workflow:

```bash
source .venv/bin/activate
```

**Verifying the environment:**
```bash
python --version  # Should output 3.12.x
which python      # Should point to the project's .venv folder
```

#### Adding New Dependencies
To ensure the whole group stays in sync, always use `uv` to add new packages. This updates `pyproject.toml` and `uv.lock`:

```bash
uv add <package-name>
```
The collaborators should then run `uv sync` to receive the updated environment.

---

### 4. Handling the LeRobot Submodule
The `lerobot/` directory is a separate Git repository nested inside ours.

* **Editable Mode**: It is installed as an "editable" package (`-e`). Any changes made inside the `lerobot/` folder will immediately affect the imports (`import lerobot`) in the main project.
* **Tracking**: Do not commit changes *inside* the `lerobot` folder to our main repo. Git only tracks the "commit hash" that the submodule points to.
* **Updating**: If the submodule version is updated by a teammate, you will see it in `git status`. Update your local folder with:

```bash
git submodule update --init --recursive
uv sync
```


## 🔌 Hardware Connectivity & Ports, Calibration

### Finding Robot Ports
USB port addresses (e.g., `/dev/tty.usbmodem5B141127361`) are **specific to the laptop and the physical USB port used.** When moved the robot to a new computer, the address will change.

Before running calibration or teleop scripts, identify the current ports:

```bash
cd lerobot
lerobot-find-port
```

### 🔧 Saving your Ports (Local Shortcut)
To avoid looking up your port addresses every time, save them locally in your shell environment.

1. Create a file named `.env` in the project root:
   ```bash
   echo "export LEADER_PORT={your leader port}" >> .env
   echo "export FOLLOWER_PORT={your follower port}" >> .env
   ```

2. Load the paths into your terminal:
    ```Bash
    source .env
    ```

### 🔧 Calibration
If you need to (re)calibrate, from the *root directory*:

1. For the leader arm:
    ```Bash
    python src/scripts/calibrate.py --teleop.port=$LEADER_PORT --teleop.type=so101_leader --robot.id=leader_arm
    ```
2. For the follower arm:
    ```Bash
    python src/scripts/calibrate.py --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=follower_arm
    ```
