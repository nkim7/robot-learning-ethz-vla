#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    # 1. Define and create the target directory
    output_dir = Path("configs/robot").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Capture the arguments passed to this script
    original_args = sys.argv[1:]

    # 3. Determine if we are calibrating a 'teleop' or a 'robot'
    # Added the missing '--' to the prefix below
    is_teleop = any("teleop" in arg for arg in original_args)
    
    if is_teleop:
        path_override = f"--teleop.calibration_dir={output_dir}"
    else:
        path_override = f"--robot.calibration_dir={output_dir}"

    # 4. Build the command
    cmd = ["lerobot-calibrate"] + original_args + [path_override]

    print(f"🚀 Redirecting calibration files to: {output_dir}")
    print(f"Running command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Calibration failed (exit code {e.returncode}).")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\n❌ Error: 'lerobot-calibrate' not found. Is your environment activated?")
        sys.exit(1)

if __name__ == "__main__":
    main()