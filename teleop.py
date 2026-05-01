#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    """
    Wrapper script for lerobot-teleoperate that forces the use of local calibration data.
    """
    # Define calibration paths relative to the workspace root
    # Based on the structure:
    # calibration/so101_group47/follower/follower_arm.json
    # calibration/so101_group47/leader/leader_arm.json
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    follower_cal = os.path.join(script_dir, "calibration", "so101_group47", "follower")
    leader_cal = os.path.join(script_dir, "calibration", "so101_group47", "leader")

    # Base command
    cmd = ["lerobot-teleoperate"]
    
    # Forced calibration arguments
    forced_args = [
        f"--robot.calibration_dir={follower_cal}",
        f"--teleop.calibration_dir={leader_cal}",
    ]
    
    # The user's arguments come first, then our forced arguments.
    # In draccus/argparse, the last occurrence of a flag usually takes precedence.
    full_cmd = cmd + sys.argv[1:] + forced_args

    print(f"Running command: {' '.join(full_cmd)}")
    
    try:
        # We use os.execvp to replace the current process with lerobot-teleoperate.
        # This ensures signals (like Ctrl+C) are handled correctly by the underlying script.
        os.execvp(full_cmd[0], full_cmd)
    except FileNotFoundError:
        print("Error: 'lerobot-teleoperate' command not found. Make sure lerobot is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
