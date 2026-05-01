#!/usr/bin/env zsh

# Install the `teleop` keyword for this local clone.
# Run from the repository root:
#   ./src/scripts/setup_teleop_keyword.sh

set -e

REPO_ROOT="$(pwd)"
ZSHRC="$HOME/.zshrc"

START_MARKER="# >>> robot-learning-ethz-vla teleop keyword >>>"
END_MARKER="# <<< robot-learning-ethz-vla teleop keyword <<<"

python - <<PY
from pathlib import Path

zshrc = Path("$ZSHRC")
repo_root = "$REPO_ROOT"

start = "$START_MARKER"
end = "$END_MARKER"

block = f'''{start}
# SO-101 teleoperation with robot camera
teleop() {{
  cd "{repo_root}" || return
  source .venv/bin/activate
  python src/scripts/teleop.py
}}
{end}
'''

text = zshrc.read_text() if zshrc.exists() else ""

if start in text and end in text:
    before = text.split(start)[0]
    after = text.split(end, 1)[1]
    text = before + block + after.lstrip("\\n")
else:
    if text and not text.endswith("\\n"):
        text += "\\n"
    text += "\\n" + block

zshrc.write_text(text)
print(f"Installed teleop keyword in {zshrc}")
print("Run: source ~/.zshrc")
print("Then use: teleop")
PY
