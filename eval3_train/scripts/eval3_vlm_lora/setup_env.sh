#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL:-}"

mkdir -p "${ROOT_DIR}/data" "${ROOT_DIR}/checkpoint" "${UV_CACHE_DIR}"

if command -v uv >/dev/null 2>&1; then
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    env UV_CACHE_DIR="${UV_CACHE_DIR}" uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
  fi

  if [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then
    env UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install \
      --python "${VENV_DIR}/bin/python" \
      --extra-index-url "${TORCH_EXTRA_INDEX_URL}" \
      -r "${SCRIPT_DIR}/requirements.txt"
  else
    env UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install \
      --python "${VENV_DIR}/bin/python" \
      -r "${SCRIPT_DIR}/requirements.txt"
  fi
else
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  "${VENV_DIR}/bin/python" -m ensurepip --upgrade
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  if [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then
    "${VENV_DIR}/bin/python" -m pip install --extra-index-url "${TORCH_EXTRA_INDEX_URL}" -r "${SCRIPT_DIR}/requirements.txt"
  else
    "${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
  fi
fi

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/check_env.py"
