#!/usr/bin/env bash
# Install semg2pose, emg2pose and UmeTrack (editable) into the current Python, using pip.
# Used by the setup cell at the top of each notebook on Colab; locally, use `uv sync` instead.
#
# pip does not read [tool.uv.sources], so the emg2pose and UmeTrack requirements of semg2pose
# are satisfied by listing the submodule folders in the same pip command.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

git submodule update --init --recursive

BACKEND=cpu
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND=cu121
fi
echo "Installing torch 2.3.1 ($BACKEND)..."
# torchvision/torchaudio are pinned too so Colab's preinstalled ones stay compatible.
"$PYTHON" -m pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url "https://download.pytorch.org/whl/$BACKEND"

echo "Installing semg2pose + emg2pose + UmeTrack..."
"$PYTHON" -m pip install -q -e emg2pose/emg2pose/UmeTrack -e emg2pose -e .
