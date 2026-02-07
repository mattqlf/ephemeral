#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# 1. Install packages to local SSD instead of network-mounted /workspace.
#    RunPod's /workspace is persistent but slow (network FS). /tmp is local NVMe.
#    This cuts `import torch` from ~120s to ~10s. Re-run this script after pod restart.
export UV_PROJECT_ENVIRONMENT="/tmp/ephemeral-venv"
PYTHON_VERSION="3.12"
echo "Syncing project environment to $UV_PROJECT_ENVIRONMENT..."
uv sync --locked --python "$PYTHON_VERSION"

# Symlink so `uv run` from the project directory finds it
ln -sfn "$UV_PROJECT_ENVIRONMENT" .venv

# 2. Install Flash Attention into the uv virtualenv (CUDA 12.8, Torch 2.8, Python 3.12)
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"
echo "Installing Flash Attention from: $FLASH_ATTN_WHEEL"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps "$FLASH_ATTN_WHEEL"

# 3. Install Hugging Face CLI
if ! command -v hf >/dev/null 2>&1; then
  echo "Installing Hugging Face CLI..."
  curl -LsSf https://hf.co/cli/install.sh | bash
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# 4. Download Model Weights
echo "Enabling hf_transfer for faster downloads..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download Matrix-Game weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Matrix-Game weights already exist, skipping download."
else
  echo "Downloading Matrix-Game weights (~30GB)..."
  hf download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "Environment: Python 3.12, CUDA 12.8, PyTorch 2.8"
echo "============================================"
