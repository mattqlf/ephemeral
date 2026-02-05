#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# 1. Sync local uv environment from lockfile (all project deps, including torch)
PYTHON_VERSION="3.12"
echo "Syncing project environment from uv.lock..."
uv sync --locked --python "$PYTHON_VERSION"

# 2. Install Flash Attention into the uv virtualenv (CUDA 12.8, Torch 2.8, Python 3.12)
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"
echo "Installing Flash Attention from: $FLASH_ATTN_WHEEL"
uv pip install --python .venv/bin/python --no-deps "$FLASH_ATTN_WHEEL"

# 3. Download Model Weights
echo "Enabling hf_transfer for faster downloads..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download Matrix-Game weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Matrix-Game weights already exist, skipping download."
else
  echo "Downloading Matrix-Game weights (~30GB)..."
  uv run --locked --no-sync huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

# Download LingBot-World weights
LINGBOT_DIR="lingbot-world-base-cam"
if [[ -d "$LINGBOT_DIR" && -n "$(ls -A "$LINGBOT_DIR" 2>/dev/null)" ]]; then
  echo "LingBot-World weights already exist, skipping download."
else
  echo "Downloading LingBot-World weights..."
  uv run --locked --no-sync huggingface-cli download robbyant/lingbot-world-base-cam --local-dir "$LINGBOT_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "Environment: Python 3.12, CUDA 12.8, PyTorch 2.8"
echo "============================================"
