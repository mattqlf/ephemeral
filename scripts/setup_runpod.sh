#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# Detect GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  echo "Detected GPU: $GPU_NAME"
fi

# Flash-attn pre-built wheel (PyTorch 2.5 + CUDA 12 + Python 3.10)
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Remove stale lockfile (may contain old flash-attn reference)
rm -f uv.lock

# Create venv and install dependencies
echo "Creating virtual environment..."
uv venv --python 3.10

echo "Installing dependencies..."
uv sync

echo "Installing flash-attn (pre-built wheel)..."
uv pip install "$FLASH_ATTN_WHEEL"

# Download model weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Model weights already exist, skipping download."
else
  echo "Downloading model weights (~30GB)..."
  uv run hf download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "  uv run python inference_streaming.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --pretrained_model_path Matrix-Game-2.0 \\"
echo "      --img_path demo_images/universal/0000.png"
echo ""
