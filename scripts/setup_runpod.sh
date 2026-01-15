#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# Detect GPU architecture
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  echo "Detected GPU: $GPU_NAME"
  case "$GPU_NAME" in
    *H100*) ARCH="9.0" ;;
    *A100*) ARCH="8.0" ;;
    *L40*|*L4*|*RTX\ 40*|*4090*|*4080*|*4070*|*4060*) ARCH="8.9" ;;
    *A10*) ARCH="8.6" ;;
    *T4*) ARCH="7.5" ;;
    *V100*) ARCH="7.0" ;;
    *) ARCH="${TORCH_CUDA_ARCH_LIST:-}" ;;
  esac
else
  ARCH="${TORCH_CUDA_ARCH_LIST:-}"
fi

if [[ -z "${ARCH}" ]]; then
  echo "Unable to infer TORCH_CUDA_ARCH_LIST. Set it explicitly, e.g.:" >&2
  echo "  export TORCH_CUDA_ARCH_LIST=9.0" >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="$ARCH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Flash-attn pre-built wheel URL (PyTorch 2.4 + CUDA 12 + Python 3.10)
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Remove flash-attn from pyproject.toml to avoid building from source
echo "Removing flash-attn from pyproject.toml (will install pre-built wheel)..."
sed -i.bak '/^[[:space:]]*"flash-attn",*$/d' pyproject.toml

# Create venv and install dependencies (without flash-attn)
echo "Creating virtual environment..."
uv venv --python 3.10

echo "Installing dependencies (this may take a few minutes)..."
uv sync

# Install pre-built flash-attn wheel (no compilation needed)
echo "Installing pre-built flash-attn wheel..."
uv pip install "$FLASH_ATTN_WHEEL"

# Restore original pyproject.toml
mv pyproject.toml.bak pyproject.toml

# Download model weights from HuggingFace
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Model weights already exist in $WEIGHTS_DIR, skipping download."
else
  echo "Downloading model weights from HuggingFace (~30GB)..."
  uv run huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Run interactive streaming:"
echo ""
echo "  uv run python inference_streaming.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --pretrained_model_path Matrix-Game-2.0 \\"
echo "      --img_path demo_images/universal/0000.png"
echo ""
