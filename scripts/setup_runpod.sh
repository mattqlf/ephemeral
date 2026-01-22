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
GPU_NAME=""
IS_B200=false
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  echo "Detected GPU: $GPU_NAME"
  # Check if B200 or Blackwell architecture
  if [[ "$GPU_NAME" == *"B200"* ]] || [[ "$GPU_NAME" == *"B100"* ]] || [[ "$GPU_NAME" == *"GB200"* ]]; then
    IS_B200=true
    echo "Blackwell GPU detected - using PyTorch 2.8+ configuration"
  fi
fi

# Detect existing PyTorch version (for pre-configured RunPod templates)
TORCH_VERSION=""
if command -v python3 >/dev/null 2>&1; then
  TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
fi
if [[ -n "$TORCH_VERSION" ]]; then
  echo "Detected existing PyTorch: $TORCH_VERSION"
fi

# Detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.11")
echo "Detected Python: $PYTHON_VERSION"

# Determine configuration based on environment
# B200s require PyTorch 2.8+ and CUDA 12.8+
if [[ "$IS_B200" == true ]] || [[ "$TORCH_VERSION" == 2.8* ]] || [[ "$TORCH_VERSION" == 2.9* ]] || [[ "$TORCH_VERSION" == 2.10* ]]; then
  echo ""
  echo "Using B200/PyTorch 2.8+ configuration..."
  USE_B200_CONFIG=true
else
  echo ""
  echo "Using standard PyTorch 2.5 configuration..."
  USE_B200_CONFIG=false
fi

# Select flash-attn wheel based on configuration
# Pre-built wheels from: https://github.com/mjun0812/flash-attention-prebuild-wheels
if [[ "$USE_B200_CONFIG" == true ]]; then
  case "$PYTHON_VERSION" in
    3.10)
      FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
      ;;
    3.11)
      FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl"
      ;;
    3.12)
      FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"
      ;;
    3.13)
      FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-2.8.3+cu128torch2.8-cp313-cp313-linux_x86_64.whl"
      ;;
    *)
      echo "Warning: Unsupported Python version $PYTHON_VERSION for B200 config, defaulting to Python 3.11 wheel"
      FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl"
      ;;
  esac
else
  # Standard config: PyTorch 2.5 + CUDA 12 + Python 3.10
  FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
fi

echo "Flash-attn wheel: $FLASH_ATTN_WHEEL"

# Remove stale lockfile (may contain old flash-attn reference)
rm -f uv.lock

if [[ "$USE_B200_CONFIG" == true ]] && [[ -n "$TORCH_VERSION" ]]; then
  # B200 RunPod template: PyTorch is pre-installed
  # Create venv with --system-site-packages to inherit torch
  echo ""
  echo "Creating virtual environment (inheriting system PyTorch $TORCH_VERSION)..."
  uv venv --system-site-packages --python "$PYTHON_VERSION"

  # Install dependencies without overriding torch
  echo "Installing dependencies (excluding torch)..."
  uv pip install \
    "accelerate>=1.1.1" \
    "av" \
    "clip @ git+https://github.com/openai/CLIP.git" \
    "dashscope" \
    "diffusers" \
    "dominate" \
    "easydict" \
    "einops" \
    "fastapi" \
    "flask" \
    "flask-socketio" \
    "ftfy" \
    "huggingface_hub[cli]" \
    "imageio" \
    "imageio-ffmpeg" \
    "lmdb" \
    "matplotlib" \
    "numpy" \
    "omegaconf" \
    "open_clip_torch" \
    "opencv-python>=4.9.0.80" \
    "pycocotools" \
    "pydantic" \
    "safetensors" \
    "scikit-image" \
    "sentencepiece" \
    "starlette" \
    "tokenizers>=0.20.3" \
    "tqdm" \
    "transformers>=4.49.0" \
    "uvicorn[standard]" \
    "wandb"

  echo "Installing flash-attn (pre-built wheel for PyTorch 2.8 + CUDA 12.8)..."
  uv pip install "$FLASH_ATTN_WHEEL"

  echo "Installing setuptools (required for torch.compile)..."
  uv pip install setuptools

  echo "Installing hf_transfer (for fast model downloads)..."
  uv pip install hf_transfer
else
  # Standard setup: create venv with Python 3.10
  echo "Creating virtual environment..."
  uv venv --python 3.10

  echo "Installing dependencies..."
  uv sync

  echo "Installing flash-attn (pre-built wheel)..."
  uv pip install "$FLASH_ATTN_WHEEL"

  echo "Installing setuptools (required for torch.compile)..."
  uv pip install setuptools

  echo "Installing hf_transfer (for fast model downloads)..."
  uv pip install hf_transfer
fi

# Download model weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Model weights already exist, skipping download."
else
  echo "Downloading model weights (~30GB)..."
  uv run huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""

echo "To run the web UI (recommended):"
echo ""
echo "  uv run python web_server.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
echo "      --pretrained_model_path Matrix-Game-2.0 \\"
echo "      --port 8888"
echo ""
echo "Or run CLI mode:"
echo ""
echo "  uv run python inference_streaming.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
echo "      --pretrained_model_path Matrix-Game-2.0"

echo ""
echo "Then open the Jupyter Lab HTTP link in RunPod to access the web UI."
echo ""
