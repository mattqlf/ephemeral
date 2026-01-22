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
# Official wheels from: https://github.com/Dao-AILab/flash-attention/releases
# Detect CXX11 ABI setting from PyTorch
CXX11_ABI="FALSE"
if [[ -n "$TORCH_VERSION" ]]; then
  CXX11_ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "FALSE")
  echo "Detected CXX11 ABI: $CXX11_ABI"
fi

if [[ "$USE_B200_CONFIG" == true ]]; then
  # PyTorch 2.8 wheels from official Dao-AILab releases
  FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${CXX11_ABI}-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_x86_64.whl"
else
  # Standard config: PyTorch 2.5 + CUDA 12 + Python 3.10
  FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
fi

echo "Flash-attn wheel: $FLASH_ATTN_WHEEL"

# Remove stale lockfile (may contain old flash-attn reference)
rm -f uv.lock

if [[ "$USE_B200_CONFIG" == true ]]; then
  # B200/Blackwell requires PyTorch with sm_100 support
  # The RunPod template's PyTorch may not have sm_100, so we install from official cu128 index
  echo ""
  echo "Creating virtual environment for B200..."
  uv venv --python "$PYTHON_VERSION"

  # Install PyTorch with sm_100 (Blackwell) support from official index
  echo "Installing PyTorch 2.8.0 with Blackwell (sm_100) support..."
  uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

  # Re-detect CXX11 ABI after installing PyTorch
  CXX11_ABI=$(uv run python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')" 2>/dev/null || echo "FALSE")
  echo "Detected CXX11 ABI: $CXX11_ABI"
  FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${CXX11_ABI}-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_x86_64.whl"

  # Install dependencies
  echo "Installing dependencies..."
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

  echo "Installing flash-attn (pre-built wheel for PyTorch 2.8)..."
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
