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

# Select flash-attn wheel for standard config only
# B200 config builds from source due to ABI compatibility issues with pre-built wheels
if [[ "$USE_B200_CONFIG" != true ]]; then
  FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  echo "Flash-attn wheel: $FLASH_ATTN_WHEEL"
fi

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

  # Build flash-attn from source for guaranteed ABI compatibility
  echo "Installing build dependencies for flash-attn..."
  uv pip install ninja packaging setuptools

  echo "Building flash-attn from source (this takes 5-10 minutes)..."
  MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

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

# Install scipy (needed by LingBot-World cam_utils)
echo "Installing scipy..."
uv pip install scipy

# Download model weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Matrix-Game weights already exist, skipping download."
else
  echo "Downloading Matrix-Game weights (~30GB)..."
  uv run huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

# Download LingBot-World weights
LINGBOT_DIR="lingbot-world-base-cam"
if [[ -d "$LINGBOT_DIR" && -n "$(ls -A "$LINGBOT_DIR" 2>/dev/null)" ]]; then
  echo "LingBot-World weights already exist, skipping download."
else
  echo "Downloading LingBot-World weights..."
  uv run huggingface-cli download robbyant/lingbot-world-base-cam --local-dir "$LINGBOT_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""

echo "Dream v1 (Matrix-Game) - real-time, 1 GPU:"
echo ""
echo "  uv run python web_server.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
echo "      --pretrained_model_path Matrix-Game-2.0 \\"
echo "      --port 8888"
echo ""
echo "Dream v2 (LingBot-World) - chunk-based, 8 GPUs:"
echo ""
echo "  uv run python launch_lingbot.py \\"
echo "      --num_gpus 8 \\"
echo "      --ckpt_dir lingbot-world-base-cam \\"
echo "      --dit_fsdp --t5_fsdp \\"
echo "      --port 8889"
echo ""
echo "Or run CLI mode (Matrix-Game):"
echo ""
echo "  uv run python inference_streaming.py \\"
echo "      --config_path configs/inference_yaml/inference_universal.yaml \\"
echo "      --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
echo "      --pretrained_model_path Matrix-Game-2.0"

echo ""
echo "Then open the Jupyter Lab HTTP link in RunPod to access the web UI."
echo ""
