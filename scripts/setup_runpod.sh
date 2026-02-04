#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# 1. Install Flash Attention from prebuilt wheel (CUDA 12.8, Torch 2.8, Python 3.12)
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"
echo "Installing Flash Attention from: $FLASH_ATTN_WHEEL"
uv pip install --system "$FLASH_ATTN_WHEEL"

# 2. Install remaining dependencies
echo "Installing dependencies..."
uv pip install --system \
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
  "wandb" \
  "scipy" \
  "hf_transfer"

# 3. Download Model Weights
echo "Enabling hf_transfer for faster downloads..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download Matrix-Game weights
WEIGHTS_DIR="Matrix-Game-2.0"
if [[ -d "$WEIGHTS_DIR" && -n "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
  echo "Matrix-Game weights already exist, skipping download."
else
  echo "Downloading Matrix-Game weights (~30GB)..."
  huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir "$WEIGHTS_DIR"
fi

# Download LingBot-World weights
LINGBOT_DIR="lingbot-world-base-cam"
if [[ -d "$LINGBOT_DIR" && -n "$(ls -A "$LINGBOT_DIR" 2>/dev/null)" ]]; then
  echo "LingBot-World weights already exist, skipping download."
else
  echo "Downloading LingBot-World weights..."
  huggingface-cli download robbyant/lingbot-world-base-cam --local-dir "$LINGBOT_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "Environment: Python 3.12, CUDA 12.8, PyTorch 2.8"
echo "============================================"
