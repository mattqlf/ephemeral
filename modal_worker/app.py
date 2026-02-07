"""
Modal App definition, image, and volume for Ephemeral workers.
"""

import modal

# Persistent volume for model weights
weights_volume = modal.Volume.from_name("ephemeral-weights", create_if_missing=True)

# Pre-built flash-attn wheel (CUDA 12.8, Torch 2.8, Python 3.11)
FLASH_ATTN_WHEEL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl"

# Modal image with all dependencies pre-installed
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchvision==0.23.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate>=1.1.1",
        "diffusers",
        "easydict",
        "einops",
        "huggingface_hub",
        "numpy",
        "omegaconf",
        "open_clip_torch",
        "opencv-python>=4.9.0.80",
        "Pillow",
        "safetensors",
        "scipy",
        "sentencepiece",
        "tokenizers>=0.20.3",
        "transformers>=4.49.0",
        "ftfy",
        "packaging",
        "setuptools",
    )
    .pip_install(FLASH_ATTN_WHEEL, extra_options="--no-deps")
    .pip_install("clip @ git+https://github.com/openai/CLIP.git")
    .env({"PYTHONPATH": "/app"})
    # Copy the project source modules so imports work
    .add_local_dir("pipeline", remote_path="/app/pipeline")
    .add_local_dir("wan", remote_path="/app/wan")
    .add_local_dir("utils", remote_path="/app/utils")
    .add_local_dir("demo_utils", remote_path="/app/demo_utils")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("modal_worker", remote_path="/app/modal_worker")
)

# Lightweight image for the ASGI hub (no torch/CUDA — just FastAPI + Anthropic + Gemini + Suno)
hub_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "anthropic", "google-genai", "httpx", "Pillow")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("web_ui", remote_path="/app/web_ui")
    .add_local_dir("modal_worker", remote_path="/app/modal_worker")
    .add_local_dir("suno_api", remote_path="/app/suno_api")
    .add_local_dir("demo_images", remote_path="/app/demo_images")
)

app = modal.App(name="ephemeral", image=image)
