"""
Upload model weights to Modal Volume.

Usage (download from HuggingFace directly inside Modal — fastest):
    modal run modal_worker/upload_weights.py::download_weights

Usage (upload from local directory):
    modal run modal_worker/upload_weights.py::upload_weights --local-weights-dir Matrix-Game-2.0

This puts the following on the 'ephemeral-weights' volume:
  - base_distill.safetensors (main checkpoint)
  - Wan2.1_VAE.pth (VAE weights)
  - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (CLIP model)
  - xlm-roberta-large/ (tokenizer directory)
"""

import os
import modal
from modal_worker.app import app, weights_volume

WEIGHTS_PATH = "/weights"
HF_REPO = "Skywork/Matrix-Game-2.0"

# HuggingFace paths → flat names on the volume
WEIGHT_FILES = {
    "base_distilled_model/base_distill.safetensors": "base_distill.safetensors",
    "Wan2.1_VAE.pth": "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
}

WEIGHT_DIRS = [
    "xlm-roberta-large",
]

# Minimal image for downloading (just needs huggingface_hub)
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/app"})
    .add_local_dir("modal_worker", remote_path="/app/modal_worker")
)


@app.function(
    image=download_image,
    volumes={WEIGHTS_PATH: weights_volume},
    timeout=3600,
)
def download_weights_fn():
    """Download weights from HuggingFace directly to the Modal volume."""
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"Downloading weights from {HF_REPO} to Modal Volume...")

    # Download individual weight files
    for hf_path, vol_name in WEIGHT_FILES.items():
        dest = os.path.join(WEIGHTS_PATH, vol_name)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  Already exists: {vol_name} ({size_mb:.0f} MB), skipping")
            continue
        print(f"  Downloading {hf_path} -> {vol_name}...")
        tmp = hf_hub_download(repo_id=HF_REPO, filename=hf_path)
        shutil.copy2(tmp, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Done: {vol_name} ({size_mb:.0f} MB)")

    # Download tokenizer directory
    for dir_name in WEIGHT_DIRS:
        dest_dir = os.path.join(WEIGHTS_PATH, dir_name)
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            print(f"  Already exists: {dir_name}/, skipping")
            continue
        print(f"  Downloading {dir_name}/...")
        tmp_dir = snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=f"{dir_name}/*",
        )
        # Copy from cache to volume
        src_dir = os.path.join(tmp_dir, dir_name)
        os.makedirs(dest_dir, exist_ok=True)
        count = 0
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, src_dir)
                dst = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
        print(f"  Done: {dir_name}/ ({count} files)")

    weights_volume.commit()
    print("\nAll weights downloaded to Modal Volume!")


@app.local_entrypoint(name="download-weights")
def download_weights():
    """Download weights from HuggingFace directly inside Modal (fast)."""
    download_weights_fn.remote()
    print("You can now run: modal run modal_worker/worker.py")


@app.local_entrypoint(name="upload-weights")
def upload_weights(local_weights_dir: str = "Matrix-Game-2.0"):
    """Upload weights from a local directory to Modal Volume."""
    if not os.path.isdir(local_weights_dir):
        print(f"ERROR: Directory not found: {local_weights_dir}")
        print("Usage: modal run modal_worker/upload_weights.py::upload_weights --local-weights-dir Matrix-Game-2.0")
        return

    print(f"Uploading weights from {local_weights_dir} to Modal Volume 'ephemeral-weights'...")

    for hf_path, vol_name in WEIGHT_FILES.items():
        local_path = os.path.join(local_weights_dir, hf_path)
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  Uploading {hf_path} ({size_mb:.0f} MB) -> /{vol_name}")
            with open(local_path, "rb") as f:
                weights_volume.write_file(vol_name, f)
            print(f"  Done: {vol_name}")
        else:
            print(f"  WARNING: {local_path} not found, skipping")

    for dir_name in WEIGHT_DIRS:
        local_dir = os.path.join(local_weights_dir, dir_name)
        if os.path.isdir(local_dir):
            file_count = 0
            for root, _dirs, files in os.walk(local_dir):
                for fname in files:
                    local_file = os.path.join(root, fname)
                    rel = os.path.relpath(local_file, local_weights_dir)
                    size_mb = os.path.getsize(local_file) / (1024 * 1024)
                    print(f"  Uploading {rel} ({size_mb:.1f} MB)")
                    with open(local_file, "rb") as f:
                        weights_volume.write_file(rel, f)
                    file_count += 1
            print(f"  Done: {dir_name}/ ({file_count} files)")
        else:
            print(f"  WARNING: {local_dir} not found, skipping")

    weights_volume.commit()
    print("\nAll weights uploaded successfully!")
    print("You can now run: modal run modal_worker/worker.py")
