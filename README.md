<p align="center">
<h1 align="center">Ephemeral</h1>
<h3 align="center">Real-Time Interactive World Generation</h3>
</p>

> *ephemeral* /ɪˈfem(ə)rəl/ — lasting for a very short time; transient, momentary, fleeting.

Worlds that never were and never will be again. Each moment rendered from thought and gone before you blink.

## Overview

**Ephemeral** generates real-time (25fps) interactive video from a single image. Control the camera and movement with keyboard inputs to explore AI-generated worlds.

## Requirements

- NVIDIA GPU with 24GB+ VRAM (A100, H100, H200, B200 recommended)
- Linux
- 64GB RAM

### Multi-GPU Options

| Setup | GPUs | VRAM per GPU | Best For |
|-------|------|--------------|----------|
| Single GPU | 1 | 24GB+ | Basic usage |
| Speculative | 2 | 24GB each | Interactive (same-key sequences) |
| Tensor Parallel | 2-4 | 16GB+ each | Lower latency |
| TP+Speculative | 4-8 | 16GB+ each | Best of both (lowest latency + speculation) |

## Installation

**RunPod template**: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game
bash scripts/setup_runpod.sh
```

## Quick Start

Launch the web UI:

```bash
uv run python -m apps.servers.web_server \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0 \
    --port 8888 \
    --fast \
    --no-warmup
```

Then open `http://<your-server-ip>:8888` in your browser.

### Flags

| Flag | Description |
|------|-------------|
| `--fast` | Use 2 denoising steps instead of 3 (~33% faster, slight quality reduction) |
| `--port` | Server port (default: 8080) |

### Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/back/right |
| Q | No movement |
| I/J/K/L | Camera up/left/down/right |
| U | No camera movement |

Press any movement key to move (camera defaults to neutral). Press any camera key to look (movement defaults to neutral).

## CLI Usage

For terminal-based interaction:

```bash
uv run python -m apps.inference.inference_streaming \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0
```

## Multi-GPU Inference

For improved performance with multiple GPUs, two parallelism strategies are available:

### Speculative Pipelining (2 GPUs)

Uses a second GPU to pre-generate the next block assuming the same action. When the user continues holding the same key, the frame is delivered instantly.

```bash
uv run python -m apps.servers.web_server_speculative \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0 \
    --port 8888 --fast
```

### Tensor Parallelism (2-4 GPUs)

Splits the model across GPUs to reduce per-block latency. Supports 2, 3, 4, or 6 GPUs.

```bash
# Using the launcher (recommended)
uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 2 \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --port 8888 --fast

# Or directly with torchrun
uv run torchrun --nproc_per_node=2 --module apps.servers.web_server_tensor_parallel \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --port 8888 --fast
```

### TP+Speculative Hybrid (4+ GPUs)

Combines tensor parallelism with speculative pipelining for the best performance. GPUs are split evenly:
- First half: Primary TP group (generates current block)
- Second half: Speculative TP group (pre-generates next block)

```bash
# 4 GPUs: 2 for primary TP, 2 for speculative TP
uv run python -m apps.launchers.launch_tp_speculative --num_gpus 4 \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --port 8888 --fast

# 6 GPUs: 3 for primary TP, 3 for speculative TP
uv run python -m apps.launchers.launch_tp_speculative --num_gpus 6 \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --port 8888 --fast
```

## Performance

| Hardware | Time per block | Effective FPS |
|----------|----------------|---------------|
| 1x A100 80GB | ~2.2s | ~5-6 |
| 1x A100 + `--fast` | ~1.5s | ~8 |
| 2x A100 speculative | ~0.1-1.5s | ~10-25* |
| 2x A100 tensor parallel | ~1.3s | ~9 |
| 4x A100 tensor parallel | ~0.9s | ~13 |
| 4x A100 TP+speculative | ~0.1-0.9s | ~15-30* |
| 6x A100 TP+speculative | ~0.1-0.7s | ~18-35* |
| 1x B200 | ~0.8-1.0s | ~12-15 |
| 1x B200 + `--fast` | ~0.5-0.7s | ~17-20 |

*Speculative FPS depends on hit rate (higher when holding same key)

## Tips

- Keep the server running to avoid reloading models (~2-3 min startup)
- Upward camera movement may cause brief glitches — adjust direction to resolve
- The `--fast` flag is recommended for interactive use

## License

MIT License

---

## Credits

Built on [Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game) by Skywork AI.

```bibtex
@article{he2025matrix,
  title={Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model},
  author={He, Xianglong and Peng, Chunli and Liu, Zexiang and Wang, Boyang and Zhang, Yifan and Cui, Qi and Kang, Fei and Jiang, Biao and An, Mengyin and Ren, Yangyang and Xu, Baixin and Guo, Hao-Xiang and Gong, Kaixiong and Wu, Cyrus and Li, Wei and Song, Xuchen and Liu, Yang and Li, Eric and Zhou, Yahui},
  journal={arXiv preprint arXiv:2508.13009},
  year={2025}
}
```
