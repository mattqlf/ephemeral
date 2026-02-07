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

For multi-world mode, each world runs on its own GPU (~14GB model weights per GPU in bf16).

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
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --port 8888 \
    --no-compile
```

Then open `http://<your-server-ip>:8888` in your browser.

### Flags

| Flag | Description |
|------|-------------|
| `--resolution WxH` | Output resolution (default: `640x352`). Supported: `832x480`, `1280x720`, `480x832`, `720x1280` |
| `--num-frames N` | Frames per generation block (default: 3). Higher = more temporal context per block |
| `--num-worlds N` | Parallel worlds, each on its own GPU. Displayed in a grid, each diverging from independent actions |
| `--fast` | Use 2 denoising steps instead of 3 (~33% faster, slight quality loss) |
| `--no-compile` | Skip torch.compile for faster startup (implies --no-warmup) |
| `--port` | Server port (default: 8080) |

### Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/back/right |
| Q | No movement |
| I/J/K/L | Camera up/left/down/right |
| U | No camera movement |

Press any movement key to move (camera defaults to neutral). Press any camera key to look (movement defaults to neutral).

### Parallel Worlds

Run multiple worlds from the same starting image to see how different actions cause them to diverge. Each world runs on its own GPU with a full model instance, so they update independently at full frame rate:

```bash
# 2 worlds on 2 GPUs
uv run python -m apps.servers.web_server \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --num-worlds 2 --no-compile
```

- **N=1** (default): single world on one GPU
- **N=2**: side-by-side split. You control world 1; world 2 acts randomly
- **N=4**: 2x2 grid. You control world 1; worlds 2-4 act randomly

Requires N GPUs with 24GB+ VRAM each. In multiplayer, remote players can choose which world to control.

## CLI Usage

For terminal-based interaction:

```bash
uv run python -m apps.inference.inference_streaming \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0
```

## Performance

| Hardware | Time per block | Effective FPS |
|----------|----------------|---------------|
| 1x A100 80GB | ~2.2s | ~5-6 |
| 1x A100 + `--fast` | ~1.5s | ~8 |
| 1x B200 | ~0.8-1.0s | ~12-15 |
| 1x B200 + `--fast` | ~0.5-0.7s | ~17-20 |

Multi-world mode runs each world at the same per-GPU FPS since each has its own model instance.

## Tips

- Keep the server running to avoid reloading models (~2-3 min startup)
- Upward camera movement may cause brief glitches — adjust direction to resolve
- Use `--resolution 832x480` on H200/B200 for higher quality at the cost of speed
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
