<p align="center">
<h1 align="center">Matrix-Game 2.0</h1>
<h3 align="center">An Open-Source, Real-Time, and Streaming Interactive World Model</h3>
</p>

<div align='center'>

[[HuggingFace](https://huggingface.co/Skywork/Matrix-Game-2.0)] [[Technical Report](https://matrix-game-v2.github.io/static/pdf/report.pdf)] [[arXiv](https://arxiv.org/abs/2508.13009)] [[Project Website](https://matrix-game-v2.github.io/)]

</div>

https://github.com/user-attachments/assets/336b0d4a-64f5-4e5c-9b60-6212ddb261c0

## Overview

**Matrix-Game-2.0** is an interactive world foundation model for real-time long video generation. Built upon an auto-regressive diffusion-based image-to-world framework, it generates real-time (25fps) long videos conditioned on keyboard and mouse inputs.

**Related Project**: [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D) for explorable 3D scenes.

## Models

| Model | Description | Config | Checkpoint |
|-------|-------------|--------|------------|
| Universal | General scenes | `inference_universal.yaml` | `base_distilled_model/base_distill.safetensors` |
| GTA Drive | Driving scenes | `inference_gta_drive.yaml` | `gta_distilled_model/gta_keyboard2dim.safetensors` |
| Temple Run | Temple Run game | `inference_templerun.yaml` | `templerun_distilled_model/templerun_7dim_onlykey.safetensors` |

## Requirements

- NVIDIA GPU with 24GB+ VRAM (A100, H100 tested)
- Linux
- 64GB RAM

## Installation

**RunPod template**: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game
bash scripts/setup_runpod.sh
```

## Usage

### Web UI (Recommended for Cloud GPUs)

Launch an interactive web interface with real-time video streaming:

```bash
uv run python web_server.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0
```

Then open `http://<your-server-ip>:8080` in your browser. The web UI provides:
- Real-time video frame streaming via WebSocket
- On-screen keyboard controls
- Visual feedback for inputs

### CLI Interactive Streaming

```bash
uv run python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0
```

The script will prompt for the image path and actions interactively.

**Controls:**
- **Universal/GTA**: Camera (I/K/J/L/U) + Movement (W/S/A/D/Q)
- **Temple Run**: W=jump, S=slide, A/D=sidestep, Z/C=turn, Q=none

### Batch Inference

```bash
uv run python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
    --pretrained_model_path Matrix-Game-2.0 \
    --output_folder outputs \
    --num_output_frames 150
```

## Tips

- Upward camera movement may cause brief glitches. Adjust direction to resolve.

## Acknowledgements

[Diffusers](https://github.com/huggingface/diffusers) | [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) | [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | [GameFactory](https://github.com/KwaiVGI/GameFactory) | [MineRL](https://github.com/minerllabs/minerl) | [Video-Pre-Training](https://github.com/openai/Video-Pre-Training)

## License

MIT License - see [LICENSE](LICENSE).

## Citation

```bibtex
@article{he2025matrix,
  title={Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model},
  author={He, Xianglong and Peng, Chunli and Liu, Zexiang and Wang, Boyang and Zhang, Yifan and Cui, Qi and Kang, Fei and Jiang, Biao and An, Mengyin and Ren, Yangyang and Xu, Baixin and Guo, Hao-Xiang and Gong, Kaixiong and Wu, Cyrus and Li, Wei and Song, Xuchen and Liu, Yang and Li, Eric and Zhou, Yahui},
  journal={arXiv preprint arXiv:2508.13009},
  year={2025}
}
```
