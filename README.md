<p align="center">
<h1 align="center">Matrix-Game 2.0</h1>
<h3 align="center">An Open-Source, Real-Time, and Streaming Interactive World Model</h3>
</p>

<div align='center'>

[[HuggingFace](https://huggingface.co/Skywork/Matrix-Game-2.0)] [[Technical Report](https://matrix-game-v2.github.io/static/pdf/report.pdf)] [[arXiv](https://arxiv.org/abs/2508.13009)] [[Project Website](https://matrix-game-v2.github.io/)]

</div>

https://github.com/user-attachments/assets/336b0d4a-64f5-4e5c-9b60-6212ddb261c0

## Overview

**Matrix-Game-2.0** is an interactive world foundation model for real-time long video generation. Built upon an auto-regressive diffusion-based image-to-world framework, it generates real-time (25fps) long videos conditioned on keyboard and mouse inputs, enabling fine-grained control and dynamic scene evolution.

**Related Project**: For explorable large-scale 3D scenes that integrate with games or VR applications, see [Matrix-3D](https://github.com/SkyworkAI/Matrix-3D).

## Models

Three pretrained model weights are available on [HuggingFace](https://huggingface.co/Skywork/Matrix-Game-2.0):

| Model | Description | Config |
|-------|-------------|--------|
| Universal | General scenes | `inference_universal.yaml` |
| GTA Drive | GTA driving scenes | `inference_gta_drive.yaml` |
| Temple Run | Temple Run game | `inference_templerun.yaml` |

## Requirements

- NVIDIA GPU with at least 24GB VRAM (A100, H100 tested)
- Linux operating system
- 64GB RAM

## Installation

### RunPod / Cloud GPU (Recommended)

Use template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game
bash scripts/setup_runpod.sh
```

This script installs all dependencies and downloads model weights (~30GB).

### Manual Installation

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game

# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv --python 3.10
uv sync

# Download model weights
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
```

## Usage

### Interactive Streaming (Real-time Control)

Generate videos with real-time keyboard/mouse input:

```bash
# Universal mode
uv run python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --img_path demo_images/universal/0000.png

# GTA driving mode
uv run python inference_streaming.py \
    --config_path configs/inference_yaml/inference_gta_drive.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --img_path demo_images/gta_drive/0000.png

# Temple Run mode
uv run python inference_streaming.py \
    --config_path configs/inference_yaml/inference_templerun.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --img_path demo_images/templerun/0000.png
```

**Controls:**
- **Universal/GTA**: Camera (I/K/J/L/U) + Movement (W/S/A/D/Q)
- **Temple Run**: Actions (W=jump, S=slide, A/D=sidestep, Z/C=turn, Q=none)

### Batch Inference (Random Actions)

Generate videos with random action trajectories:

```bash
uv run python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --img_path demo_images/universal/0000.png \
    --output_folder outputs \
    --num_output_frames 150 \
    --seed 42
```

### Scripted Actions

Run with a predefined action sequence:

```bash
uv run python inference_streaming.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --pretrained_model_path Matrix-Game-2.0 \
    --img_path demo_images/universal/0000.png \
    --actions_path actions.json \
    --output_name demo_stream \
    --no_continue_prompt
```

Action file format (`actions.json`):
```json
[
  {"keyboard": [1, 0, 0, 0], "mouse": [0.1, 0.0]},
  {"keyboard": [0, 0, 1, 0], "mouse": [0.0, -0.1]}
]
```

For Temple Run, omit `mouse` and use a 7-dim keyboard vector.

## Tips

- Upward camera movement may cause brief rendering glitches (black screens). Adjust movement slightly or change direction to resolve.

## Acknowledgements

- [Diffusers](https://github.com/huggingface/diffusers) for their diffusion model framework
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) for the base model
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for the action control module idea
- [MineRL](https://github.com/minerllabs/minerl) for the gym framework
- [Video-Pre-Training](https://github.com/openai/Video-Pre-Training) for the Inverse Dynamics Model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{he2025matrix,
  title={Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model},
  author={He, Xianglong and Peng, Chunli and Liu, Zexiang and Wang, Boyang and Zhang, Yifan and Cui, Qi and Kang, Fei and Jiang, Biao and An, Mengyin and Ren, Yangyang and Xu, Baixin and Guo, Hao-Xiang and Gong, Kaixiong and Wu, Cyrus and Li, Wei and Song, Xuchen and Liu, Yang and Li, Eric and Zhou, Yahui},
  journal={arXiv preprint arXiv:2508.13009},
  year={2025}
}
```
