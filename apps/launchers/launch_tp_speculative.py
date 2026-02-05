#!/usr/bin/env python
"""
Launcher script for Tensor Parallel + Speculative Hybrid Web Server

This script handles the multi-process setup for the hybrid TP+Speculative system,
launching processes across GPUs using torchrun.

GPU Layout:
- First half: Primary TP group (generates current block)
- Second half: Speculative TP group (pre-generates next block)

Examples:
    # 4 GPUs: 2 for primary TP, 2 for speculative TP
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 4 \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # 6 GPUs: 3 for primary TP, 3 for speculative TP
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 6 --fast \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # 8 GPUs: 4 for primary TP, 4 for speculative TP
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 8 --port 8888 --fast \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch TP+Speculative Hybrid Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 4 GPUs (2 TP per group)
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 4 \\
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # 6 GPUs (3 TP per group) with fast mode
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 6 --fast \\
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
        """
    )

    # Launcher arguments
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Total number of GPUs (must be even, >= 4). Half for primary TP, half for speculative TP.")
    parser.add_argument("--master_port", type=str, default="29501",
                        help="Port for distributed communication (default: 29501)")

    # Server arguments
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--max_num_output_frames", type=int, default=360)
    parser.add_argument("--max_blocks", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fast", action="store_true",
                        help="Use fast mode (2 denoising steps)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup")

    args = parser.parse_args()

    # Validate GPU count
    import torch
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Error: Requested {args.num_gpus} GPUs but only {available_gpus} available")
        sys.exit(1)

    if args.num_gpus < 4:
        print(f"Error: TP+Speculative requires at least 4 GPUs")
        print(f"  - First half: Primary tensor parallel group")
        print(f"  - Second half: Speculative tensor parallel group")
        print(f"Got: {args.num_gpus} GPUs")
        sys.exit(1)

    if args.num_gpus % 2 != 0:
        print(f"Error: TP+Speculative requires an even number of GPUs")
        print(f"Got: {args.num_gpus} GPUs")
        sys.exit(1)

    tp_size = args.num_gpus // 2

    # Validate TP size against model constraints
    # Model has 12 heads, so valid TP sizes are 1, 2, 3, 4, 6, 12
    valid_tp_sizes = [1, 2, 3, 4, 6, 12]
    if tp_size not in valid_tp_sizes:
        print(f"Error: TP size must be a divisor of 12 (attention heads)")
        print(f"Valid TP sizes: {valid_tp_sizes}")
        print(f"Got TP size: {tp_size} (from {args.num_gpus} total GPUs)")
        print(f"\nTry one of these total GPU counts: 4, 6, 8, 12, 24")
        sys.exit(1)

    # Validate ffn_dim (8960) divisibility
    if 8960 % tp_size != 0:
        print(f"Error: TP size ({tp_size}) must divide ffn_dim (8960)")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"TP+Speculative Launcher")
    print(f"{'='*60}")
    print(f"Total GPUs: {args.num_gpus}")
    print(f"TP size per group: {tp_size}")
    print(f"Primary group: GPUs 0-{tp_size-1}")
    print(f"Speculative group: GPUs {tp_size}-{args.num_gpus-1}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Port: {args.port}")
    print(f"Fast mode: {args.fast}")
    print(f"Skip warmup: {args.no_warmup}")
    print(f"{'='*60}")

    # Build torchrun command
    cmd = [
        "uv", "run", "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={args.master_port}",
        "--module",
        "apps.servers.web_server_tp_speculative",
        f"--config_path={args.config_path}",
        f"--checkpoint_path={args.checkpoint_path}",
        f"--pretrained_model_path={args.pretrained_model_path}",
        f"--max_num_output_frames={args.max_num_output_frames}",
        f"--max_blocks={args.max_blocks}",
        f"--seed={args.seed}",
        f"--host={args.host}",
        f"--port={args.port}",
    ]

    if args.fast:
        cmd.append("--fast")

    if args.no_warmup:
        cmd.append("--no-warmup")

    print(f"\nExecuting: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Process exited with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
