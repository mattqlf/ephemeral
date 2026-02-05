#!/usr/bin/env python
"""
Launcher script for Tensor Parallel Web Server

This script handles the multi-process setup for tensor parallelism,
launching one process per GPU using torchrun.

Usage:
    # 2-GPU tensor parallelism
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 2 \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # 4-GPU tensor parallelism with fast mode
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 4 --fast \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # Custom port
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 2 --port 8888 \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch Tensor Parallel Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 2-GPU setup
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 2 \\
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

    # 4-GPU with fast mode (2 denoising steps)
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 4 --fast \\
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
        """
    )

    # Launcher arguments
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for tensor parallelism (default: 2)")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Port for distributed communication (default: 29500)")

    # Server arguments (passed through to apps.servers.web_server_tensor_parallel)
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml",
                        help="Path to inference config")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0",
                        help="Path to pretrained model directory")
    parser.add_argument("--max_num_output_frames", type=int, default=360,
                        help="Maximum output frames (default: 360)")
    parser.add_argument("--max_blocks", type=int, default=999999,
                        help="Maximum blocks to generate (default: unlimited)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (default: 8080)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast mode (2 denoising steps instead of 3)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup (kernels will compile lazily on first inference)")

    args = parser.parse_args()

    # Validate GPU count
    import torch
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Error: Requested {args.num_gpus} GPUs but only {available_gpus} available")
        sys.exit(1)

    # Validate that num_heads is divisible by num_gpus
    # Model has 12 heads, so valid world sizes are 1, 2, 3, 4, 6, 12
    valid_world_sizes = [1, 2, 3, 4, 6, 12]
    if args.num_gpus not in valid_world_sizes:
        print(f"Error: num_gpus must be a divisor of 12 (attention heads)")
        print(f"Valid values: {valid_world_sizes}")
        sys.exit(1)

    # Also validate ffn_dim (8960) divisibility
    # 8960 = 2^7 * 5 * 7 = 128 * 70
    # Divisors: 1, 2, 4, 5, 7, 8, 10, 14, 16, 20, 28, 32, 35, 40, 56, 64, 70, 80...
    if 8960 % args.num_gpus != 0:
        print(f"Error: num_gpus ({args.num_gpus}) must divide ffn_dim (8960)")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Tensor Parallel Launcher")
    print(f"=" * 60)
    print(f"GPUs: {args.num_gpus}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Port: {args.port}")
    print(f"Fast mode: {args.fast}")
    print(f"Skip warmup: {args.no_warmup}")
    print(f"=" * 60)

    # Build torchrun command via uv
    cmd = [
        "uv", "run", "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={args.master_port}",
        "--module",
        "apps.servers.web_server_tensor_parallel",
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

    # Run the command
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
