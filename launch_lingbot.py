#!/usr/bin/env python
"""
Launcher script for LingBot-World Web Server

Launches the LingBot-World web server with multi-GPU support via torchrun.
Uses FSDP + Ulysses sequence parallelism for distributed inference.

Usage:
    # 4-GPU with sequence parallelism (recommended)
    uv run python launch_lingbot.py --num_gpus 4 \
        --ckpt_dir lingbot-world-base-cam

    # Single GPU (slower, no parallelism)
    uv run python launch_lingbot.py --num_gpus 1 \
        --ckpt_dir lingbot-world-base-cam

    # Custom port and resolution
    uv run python launch_lingbot.py --num_gpus 4 \
        --ckpt_dir lingbot-world-base-cam \
        --port 8889 --size 720*1280
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch LingBot-World Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 4-GPU with sequence parallelism
    uv run python launch_lingbot.py --num_gpus 4 \\
        --ckpt_dir lingbot-world-base-cam

    # Single GPU
    uv run python launch_lingbot.py --num_gpus 1 \\
        --ckpt_dir lingbot-world-base-cam
        """
    )

    # Launcher arguments
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs (default: 4)")
    parser.add_argument("--master_port", type=str, default="29501",
                        help="Port for distributed communication (default: 29501)")

    # Server arguments (passed through)
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to LingBot-World checkpoint directory")
    parser.add_argument("--size", type=str, default="480*832",
                        help="Output resolution (default: 480*832)")
    parser.add_argument("--sample_shift", type=float, default=10.0,
                        help="Sampling shift factor (default: 10.0)")
    parser.add_argument("--t5_fsdp", action="store_true", default=False,
                        help="Enable FSDP for T5")
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="Place T5 on CPU to save GPU memory")
    parser.add_argument("--dit_fsdp", action="store_true", default=False,
                        help="Enable FSDP for DiT models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8889,
                        help="Server port (default: 8889)")

    args = parser.parse_args()

    # Validate GPU count
    if args.num_gpus > 1:
        import torch
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"Error: Requested {args.num_gpus} GPUs but only {available_gpus} available")
            sys.exit(1)

    # Validate attention head divisibility for sequence parallelism
    # LingBot-World has 40 heads
    if args.num_gpus > 1 and 40 % args.num_gpus != 0:
        print(f"Error: num_gpus ({args.num_gpus}) must divide num_heads (40)")
        print(f"Valid values: 1, 2, 4, 5, 8, 10, 20, 40")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"LingBot-World Launcher")
    print(f"{'=' * 60}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Resolution: {args.size}")
    print(f"Port: {args.port}")
    print(f"T5 FSDP: {args.t5_fsdp}")
    print(f"T5 CPU: {args.t5_cpu}")
    print(f"DiT FSDP: {args.dit_fsdp}")
    print(f"{'=' * 60}")

    if args.num_gpus > 1:
        # Multi-GPU: use torchrun
        cmd = [
            "uv", "run", "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            f"--master_port={args.master_port}",
            "web_server_lingbot.py",
        ]
    else:
        # Single GPU: direct execution
        cmd = ["uv", "run", "python", "web_server_lingbot.py"]

    # Append server arguments
    cmd.extend([
        f"--ckpt_dir={args.ckpt_dir}",
        f"--size={args.size}",
        f"--sample_shift={args.sample_shift}",
        f"--seed={args.seed}",
        f"--host={args.host}",
        f"--port={args.port}",
    ])

    if args.num_gpus > 1:
        cmd.append(f"--ulysses_size={args.num_gpus}")

    if args.t5_fsdp:
        cmd.append("--t5_fsdp")

    if args.t5_cpu:
        cmd.append("--t5_cpu")

    if args.dit_fsdp:
        cmd.append("--dit_fsdp")

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
