"""
Tensor Parallelism Utilities for Multi-GPU Inference

This module provides core primitives for tensor parallel inference:
- Process group management
- Column-parallel and row-parallel linear layers
- All-reduce and all-gather communication operations
- Utility functions for weight sharding

Tensor Parallelism splits model parameters across GPUs:
- Attention: Q, K, V projections split column-wise (by heads)
- Attention: Output projection split row-wise
- FFN: First linear split column-wise, second split row-wise

Usage:
    # Initialize process group
    init_tensor_parallel(world_size=2)

    # Create parallel layers
    col_linear = ColumnParallelLinear(in_features=1536, out_features=1536)
    row_linear = RowParallelLinear(in_features=1536, out_features=1536)
"""

import os
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import custom_fwd, custom_bwd


# Global tensor parallel state
_TENSOR_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_TENSOR_PARALLEL_WORLD_SIZE: int = 1
_TENSOR_PARALLEL_RANK: int = 0


def init_tensor_parallel(
    world_size: int,
    rank: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500"
) -> dist.ProcessGroup:
    """
    Initialize tensor parallel process group.

    Args:
        world_size: Number of GPUs for tensor parallelism
        rank: Current GPU rank (0 to world_size-1)
        backend: Distributed backend (nccl recommended for GPU)
        master_addr: Master node address
        master_port: Master node port

    Returns:
        The tensor parallel process group
    """
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_WORLD_SIZE, _TENSOR_PARALLEL_RANK

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank
        )

    # Create tensor parallel group with all ranks
    ranks = list(range(world_size))
    _TENSOR_PARALLEL_GROUP = dist.new_group(ranks)
    _TENSOR_PARALLEL_WORLD_SIZE = world_size
    _TENSOR_PARALLEL_RANK = rank

    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor parallel process group."""
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_world_size() -> int:
    """Get tensor parallel world size."""
    return _TENSOR_PARALLEL_WORLD_SIZE


def get_tensor_parallel_rank() -> int:
    """Get current tensor parallel rank."""
    return _TENSOR_PARALLEL_RANK


def set_tensor_parallel_state(world_size: int, rank: int, group: Optional[dist.ProcessGroup] = None):
    """
    Manually set tensor parallel state (for testing or custom initialization).
    """
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_WORLD_SIZE, _TENSOR_PARALLEL_RANK
    _TENSOR_PARALLEL_WORLD_SIZE = world_size
    _TENSOR_PARALLEL_RANK = rank
    _TENSOR_PARALLEL_GROUP = group


def is_tensor_parallel_initialized() -> bool:
    """Check if tensor parallelism is initialized."""
    return _TENSOR_PARALLEL_WORLD_SIZE > 1


# ---------------------------------------------------------------------------
# Communication Operations
# ---------------------------------------------------------------------------

class _AllReduceFunc(torch.autograd.Function):
    """All-reduce the input tensor across tensor parallel group."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_: Tensor) -> Tensor:
        if get_tensor_parallel_world_size() == 1:
            return input_

        output = input_.clone()
        dist.all_reduce(output, group=get_tensor_parallel_group())
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tensor:
        # Gradient is also all-reduced in backward pass
        if get_tensor_parallel_world_size() == 1:
            return grad_output

        output = grad_output.clone()
        dist.all_reduce(output, group=get_tensor_parallel_group())
        return output


def all_reduce(input_: Tensor) -> Tensor:
    """All-reduce tensor across tensor parallel group."""
    return _AllReduceFunc.apply(input_)


class _AllGatherFunc(torch.autograd.Function):
    """All-gather the input tensor across tensor parallel group along dim."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        world_size = get_tensor_parallel_world_size()

        if world_size == 1:
            return input_

        # Gather tensors from all ranks
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=get_tensor_parallel_group())

        # Concatenate along specified dimension
        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        dim = ctx.dim
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        if world_size == 1:
            return grad_output, None

        # Split gradient and return local chunk
        grad_chunks = torch.chunk(grad_output, world_size, dim=dim)
        return grad_chunks[rank].contiguous(), None


def all_gather(input_: Tensor, dim: int = -1) -> Tensor:
    """All-gather tensor across tensor parallel group along dimension."""
    return _AllGatherFunc.apply(input_, dim)


class _ReduceScatterFunc(torch.autograd.Function):
    """Reduce-scatter the input tensor across tensor parallel group along dim."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        if world_size == 1:
            return input_

        # Split input along dimension
        input_chunks = list(torch.chunk(input_, world_size, dim=dim))

        # Allocate output tensor
        output = torch.empty_like(input_chunks[0])

        # Reduce-scatter
        dist.reduce_scatter(output, input_chunks, group=get_tensor_parallel_group())
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        dim = ctx.dim
        world_size = get_tensor_parallel_world_size()

        if world_size == 1:
            return grad_output, None

        # All-gather gradients
        grad_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_list, grad_output, group=get_tensor_parallel_group())

        return torch.cat(grad_list, dim=dim), None


def reduce_scatter(input_: Tensor, dim: int = -1) -> Tensor:
    """Reduce-scatter tensor across tensor parallel group along dimension."""
    return _ReduceScatterFunc.apply(input_, dim)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass input through, all-reduce on backward."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_: Tensor) -> Tensor:
        return input_

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return all_reduce(grad_output)


def copy_to_tensor_parallel_region(input_: Tensor) -> Tensor:
    """Identity forward, all-reduce backward."""
    return _CopyToModelParallelRegion.apply(input_)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce forward, identity backward."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_: Tensor) -> Tensor:
        return all_reduce(input_)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def reduce_from_tensor_parallel_region(input_: Tensor) -> Tensor:
    """All-reduce forward, identity backward."""
    return _ReduceFromModelParallelRegion.apply(input_)


# ---------------------------------------------------------------------------
# Parallel Linear Layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix is split along the output dimension (columns).
    Y = XA^T, where A is split column-wise across GPUs.
    Each GPU computes a portion of the output.

    Args:
        in_features: Input dimension (not split)
        out_features: Output dimension (split across GPUs)
        bias: Whether to include bias
        gather_output: If True, all-gather output across GPUs
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        world_size = get_tensor_parallel_world_size()
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"

        self.out_features_per_partition = out_features // world_size

        # Initialize weight and bias for this partition
        self.weight = nn.Parameter(torch.empty(
            self.out_features_per_partition,
            in_features,
            dtype=dtype,
            device=device
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.out_features_per_partition,
                dtype=dtype,
                device=device
            ))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: Tensor) -> Tensor:
        # Copy input for backward pass gradient all-reduce
        input_parallel = copy_to_tensor_parallel_region(input_)

        # Local matmul
        output = torch.nn.functional.linear(input_parallel, self.weight, self.bias)

        # Optionally gather output
        if self.gather_output:
            output = all_gather(output, dim=-1)

        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, gather_output: bool = False) -> "ColumnParallelLinear":
        """Create ColumnParallelLinear from existing nn.Linear by sharding weights."""
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        in_features = linear.in_features
        out_features = linear.out_features

        parallel_linear = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias is not None,
            gather_output=gather_output,
            dtype=linear.weight.dtype,
            device=linear.weight.device
        )

        # Shard weight along output dimension (dim 0)
        out_per_partition = out_features // world_size
        start_idx = rank * out_per_partition
        end_idx = start_idx + out_per_partition

        with torch.no_grad():
            parallel_linear.weight.copy_(linear.weight[start_idx:end_idx, :])
            if linear.bias is not None:
                parallel_linear.bias.copy_(linear.bias[start_idx:end_idx])

        return parallel_linear


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The weight matrix is split along the input dimension (rows).
    Y = XA^T, where A is split row-wise across GPUs.
    Input must already be split, output is all-reduced.

    Args:
        in_features: Input dimension (split across GPUs)
        out_features: Output dimension (not split)
        bias: Whether to include bias
        input_is_parallel: If True, input is already split across GPUs
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        world_size = get_tensor_parallel_world_size()
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"

        self.in_features_per_partition = in_features // world_size

        # Initialize weight and bias for this partition
        self.weight = nn.Parameter(torch.empty(
            out_features,
            self.in_features_per_partition,
            dtype=dtype,
            device=device
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features,
                dtype=dtype,
                device=device
            ))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: Tensor) -> Tensor:
        # If input is not already parallel, split it
        if not self.input_is_parallel:
            world_size = get_tensor_parallel_world_size()
            rank = get_tensor_parallel_rank()
            input_chunks = torch.chunk(input_, world_size, dim=-1)
            input_ = input_chunks[rank]

        # Local matmul (no bias yet)
        output = torch.nn.functional.linear(input_, self.weight, None)

        # All-reduce across GPUs
        output = reduce_from_tensor_parallel_region(output)

        # Add bias after all-reduce (only on rank 0 or replicated)
        if self.bias is not None:
            output = output + self.bias

        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_is_parallel: bool = True) -> "RowParallelLinear":
        """Create RowParallelLinear from existing nn.Linear by sharding weights."""
        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        in_features = linear.in_features
        out_features = linear.out_features

        parallel_linear = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias is not None,
            input_is_parallel=input_is_parallel,
            dtype=linear.weight.dtype,
            device=linear.weight.device
        )

        # Shard weight along input dimension (dim 1)
        in_per_partition = in_features // world_size
        start_idx = rank * in_per_partition
        end_idx = start_idx + in_per_partition

        with torch.no_grad():
            parallel_linear.weight.copy_(linear.weight[:, start_idx:end_idx])
            # Bias is not sharded - replicated on all ranks
            if linear.bias is not None:
                parallel_linear.bias.copy_(linear.bias)

        return parallel_linear


# ---------------------------------------------------------------------------
# Parallel Embedding and Other Utilities
# ---------------------------------------------------------------------------

def shard_qkv_linear(
    q_linear: nn.Linear,
    k_linear: nn.Linear,
    v_linear: nn.Linear
) -> Tuple[ColumnParallelLinear, ColumnParallelLinear, ColumnParallelLinear]:
    """
    Convert separate Q, K, V linear layers to column-parallel versions.

    Returns:
        Tuple of (q_parallel, k_parallel, v_parallel)
    """
    q_parallel = ColumnParallelLinear.from_linear(q_linear, gather_output=False)
    k_parallel = ColumnParallelLinear.from_linear(k_linear, gather_output=False)
    v_parallel = ColumnParallelLinear.from_linear(v_linear, gather_output=False)
    return q_parallel, k_parallel, v_parallel


def shard_attention_output(o_linear: nn.Linear) -> RowParallelLinear:
    """
    Convert attention output linear to row-parallel version.
    """
    return RowParallelLinear.from_linear(o_linear, input_is_parallel=True)


def shard_ffn(
    ffn: nn.Sequential
) -> nn.Sequential:
    """
    Convert FFN (Linear -> GELU -> Linear) to tensor-parallel version.

    First linear is column-parallel, second is row-parallel.
    """
    assert len(ffn) == 3, "FFN must be Linear -> activation -> Linear"
    assert isinstance(ffn[0], nn.Linear)
    assert isinstance(ffn[2], nn.Linear)

    first_linear = ColumnParallelLinear.from_linear(ffn[0], gather_output=False)
    activation = ffn[1]  # Keep activation as-is
    second_linear = RowParallelLinear.from_linear(ffn[2], input_is_parallel=True)

    return nn.Sequential(first_linear, activation, second_linear)


# ---------------------------------------------------------------------------
# Weight Loading Utilities
# ---------------------------------------------------------------------------

def load_tensor_parallel_weights(
    model: nn.Module,
    checkpoint: dict,
    strict: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load checkpoint weights into a tensor-parallel model.

    This function handles the weight sharding automatically based on
    which layers are ColumnParallelLinear or RowParallelLinear.

    Args:
        model: The tensor-parallel model
        checkpoint: State dict with full (non-sharded) weights
        strict: Whether to require all keys to match

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    world_size = get_tensor_parallel_world_size()
    rank = get_tensor_parallel_rank()

    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = list(checkpoint.keys())

    for name, param in model.named_parameters():
        if name not in checkpoint:
            missing_keys.append(name)
            continue

        unexpected_keys.remove(name)
        src_tensor = checkpoint[name]

        # Check if this is a sharded parameter
        # Find the parent module to determine sharding strategy
        module = model
        parts = name.replace('.weight', '').replace('.bias', '').split('.')
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        final_part = parts[-1]
        if hasattr(module, final_part):
            layer = getattr(module, final_part)
        else:
            layer = module

        if isinstance(layer, ColumnParallelLinear):
            # Shard along output dimension (dim 0 for weight, dim 0 for bias)
            if 'weight' in name:
                out_per_partition = src_tensor.shape[0] // world_size
                start = rank * out_per_partition
                end = start + out_per_partition
                src_tensor = src_tensor[start:end, :]
            elif 'bias' in name:
                out_per_partition = src_tensor.shape[0] // world_size
                start = rank * out_per_partition
                end = start + out_per_partition
                src_tensor = src_tensor[start:end]

        elif isinstance(layer, RowParallelLinear):
            # Shard along input dimension (dim 1 for weight, bias is replicated)
            if 'weight' in name:
                in_per_partition = src_tensor.shape[1] // world_size
                start = rank * in_per_partition
                end = start + in_per_partition
                src_tensor = src_tensor[:, start:end]
            # Bias is replicated, no sharding needed

        # Copy to parameter
        with torch.no_grad():
            param.copy_(src_tensor)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Error loading state dict:\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    return missing_keys, unexpected_keys
