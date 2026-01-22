"""
Tensor Parallel version of CausalWanModel

This module provides tensor-parallel versions of the causal diffusion model
components for multi-GPU inference:
- TPCausalWanSelfAttention: Tensor-parallel self-attention
- TPCausalWanAttentionBlock: Tensor-parallel transformer block
- TPCausalWanModel: Full tensor-parallel model

Tensor parallelism splits:
- Attention heads across GPUs (Q, K, V projections)
- FFN intermediate dimension across GPUs

Usage:
    from wan.distributed.tensor_parallel import init_tensor_parallel
    from wan.modules.causal_model_tp import TPCausalWanModel

    # Initialize TP
    init_tensor_parallel(world_size=2, rank=local_rank)

    # Create model
    model = TPCausalWanModel(...)

    # Or convert from existing model
    model = TPCausalWanModel.from_pretrained(original_model)
"""

import math
import copy
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
from einops import rearrange

from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    MLPProj,
    sinusoidal_embedding_1d,
    rope_params,
    flash_attention,
)
from wan.modules.causal_model import (
    causal_rope_apply,
    CausalHead,
)
from wan.distributed.tensor_parallel import (
    get_tensor_parallel_world_size,
    get_tensor_parallel_rank,
    all_reduce,
    all_gather,
    reduce_from_tensor_parallel_region,
    copy_to_tensor_parallel_region,
    ColumnParallelLinear,
    RowParallelLinear,
)

from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

# Compile flex_attention
print("[compile] Compiling flex_attention in causal_model_tp.py...")
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
print("[compile] flex_attention (causal_model_tp) compiled.")


class TPWanRMSNorm(nn.Module):
    """
    RMSNorm that operates on tensor-parallel sharded tensors.

    When used after column-parallel linear (sharded heads), this normalizes
    each partition independently, which is mathematically correct for RMSNorm
    applied per-head.
    """

    def __init__(self, dim: int, eps: float = 1e-6, world_size: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.world_size = world_size

        # Weight is sharded - each rank has dim/world_size
        self.dim_per_partition = dim // world_size
        self.weight = nn.Parameter(torch.ones(self.dim_per_partition))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Shape [..., dim_per_partition]
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class TPCausalWanSelfAttention(nn.Module):
    """
    Tensor-parallel version of CausalWanSelfAttention.

    Parallelism strategy:
    - Q, K, V projections: Column-parallel (split output dim by heads)
    - O projection: Row-parallel (all-reduce output)
    - Each GPU handles num_heads/world_size attention heads
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        assert dim % num_heads == 0
        assert num_heads % world_size == 0, \
            f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Tensor parallel dimensions
        self.num_heads_per_partition = num_heads // world_size
        self.dim_per_partition = self.num_heads_per_partition * self.head_dim

        self.max_attention_size = 15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880

        # Column-parallel Q, K, V (each rank gets a subset of heads)
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)

        # Row-parallel output projection (all-reduces result)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)

        # QK norms - applied per-head, so we use sharded version
        if qk_norm:
            self.norm_q = TPWanRMSNorm(dim, eps=eps, world_size=world_size)
            self.norm_k = TPWanRMSNorm(dim, eps=eps, world_size=world_size)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

    def forward(
        self,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        block_mask: Optional[BlockMask],
        kv_cache: Optional[dict] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            x: Shape [B, L, C]
            seq_lens: Shape [B]
            grid_sizes: Shape [B, 3] containing (F, H, W) or shape [3]
            freqs: RoPE frequencies
            block_mask: Attention mask for flex_attention
            kv_cache: Optional KV cache for inference
            current_start: Starting position for KV cache
            cache_start: Cache start position
        """
        b, s = x.shape[:2]
        n = self.num_heads_per_partition
        d = self.head_dim

        if cache_start is None:
            cache_start = current_start

        # Compute Q, K, V (column-parallel, each rank has subset of heads)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            # Training / full sequence inference
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            # Padding for flex_attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat([
                roped_query,
                torch.zeros([b, padded_length, n, d], device=q.device, dtype=v.dtype)
            ], dim=1)
            padded_roped_key = torch.cat([
                roped_key,
                torch.zeros([b, padded_length, n, d], device=k.device, dtype=v.dtype)
            ], dim=1)
            padded_v = torch.cat([
                v,
                torch.zeros([b, padded_length, n, d], device=v.device, dtype=v.dtype)
            ], dim=1)

            # Attention with flex_attention
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)

        else:
            # Cached inference
            assert grid_sizes.ndim == 1
            frame_seqlen = math.prod(grid_sizes[1:]).item()
            current_start_frame = current_start // frame_seqlen

            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen

            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            # Handle KV cache eviction
            if (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):

                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            # Attention with cached KV
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection (row-parallel, includes all-reduce)
        x = x.flatten(2)
        x = self.o(x)
        return x


class TPWanI2VCrossAttention(nn.Module):
    """
    Tensor-parallel I2V cross-attention.

    Same parallelism as self-attention:
    - Q, K, V: Column-parallel
    - O: Row-parallel
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        world_size = get_tensor_parallel_world_size()

        assert dim % num_heads == 0
        assert num_heads % world_size == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_heads_per_partition = num_heads // world_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Column-parallel Q, K, V
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)

        # Row-parallel O
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)

        # QK norms
        if qk_norm:
            self.norm_q = TPWanRMSNorm(dim, eps=eps, world_size=world_size)
            self.norm_k = TPWanRMSNorm(dim, eps=eps, world_size=world_size)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        crossattn_cache: Optional[dict] = None,
    ) -> Tensor:
        """
        Args:
            x: Query tensor [B, L1, C]
            context: Context tensor [B, L2, C]
            crossattn_cache: Optional cache for K, V
        """
        b = x.size(0)
        n = self.num_heads_per_partition
        d = self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=None)

        x = x.flatten(2)
        x = self.o(x)
        return x


class TPCausalWanAttentionBlock(nn.Module):
    """
    Tensor-parallel transformer block.

    Parallelism:
    - Self-attention: Q, K, V column-parallel, O row-parallel
    - Cross-attention: Same pattern
    - FFN: First linear column-parallel, second row-parallel
    - Action module: Not parallelized (relatively small)
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        action_config: dict = {},
        block_idx: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()

        world_size = get_tensor_parallel_world_size()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Action module (not parallelized - runs same on all ranks)
        if len(action_config) != 0 and block_idx in action_config['blocks']:
            from wan.modules.action_module import ActionModule
            self.action_model = ActionModule(**action_config, local_attn_size=local_attn_size)
        else:
            self.action_model = None

        # Layer norms (replicated)
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # Tensor-parallel self-attention
        self.self_attn = TPCausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps)

        # Tensor-parallel cross-attention
        self.cross_attn = TPWanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # Tensor-parallel FFN
        # First linear: column-parallel (split ffn_dim)
        # Second linear: row-parallel (all-reduce output)
        assert ffn_dim % world_size == 0, \
            f"ffn_dim ({ffn_dim}) must be divisible by world_size ({world_size})"

        self.ffn_fc1 = ColumnParallelLinear(dim, ffn_dim, bias=True, gather_output=False)
        self.ffn_act = nn.GELU(approximate='tanh')
        self.ffn_fc2 = RowParallelLinear(ffn_dim, dim, bias=True, input_is_parallel=True)

        # Modulation parameters (replicated)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        context: Tensor,
        block_mask: Optional[BlockMask],
        block_mask_mouse: Optional[BlockMask],
        block_mask_keyboard: Optional[BlockMask],
        num_frame_per_block: int = 3,
        use_rope_keyboard: bool = False,
        mouse_cond: Optional[Tensor] = None,
        keyboard_cond: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        kv_cache_mouse: Optional[dict] = None,
        kv_cache_keyboard: Optional[dict] = None,
        crossattn_cache: Optional[dict] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the block."""
        assert e.ndim == 4
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x.to(context.dtype)), context, crossattn_cache=crossattn_cache)

        # Action module (same computation on all ranks)
        if self.action_model is not None:
            assert mouse_cond is not None or keyboard_cond is not None
            x = self.action_model(
                x.to(context.dtype), grid_sizes[0], grid_sizes[1], grid_sizes[2],
                mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard,
                is_causal=True, kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard,
                start_frame=current_start // math.prod(grid_sizes[1:]).item(),
                use_rope_keyboard=use_rope_keyboard, num_frame_per_block=num_frame_per_block
            )

        # FFN with tensor parallelism
        ffn_input = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        y = self.ffn_fc2(self.ffn_act(self.ffn_fc1(ffn_input)))

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x


class TPCausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    """
    Tensor-parallel version of CausalWanModel.

    This model distributes computation across multiple GPUs using tensor parallelism:
    - Attention heads split across GPUs
    - FFN intermediate dimension split across GPUs
    - Communication via all-reduce operations
    """

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['TPCausalWanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type: str = 'i2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        action_config: dict = {},
        eps: float = 1e-6,
    ):
        super().__init__()

        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()

        # Validate parallelism constraints
        assert num_heads % world_size == 0, \
            f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
        assert ffn_dim % world_size == 0, \
            f"ffn_dim ({ffn_dim}) must be divisible by world_size ({world_size})"
        assert dim % world_size == 0, \
            f"dim ({dim}) must be divisible by world_size ({world_size})"

        assert model_type == 'i2v', "Only i2v model type supported"

        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings (replicated on all ranks)
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # I2V image embedding (replicated)
        self.img_emb = MLPProj(1280, dim)

        # Tensor-parallel transformer blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            TPCausalWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm,
                action_config=action_config, eps=eps, block_idx=idx
            )
            for idx in range(num_layers)
        ])

        # Output head (replicated)
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.init_weights()

        self.gradient_checkpointing = False
        self.block_mask = None
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True
        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device,
        num_frames: int = 9,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1
    ) -> BlockMask:
        """Create block-wise causal attention mask."""
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(
            start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device
        )

        world_size = get_tensor_parallel_world_size()
        rank = get_tensor_parallel_rank()
        if rank == 0:
            print(f"[TP] Cached block-wise causal mask with block size {num_frame_per_block} frames")

        return block_mask

    def _forward_inference(
        self,
        x: Tensor,
        t: Tensor,
        visual_context: Tensor,
        cond_concat: Tensor,
        mouse_cond: Optional[Tensor] = None,
        keyboard_cond: Optional[Tensor] = None,
        kv_cache: Optional[List[dict]] = None,
        kv_cache_mouse: Optional[List[dict]] = None,
        kv_cache_keyboard: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: int = 0,
        cache_start: int = 0,
    ) -> Tensor:
        """Forward pass for inference with KV caching."""
        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1)

        # Embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        assert seq_lens[0] <= 15 * 1 * 880

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context = self.img_emb(visual_context)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            context_lens=None,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block,
        )

        for block_index, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "kv_cache_mouse": kv_cache_mouse[block_index],
                "kv_cache_keyboard": kv_cache_keyboard[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
            })
            x = block(x, **kwargs)

        # Output head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            raise NotImplementedError("Training not supported for TP model")

    def unpatchify(self, x: Tensor, grid_sizes: Tensor) -> Tensor:
        """Reconstruct video from patches."""
        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return x

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
                # Parallel layers init their own weights
                pass

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained_model(
        cls,
        model: "CausalWanModel",
        world_size: int,
        rank: int,
    ) -> "TPCausalWanModel":
        """
        Create a tensor-parallel model from a pretrained CausalWanModel.

        This method handles weight sharding automatically.
        """
        from wan.distributed.tensor_parallel import set_tensor_parallel_state

        # Set TP state
        set_tensor_parallel_state(world_size, rank)

        # Get config from original model
        config = model.config

        # Create TP model
        tp_model = cls(
            model_type=config.get('model_type', 'i2v'),
            patch_size=config.get('patch_size', (1, 2, 2)),
            text_len=config.get('text_len', 512),
            in_dim=config.get('in_dim', 36),
            dim=config.get('dim', 1536),
            ffn_dim=config.get('ffn_dim', 8960),
            freq_dim=config.get('freq_dim', 256),
            text_dim=config.get('text_dim', 4096),
            out_dim=config.get('out_dim', 16),
            num_heads=config.get('num_heads', 12),
            num_layers=config.get('num_layers', 30),
            local_attn_size=config.get('local_attn_size', -1),
            sink_size=config.get('sink_size', 0),
            qk_norm=config.get('qk_norm', True),
            cross_attn_norm=config.get('cross_attn_norm', True),
            action_config=config.get('action_config', {}),
            eps=config.get('eps', 1e-6),
        )

        # Copy and shard weights
        tp_model._load_weights_from_model(model, world_size, rank)

        return tp_model

    def _load_weights_from_model(self, model: "CausalWanModel", world_size: int, rank: int):
        """Load and shard weights from a non-parallel model."""
        with torch.no_grad():
            # Copy non-parallel weights directly
            self.patch_embedding.load_state_dict(model.patch_embedding.state_dict())
            self.time_embedding.load_state_dict(model.time_embedding.state_dict())
            self.time_projection.load_state_dict(model.time_projection.state_dict())
            self.img_emb.load_state_dict(model.img_emb.state_dict())
            self.head.load_state_dict(model.head.state_dict())
            self.freqs = model.freqs.clone()

            # Copy block mask settings
            self.block_mask = model.block_mask
            self.block_mask_keyboard = model.block_mask_keyboard
            self.block_mask_mouse = model.block_mask_mouse
            self.use_rope_keyboard = model.use_rope_keyboard
            self.num_frame_per_block = model.num_frame_per_block

            # Shard transformer blocks
            for tp_block, orig_block in zip(self.blocks, model.blocks):
                self._shard_block_weights(tp_block, orig_block, world_size, rank)

    def _shard_block_weights(
        self,
        tp_block: TPCausalWanAttentionBlock,
        orig_block: "CausalWanAttentionBlock",
        world_size: int,
        rank: int
    ):
        """Shard weights for a single transformer block."""
        dim = self.dim
        ffn_dim = self.ffn_dim
        num_heads = self.num_heads
        head_dim = dim // num_heads

        heads_per_partition = num_heads // world_size
        dim_per_partition = heads_per_partition * head_dim
        ffn_per_partition = ffn_dim // world_size

        # Compute partition indices
        head_start = rank * heads_per_partition
        head_end = head_start + heads_per_partition
        dim_start = rank * dim_per_partition
        dim_end = dim_start + dim_per_partition
        ffn_start = rank * ffn_per_partition
        ffn_end = ffn_start + ffn_per_partition

        # Copy layer norms (replicated)
        tp_block.norm1.load_state_dict(orig_block.norm1.state_dict())
        tp_block.norm2.load_state_dict(orig_block.norm2.state_dict())
        if hasattr(orig_block.norm3, 'weight'):
            tp_block.norm3.load_state_dict(orig_block.norm3.state_dict())

        # Copy modulation (replicated)
        tp_block.modulation.copy_(orig_block.modulation)

        # Shard self-attention Q, K, V (column-parallel: split output dim)
        tp_block.self_attn.q.weight.copy_(orig_block.self_attn.q.weight[dim_start:dim_end, :])
        tp_block.self_attn.k.weight.copy_(orig_block.self_attn.k.weight[dim_start:dim_end, :])
        tp_block.self_attn.v.weight.copy_(orig_block.self_attn.v.weight[dim_start:dim_end, :])
        if orig_block.self_attn.q.bias is not None:
            tp_block.self_attn.q.bias.copy_(orig_block.self_attn.q.bias[dim_start:dim_end])
            tp_block.self_attn.k.bias.copy_(orig_block.self_attn.k.bias[dim_start:dim_end])
            tp_block.self_attn.v.bias.copy_(orig_block.self_attn.v.bias[dim_start:dim_end])

        # Shard self-attention O (row-parallel: split input dim)
        tp_block.self_attn.o.weight.copy_(orig_block.self_attn.o.weight[:, dim_start:dim_end])
        if orig_block.self_attn.o.bias is not None:
            tp_block.self_attn.o.bias.copy_(orig_block.self_attn.o.bias)

        # Shard QK norms
        if hasattr(orig_block.self_attn.norm_q, 'weight'):
            tp_block.self_attn.norm_q.weight.copy_(
                orig_block.self_attn.norm_q.weight[dim_start:dim_end])
            tp_block.self_attn.norm_k.weight.copy_(
                orig_block.self_attn.norm_k.weight[dim_start:dim_end])

        # Shard cross-attention
        tp_block.cross_attn.q.weight.copy_(orig_block.cross_attn.q.weight[dim_start:dim_end, :])
        tp_block.cross_attn.k.weight.copy_(orig_block.cross_attn.k.weight[dim_start:dim_end, :])
        tp_block.cross_attn.v.weight.copy_(orig_block.cross_attn.v.weight[dim_start:dim_end, :])
        if orig_block.cross_attn.q.bias is not None:
            tp_block.cross_attn.q.bias.copy_(orig_block.cross_attn.q.bias[dim_start:dim_end])
            tp_block.cross_attn.k.bias.copy_(orig_block.cross_attn.k.bias[dim_start:dim_end])
            tp_block.cross_attn.v.bias.copy_(orig_block.cross_attn.v.bias[dim_start:dim_end])
        tp_block.cross_attn.o.weight.copy_(orig_block.cross_attn.o.weight[:, dim_start:dim_end])
        if orig_block.cross_attn.o.bias is not None:
            tp_block.cross_attn.o.bias.copy_(orig_block.cross_attn.o.bias)

        if hasattr(orig_block.cross_attn.norm_q, 'weight'):
            tp_block.cross_attn.norm_q.weight.copy_(
                orig_block.cross_attn.norm_q.weight[dim_start:dim_end])
            tp_block.cross_attn.norm_k.weight.copy_(
                orig_block.cross_attn.norm_k.weight[dim_start:dim_end])

        # Shard FFN
        # First linear (column-parallel): split output dim
        tp_block.ffn_fc1.weight.copy_(orig_block.ffn[0].weight[ffn_start:ffn_end, :])
        if orig_block.ffn[0].bias is not None:
            tp_block.ffn_fc1.bias.copy_(orig_block.ffn[0].bias[ffn_start:ffn_end])

        # Second linear (row-parallel): split input dim
        tp_block.ffn_fc2.weight.copy_(orig_block.ffn[2].weight[:, ffn_start:ffn_end])
        if orig_block.ffn[2].bias is not None:
            tp_block.ffn_fc2.bias.copy_(orig_block.ffn[2].bias)

        # Copy action module if present (not sharded)
        if tp_block.action_model is not None and orig_block.action_model is not None:
            tp_block.action_model.load_state_dict(orig_block.action_model.state_dict())
