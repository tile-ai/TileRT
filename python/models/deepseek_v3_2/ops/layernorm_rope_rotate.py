"""Layernorm_rope_rotate operation module."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.rotate import rotate_activation
from tilert.models.utils import apply_rotary_emb
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "layernorm_rope_rotate",
    "LayerNormRoPERotate",
    "LayerNormRoPERotateRefWeightsAlias",
    "LayerNormRoPERotateTilertWeightsAlias",
]


def layernorm_rope_rotate(
    input_raw: torch.Tensor,
    cur_pos: torch.Tensor,
    k_cache_raw: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    freqs_cis: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Layernorm_rope_rotate operation.

    Layernorm_rope_rotate the input tensor `input_raw` and stores the result in `k_cache_raw`.

    Args:
        input_raw (torch.Tensor): The input tensor.
        cur_pos (torch.Tensor): The current position tensor.
        k_cache_raw (torch.Tensor): The output tensor where the result will be stored.
        weight (torch.Tensor): The weight tensor.
        bias (torch.Tensor): The bias tensor.
        freqs_cis (torch.Tensor): The frequency tensor.
        profile_logs (torch.Tensor): Tensor for storing profiling logs.

    Returns:
        None
    """
    if input_raw.dtype != torch.bfloat16:
        raise ValueError("input must be a bfloat16 tensor.")
    if cur_pos.dtype != torch.int32:
        raise ValueError("cur_pos must be a int32 tensor.")
    if k_cache_raw.dtype != torch.bfloat16:
        raise ValueError("k_cache must be a bfloat16 tensor.")

    if weight.dtype != torch.float32:
        raise ValueError("weight must be a float32 tensor.")

    if bias.dtype != torch.float32:
        raise ValueError("bias must be a float32 tensor.")

    if freqs_cis.dtype != torch.float32:
        raise ValueError("freqs_cis must be a float32 tensor.")

    batch, seq, dim = input_raw.shape
    if dim != 128:
        raise ValueError("dim must be 128, as we precompute scale inner kernel")
    if batch != 1:
        raise ValueError("batch must be 1 in this version")

    torch.ops.tilert.layernorm_rope_rotate_op(
        input_raw, cur_pos, k_cache_raw, weight, bias, freqs_cis, profile_logs
    )


@dataclass
class LayerNormRoPERotateRefWeightsAlias:
    """Reference weights alias for LayerNormRoPERotate."""

    k_weight = "self_attn.indexer.k_norm.weight"
    k_bias = "self_attn.indexer.k_norm.bias"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.k_weight, self.k_bias]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class LayerNormRoPERotateTilertWeightsAlias:
    """TileRT weights alias for LayerNormRoPERotate."""

    k_weight = "k_weights"
    k_bias = "k_bias"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.k_weight, self.k_bias]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class LayerNormRoPERotate(TileRTModule):
    """LayerNormRoPERotate module: LayerNorm + RoPE + rotate on K indexer output."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: LayerNormRoPERotateRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = LayerNormRoPERotateTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else LayerNormRoPERotateRefWeightsAlias()
        )

        self.rope_head_dim = self.model_args.qk_rope_head_dim
        self.head_dim = self.model_args.index_head_dim

        self.ref_weight: torch.Tensor | None = None
        self.ref_bias: torch.Tensor | None = None
        self.tilert_weight: torch.Tensor | None = None
        self.tilert_bias: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_weight, self.tilert_bias]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: replicate weight and bias for each device.

        Args:
            weights_map: Map from ref weight alias to tensor.

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        k_weight = weights_map[self.ref_weights_alias.k_weight][None, ...].repeat(
            self.num_devices, 1
        )
        k_bias = weights_map[self.ref_weights_alias.k_bias][None, ...].repeat(self.num_devices, 1)
        return {
            self.tilert_weights_alias.k_weight: k_weight,
            self.tilert_weights_alias.k_bias: k_bias,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.ref_weight = state_dict[self.ref_weights_alias.k_weight].contiguous().float()
        self.ref_bias = state_dict[self.ref_weights_alias.k_bias].contiguous().float()

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tilert_weight = state_dict[self.tilert_weights_alias.k_weight].contiguous().float()
        self.tilert_bias = state_dict[self.tilert_weights_alias.k_bias].contiguous().float()

    def init_random_weights(self) -> None:
        ref_weight = torch.ones(self.head_dim, dtype=torch.float32)
        ref_bias = torch.zeros(self.head_dim, dtype=torch.float32)
        ref_state_dict = dict(zip(self.ref_weights_alias(), [ref_weight, ref_bias]))
        self.init_reference_weights(ref_state_dict)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state_dict).items()}
        )

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.cur_pos = torch.tensor([0], dtype=torch.int32)
        self.output = torch.zeros((batch_size, seq_len, self.head_dim), dtype=torch.bfloat16)
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, idx_k: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        assert self.ref_weight is not None and self.ref_bias is not None
        k = F.layer_norm(
            idx_k.float(),
            (self.head_dim,),
            self.ref_weight,
            self.ref_bias,
            1e-6,
        ).to(idx_k.dtype)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        return rotate_activation(k)

    def tilert_forward(self, idx_k: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        assert self.tilert_weight is not None and self.tilert_bias is not None
        assert self.output is not None and self.profile_logs is not None
        rope_freqs = (
            torch.view_as_real(freqs_cis).reshape(*freqs_cis.shape[:-1], -1).float().unsqueeze(1)
        )
        layernorm_rope_rotate(
            idx_k,
            self.cur_pos,
            self.output,
            self.tilert_weight,
            self.tilert_bias,
            rope_freqs,
            self.profile_logs,
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.output

    def __call__(self, idx_k: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        if self.flag_enable_tilert:
            return self.tilert_forward(idx_k, freqs_cis)
        return self.golden_forward(idx_k, freqs_cis)
