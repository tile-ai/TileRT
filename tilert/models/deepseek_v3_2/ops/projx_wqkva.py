"""ProjXWqkva operation module."""

from enum import Enum

import torch

from tilert.models.base import TileRTModule
from tilert.models.common import weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqkva import (
    RMSNormProjQKVAFP8MMAWeightsConverter,
    RMSNormProjQKVAFP16MMAWeightsConverter,
)
from tilert.utils import get_profile_log_tensor

__all__ = [
    "ProjXWqkva",
    "projx_wqkva",
]


def projx_wqkva(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    wqkva: torch.Tensor,
    cur_pos: torch.Tensor,
    q_out: torch.Tensor,
    kv_out: torch.Tensor,
    pe_cache_out: torch.Tensor,
    profile_logs: torch.Tensor,
    compute_kernel_type: str = "fp8mma",
    *,
    model_arch: str,
) -> None:
    """FP8 MMA projection for q, kv, pe_cache (DSV3.2)."""
    torch.ops.tilert.projx_wqkva_op(
        x_quant,
        x_scale,
        wqkva,
        cur_pos,
        q_out,
        kv_out,
        pe_cache_out,
        model_arch,
        compute_kernel_type,
        profile_logs,
        torch.empty(0, dtype=torch.int64, device=x_quant.device),
    )


class ProjXWqkvaRefWeightsAlias:
    """Reference weight aliases for ProjXWqkva."""

    x_rmsnorm_gamma = "input_layernorm.weight"
    q_a_weights = "self_attn.q_a_proj.weight"
    q_a_scales = "self_attn.q_a_proj.weight_scale_inv"
    kv_a_weights = "self_attn.kv_a_proj_with_mqa.weight"
    kv_a_scales = "self_attn.kv_a_proj_with_mqa.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.kv_a_weights,
            self.kv_a_scales,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


class ProjXWqkvaTilertWeightsAlias:
    """Tilert weight aliases for ProjXWqkva."""

    q_a_weights = "q_a_weights"
    q_a_scales = "q_a_scales"
    kv_a_weights = "kv_a_weights"
    kv_a_scales = "kv_a_scales"
    w_pe_weights = "w_pe_weights"
    w_pe_scales = "w_pe_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.q_a_weights,
            self.q_a_scales,
            self.kv_a_weights,
            self.kv_a_scales,
            self.w_pe_weights,
            self.w_pe_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ProjXWqkvaAlgorithm(Enum):
    """ProjXWqkva algorithm."""

    FP8MMA = "fp8mma"
    FP16MMA = "fp16mma"


class ProjXWqkva(TileRTModule):
    """FP8 MMA projection module for q, kv, pe_cache."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [ProjXWqkvaAlgorithm.FP8MMA],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: ProjXWqkvaRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = ProjXWqkvaTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else ProjXWqkvaRefWeightsAlias()
        )

        self.dim = self.model_args.dim
        self.q_lora_rank = self.model_args.q_lora_rank
        self.kv_lora_rank = self.model_args.kv_lora_rank
        self.qk_rope_head_dim = self.model_args.qk_rope_head_dim
        self.block_size = self.model_args.block_size
        self.eps = self.model_args.eps

        self.ref_wq_a: torch.Tensor | None = None
        self.ref_wkv_a: torch.Tensor | None = None
        self.ref_w_pe: torch.Tensor | None = None

        self.tilert_wqkva: torch.Tensor | None = None

        self.q_out: torch.Tensor | None = None
        self.kv_out: torch.Tensor | None = None
        self.pe_cache_out: torch.Tensor | None = None
        self.cur_pos: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self.compute_kernel_type = "fp8mma"

    def set_algorithm(self, algorithm: Enum) -> None:
        super().set_algorithm(algorithm)
        if algorithm == ProjXWqkvaAlgorithm.FP16MMA:
            self.compute_kernel_type = "fp16mma"
        else:
            self.compute_kernel_type = "fp8mma"

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Repeat weights for device sharding."""
        q_a_proj_weight = weights_map[self.ref_weights_alias.q_a_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        q_a_proj_weight_scale = weights_map[self.ref_weights_alias.q_a_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        kv_a_mqa = weights_map[self.ref_weights_alias.kv_a_weights]
        kv_a_proj_weight = kv_a_mqa[: self.kv_lora_rank, :][None, ...].repeat(
            self.num_devices, 1, 1
        )
        w_pe_weight = kv_a_mqa[self.kv_lora_rank :, :][None, ...].repeat(self.num_devices, 1, 1)
        kv_a_mqa_scale = weights_map[self.ref_weights_alias.kv_a_scales]
        kv_scale_rows = (self.kv_lora_rank + self.block_size - 1) // self.block_size
        kv_a_proj_weight_scale = kv_a_mqa_scale[:kv_scale_rows, :][None, ...].repeat(
            self.num_devices, 1, 1
        )
        w_pe_weight_scale = kv_a_mqa_scale[kv_scale_rows:, :][None, ...].repeat(
            self.num_devices, 1, 1
        )
        return {
            self.tilert_weights_alias.q_a_weights: q_a_proj_weight,
            self.tilert_weights_alias.q_a_scales: q_a_proj_weight_scale,
            self.tilert_weights_alias.kv_a_weights: kv_a_proj_weight,
            self.tilert_weights_alias.kv_a_scales: kv_a_proj_weight_scale,
            self.tilert_weights_alias.w_pe_weights: w_pe_weight,
            self.tilert_weights_alias.w_pe_scales: w_pe_weight_scale,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        aliases = self.ref_weights_alias()
        self.ref_wq_a = weight_dequant(state_dict[aliases[1]], state_dict[aliases[2]])
        kv_a_mqa = weight_dequant(state_dict[aliases[3]], state_dict[aliases[4]])
        self.ref_wkv_a = kv_a_mqa[: self.kv_lora_rank, :]
        self.ref_w_pe = kv_a_mqa[self.kv_lora_rank :, :]

        assert self.ref_wq_a.shape == (self.q_lora_rank, self.dim)
        assert self.ref_wkv_a.shape == (self.kv_lora_rank, self.dim)
        assert self.ref_w_pe.shape == (self.qk_rope_head_dim, self.dim)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        tilert_aliases = self.tilert_weights_alias()
        wq_a = state_dict[tilert_aliases[0]]
        wq_a_scale = state_dict[tilert_aliases[1]]
        wkv_a = state_dict[tilert_aliases[2]]
        wkv_a_scale = state_dict[tilert_aliases[3]]
        w_pe = state_dict[tilert_aliases[4]]
        w_pe_scale = state_dict[tilert_aliases[5]]
        dummy_gamma = torch.zeros(self.dim, dtype=torch.float32, device=wq_a.device)

        if self.algorithm == ProjXWqkvaAlgorithm.FP16MMA:
            self.tilert_wqkva, _ = RMSNormProjQKVAFP16MMAWeightsConverter.convert_to_fp16_mma_gemv(
                wq_a,
                wq_a_scale,
                wkv_a,
                wkv_a_scale,
                w_pe,
                w_pe_scale,
                dummy_gamma,
                hidden_dim=self.dim,
                q_lora_rank=self.q_lora_rank,
            )
        else:
            self.tilert_wqkva, _ = RMSNormProjQKVAFP8MMAWeightsConverter.convert_to_fp8_mma_gemv(
                wq_a,
                wq_a_scale,
                wkv_a,
                wkv_a_scale,
                w_pe,
                w_pe_scale,
                dummy_gamma,
                hidden_dim=self.dim,
                q_lora_rank=self.q_lora_rank,
            )

    def init_tilert_vars(self, batch_size: int, seq_len: int, max_len: int = 128) -> None:
        self.q_out = torch.zeros((batch_size, seq_len, self.q_lora_rank), dtype=torch.bfloat16)
        self.kv_out = torch.zeros((batch_size, seq_len, self.kv_lora_rank), dtype=torch.bfloat16)
        self.pe_cache_out = torch.zeros(
            (batch_size, max_len, self.qk_rope_head_dim), dtype=torch.bfloat16
        )
        self.cur_pos = torch.zeros((1,), dtype=torch.int32)
        self.profile_logs = get_profile_log_tensor()
        self.is_init = True

    def init_random_weights(self) -> None:
        bs = self.block_size
        dim_scale_dim = self.dim // bs
        q_scale_dim = (self.q_lora_rank + bs - 1) // bs
        kv_mqa_rows = self.kv_lora_rank + self.qk_rope_head_dim
        kv_mqa_scale_dim = (kv_mqa_rows + bs - 1) // bs
        scale_dtype = torch.bfloat16

        tensor_list = [
            torch.randn(self.dim, dtype=torch.float32),
            torch.randn(self.q_lora_rank, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(q_scale_dim, dim_scale_dim, dtype=scale_dtype),
            torch.randn(kv_mqa_rows, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(kv_mqa_scale_dim, dim_scale_dim, dtype=scale_dtype),
        ]
        ref_state_dict = dict(zip(self.ref_weights_alias(), tensor_list))
        self.init_reference_weights(ref_state_dict)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state_dict).items()}
        )

    def golden_forward(
        self,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor,
        cur_pos: int = 0,  # noqa: U100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch reference: dequant FP8 -> matmul -> q, kv, pe."""
        assert self.ref_wq_a is not None
        assert self.ref_wkv_a is not None
        assert self.ref_w_pe is not None

        if self.algorithm == ProjXWqkvaAlgorithm.FP16MMA:
            x_float = x_quant.float()
        else:
            x_fp8 = x_quant.to(torch.float32)
            scale_expanded = x_scale.unsqueeze(-1).repeat(1, 1, 1, self.block_size)
            scale_expanded = scale_expanded.reshape(x_quant.shape)
            x_float = x_fp8 * scale_expanded

        q_out = torch.matmul(x_float, self.ref_wq_a.transpose(0, 1).float())
        kv_out = torch.matmul(x_float, self.ref_wkv_a.transpose(0, 1).float())
        pe_out = torch.matmul(x_float, self.ref_w_pe.transpose(0, 1).float())
        return (
            q_out.to(torch.bfloat16),
            kv_out.to(torch.bfloat16),
            pe_out.to(torch.bfloat16),
        )

    def tilert_forward(
        self,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor,
        cur_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run FP8 QMMA GEMV via TileRT CUDA kernel."""
        assert self.cur_pos is not None
        assert self.pe_cache_out is not None
        self.cur_pos.fill_(cur_pos)
        projx_wqkva(
            x_quant,
            x_scale,
            self.tilert_wqkva,
            self.cur_pos,
            self.q_out,
            self.kv_out,
            self.pe_cache_out,
            self.profile_logs,
            self.compute_kernel_type,
            model_arch=self.model_args.arch_name,
        )

        seq_len = x_quant.size(-2)
        pe_at_pos = self.pe_cache_out[:, cur_pos : cur_pos + seq_len, :]
        return self.q_out, self.kv_out, pe_at_pos

    def __call__(
        self,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor,
        cur_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.golden_forward(x_quant, x_scale, cur_pos)
