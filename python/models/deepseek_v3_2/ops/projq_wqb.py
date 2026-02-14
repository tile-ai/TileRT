"""ProjQB operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import init_func, weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "projq_wqb",
    "ProjqWqb",
    "ProjqWqbAlgorithm",
    "ProjqWqbWeightsConverter",
    "ProjqWqbRefWeightsAlias",
    "ProjqWqbTilertWeightsAlias",
]


def projq_wqb(
    q_nope_in: torch.Tensor,
    wkv_b_a: torch.Tensor,
    wkv_b_a_scales: torch.Tensor,
    output: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the ProjqWqb operation.

    Args:
        q_nope_in: Input tensor.
        wkv_b_a: Weight tensor.
        wkv_b_a_scales: Scale tensor.
        output: Output tensor.
        profile_logs: Profile logs tensor.
    """
    if q_nope_in.shape[-1] == 128:
        torch.ops.tilert.projq_wqb_op(q_nope_in, wkv_b_a, wkv_b_a_scales, output, profile_logs)
    elif q_nope_in.shape[-1] == 192:
        torch.ops.tilert.proj_qb_glm5_op(q_nope_in, wkv_b_a, wkv_b_a_scales, output, profile_logs)


class ProjqWqbAlgorithm(Enum):
    """ProjqWqb algorithm"""

    GENERAL = "general"


class ProjqWqbWeightsConverter(TilertWeightsConverter):
    def __init__(self, model_args: ModelArgs, num_devices: int, head_dim_block_size: int):
        super().__init__(model_args, num_devices)
        self.head_dim_block_size = head_dim_block_size
        self.impl_block_size = 64

    def convert_to_general(self, weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            tilert_wkv_b_weights, tilert_wkv_b_scales = weights

            n_local_heads = self.model_args.n_heads // self.num_devices

            wkv_b = tilert_wkv_b_weights
            wkv_b_scales_raw = tilert_wkv_b_scales
            wkv_b = wkv_b.view(n_local_heads, -1, self.model_args.kv_lora_rank)
            assert self.model_args.kv_lora_rank % self.model_args.block_size == 0
            wkv_b_scales_raw = wkv_b_scales_raw.view(
                n_local_heads, -1, self.model_args.kv_lora_rank // self.model_args.block_size
            )
            wkv_b_a = wkv_b[:, : self.model_args.qk_nope_head_dim].transpose(1, 2).contiguous()
            assert self.model_args.qk_nope_head_dim % self.head_dim_block_size == 0
            wkv_b_a_scales = (
                wkv_b_scales_raw[:, : self.model_args.qk_nope_head_dim // self.head_dim_block_size]
                .transpose(1, 2)
                .contiguous()
            )
            if self.model_args.arch_name == "glm_5":
                if wkv_b_a_scales.dtype != torch.float32:
                    print(
                        "Warning: ProjqWqbWeightsConverter: "
                        + f"wkv_b_a_scales.dtype: {wkv_b_a_scales.dtype} "
                        + "is not float32, convert to float32."
                    )
                wkv_b_a_scales = wkv_b_a_scales.to(torch.float32)
            else:  # DS v3.2, use bfloat16 for wkv_b_a_scales
                wkv_b_a_scales = wkv_b_a_scales.to(torch.bfloat16)
            # Tiling to fit tilert input
            if self.head_dim_block_size != self.impl_block_size:
                repeats = self.head_dim_block_size // self.impl_block_size
                wkv_b_a_scales = wkv_b_a_scales.repeat(1, 1, repeats).contiguous()

            wkv_b_a = wkv_b_a.detach()
            wkv_b_a_scales = wkv_b_a_scales.detach()

        return wkv_b_a, wkv_b_a_scales


@dataclass
class ProjqWqbRefWeightsAlias:
    """Reference weights alias for ProjqWqb."""

    wkv_b_weights = "self_attn.kv_b_proj.weight"
    wkv_b_scales = "self_attn.kv_b_proj.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.wkv_b_weights, self.wkv_b_scales]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ProjqWqbTilertWeightsAlias:
    """TileRT weights alias for ProjqWqb."""

    wkv_b_weights = "wkv_b1_weights"
    wkv_b_scales = "wkv_b1_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.wkv_b_weights, self.wkv_b_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ProjqWqb(TileRTModule):
    """ProjqWqb module: Q projection (wkv_b) for KV LoRA."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ProjqWqbRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = ProjqWqbTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else ProjqWqbRefWeightsAlias()
        )

        self.ref_wkv_b: torch.Tensor | None = None
        self.tilert_wkv_b_a: torch.Tensor | None = None
        self.tilert_wkv_b_a_scales: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

        self.num_local_heads = self.model_args.n_heads // self.num_devices

        # lora dim and quant block size
        self.wkvb_lora_rank = self.model_args.kv_lora_rank
        self.wkvb_lora_rank_qsize = self.wkvb_lora_rank // self.model_args.block_size

        self.wkvb_head_dim = self.model_args.qk_nope_head_dim + self.model_args.v_head_dim
        self.wkvb_nope_head_dim = self.model_args.qk_nope_head_dim
        left_head_dim = self.wkvb_head_dim % self.model_args.block_size
        if left_head_dim != 0:
            assert self.model_args.block_size % left_head_dim == 0
            self.head_dim_block_size = left_head_dim
            self.head_dim_scale_repeat = self.model_args.block_size // self.head_dim_block_size
        else:
            self.head_dim_scale_repeat = 1
            self.head_dim_block_size = self.model_args.block_size
        self.wkvb_head_qsize = self.wkvb_head_dim // self.head_dim_block_size
        self.wkvb_nope_head_qsize = self.wkvb_nope_head_dim // self.head_dim_block_size

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_wkv_b_a, self.tilert_wkv_b_a_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: split weights and scales per device.

        Args:
            weights_map: Map from ref weight alias to tensor.

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        kv_b_proj_weight = weights_map[self.ref_weights_alias.wkv_b_weights]
        kv_b_proj_weight_scale = weights_map[self.ref_weights_alias.wkv_b_scales]

        dev_heads = (self.num_devices, self.num_local_heads)
        wkvb = kv_b_proj_weight.view(*dev_heads, self.wkvb_head_dim, self.wkvb_lora_rank)[
            :, :, : self.wkvb_nope_head_dim
        ]
        wkvb_scales = (
            kv_b_proj_weight_scale.view(
                self.num_devices,
                self.num_local_heads * self.wkvb_head_dim // self.model_args.block_size,
                1,
                self.wkvb_lora_rank_qsize,
            )
            .contiguous()
            .repeat(1, 1, self.head_dim_scale_repeat, 1)
            .view(
                self.num_devices,
                self.num_local_heads,
                self.wkvb_head_qsize,
                self.wkvb_lora_rank_qsize,
            )
            .contiguous()[:, :, : self.wkvb_nope_head_qsize]
        )
        return {
            self.tilert_weights_alias.wkv_b_weights: wkvb.contiguous(),
            self.tilert_weights_alias.wkv_b_scales: wkvb_scales.contiguous(),
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        sharding_size = self.num_local_heads * self.wkvb_head_dim
        sharding_start = self.device_id * sharding_size
        sharding_end = sharding_start + sharding_size
        wkv_b = weight_dequant(
            state_dict[self.ref_weights_alias.wkv_b_weights],
            state_dict[self.ref_weights_alias.wkv_b_scales],
        )
        wkv_b = wkv_b[sharding_start:sharding_end, :]
        wkv_b = wkv_b.view(self.num_local_heads, self.wkvb_head_dim, self.wkvb_lora_rank)
        self.ref_wkv_b = wkv_b[:, : self.wkvb_nope_head_dim]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tilert_wkv_b_a, self.tilert_wkv_b_a_scales = ProjqWqbWeightsConverter(
            self.model_args, self.num_devices, self.head_dim_block_size
        ).dispatch(
            ProjqWqbAlgorithm.GENERAL,
            [
                state_dict[self.tilert_weights_alias.wkv_b_weights],
                state_dict[self.tilert_weights_alias.wkv_b_scales],
            ],
        )

    def init_random_weights(self) -> None:
        wkv_b = init_func(
            torch.empty(
                self.model_args.n_heads * self.wkvb_head_dim,
                self.wkvb_lora_rank,
                dtype=torch.float8_e4m3fn,
            )
        )
        wkv_b_scales = init_func(
            torch.empty(
                # Block quant should be applied to the original weight dimension (including head
                # dimension)
                self.model_args.n_heads * self.wkvb_head_dim // self.model_args.block_size,
                self.wkvb_lora_rank_qsize,
                dtype=torch.float32,
            )
        )
        ref_state_dict = dict(zip(self.ref_weights_alias(), [wkv_b, wkv_b_scales]))
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        self.init_tilert_weights({k: v[self.device_id] for k, v in sharded.items()})

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.output = torch.zeros(
            (batch_size, seq_len, self.num_local_heads, self.wkvb_lora_rank), dtype=torch.bfloat16
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, q_nope: torch.Tensor) -> torch.Tensor:
        assert self.ref_wkv_b is not None
        return torch.einsum("bshd,hdc->bshc", q_nope, self.ref_wkv_b)

    def tilert_forward(self, q_nope: torch.Tensor) -> torch.Tensor:
        assert self.tilert_wkv_b_a is not None
        assert self.tilert_wkv_b_a_scales is not None
        assert self.output is not None
        assert self.profile_logs is not None
        projq_wqb(
            q_nope,
            self.tilert_wkv_b_a,
            self.tilert_wkv_b_a_scales,
            self.output,
            self.profile_logs,
        )
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.output
