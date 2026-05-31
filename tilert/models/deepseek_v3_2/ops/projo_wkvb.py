"""ProjOWkvb operation module."""

import math
from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import init_func, weight_dequant
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "projo_wkvb",
    "ProjoWKVb",
    "ProjoWKVbAlgorithm",
    "ProjoWKVbWeightsConverter",
    "ProjoWKVbRefWeightsAlias",
    "ProjoWKVbTilertWeightsAlias",
]


def projo_wkvb(
    o_in: torch.Tensor,
    wkv_b_b: torch.Tensor,
    wkv_b_scales: torch.Tensor,
    output: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "fp16mma",
) -> None:
    """
    Define the ProjOWkvb operation.

    Args:
        o_in: Input tensor.
        wkv_b_b: Weight tensor.
        wkv_b_scales: Scale tensor.
        output: Output tensor.
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
        compute_kernel_type: Kernel type ("fp16mma" for both DSv32 and GLM5).
    """
    torch.ops.tilert.projo_wkvb_op(
        o_in,
        wkv_b_b,
        wkv_b_scales,
        output,
        model_arch,
        compute_kernel_type,
        profile_logs,
        torch.empty(0, dtype=torch.int64, device=o_in.device),
    )


class ProjoWKVbAlgorithm(Enum):
    """ProjoWKVb algorithm"""

    GENERAL = "general"
    FP16MMA = "fp16mma"
    BF16MMA = "bf16mma"


class ProjoWKVbWeightsConverter(TilertWeightsConverter):
    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args, num_devices)

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        """Swizzle [*, 16, 16] for m16n8k16 MMA register layout."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16_for_pages(mat_in: torch.Tensor, k_dim: int, pages: int) -> torch.Tensor:
        """Swizzle [*, 16, K] matrix for paged MMA layout."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == k_dim
        pre_shape = mat_in.shape[:-2]
        k_per_page = k_dim // pages
        n_k_tiles = k_per_page // 16
        mat_in = mat_in.reshape(*pre_shape, 16, pages, k_per_page).transpose(-3, -2)
        mat_in = mat_in.reshape(*pre_shape, pages, 16, n_k_tiles, 16).transpose(-3, -2)
        mat_in = ProjoWKVbWeightsConverter._swizzle_mma_16x16(mat_in)
        return mat_in.contiguous()

    def convert_to_fp16mma(self, weights: list[torch.Tensor]) -> torch.Tensor:
        """Convert weights to HMMA packed format: [num_ctas, page_size] fp8."""
        with torch.inference_mode():
            wkv_b_b, wkv_b_b_scales = self.convert_to_general(weights)
            # wkv_b_b: [n_heads, v_head_dim, kv_lora_rank] fp8
            # wkv_b_b_scales: [n_heads, v_head_dim//block_size, kv_lora_rank//block_size]

            n_heads = wkv_b_b.size(0)
            v_head_dim = wkv_b_b.size(1)
            kv_lora_rank = wkv_b_b.size(2)  # 512
            num_ctas = 80
            rows_per_cta = (n_heads * v_head_dim) // num_ctas  # 32

            is_glm5 = self.model_args.arch_name == "glm_5"

            # Reshape + swizzle (K dimension is kv_lora_rank=512)
            w_flat = wkv_b_b.reshape(num_ctas, rows_per_cta // 16, 16, kv_lora_rank)
            w_swizzled = ProjoWKVbWeightsConverter._swizzle_mma_16x16_for_pages(
                w_flat, kv_lora_rank, pages=1
            )
            w_bytes = w_swizzled.reshape(num_ctas, -1)

            # Scales: always float (GemvHMMA hardcodes float)
            scale_k_block = 128
            n_scale_k = kv_lora_rank // scale_k_block  # 4
            ctas_per_head = num_ctas // n_heads

            if is_glm5:
                # GLM5: scale_n_block=64, 2 CTAs share 1 scale row
                ctas_per_scale_row = 64 // rows_per_cta  # 2
                scales_per_cta = wkv_b_b_scales.repeat_interleave(ctas_per_scale_row, dim=1)
                scales_per_cta = scales_per_cta.reshape(num_ctas, n_scale_k)
            else:
                # DSV3.2: scale_n_block=128, 4 CTAs per head, all share same scale
                scales_per_cta = wkv_b_b_scales.squeeze(1).repeat_interleave(
                    ctas_per_head, dim=0
                )  # [80, 4]

            # Promote all scales to float32 for GemvHMMA
            scale_dtype = torch.float32
            scales_per_cta = scales_per_cta.to(scale_dtype)

            # Pack per CTA, 128-byte aligned
            mat_bytes = rows_per_cta * kv_lora_rank
            scale_bytes = n_scale_k * 4  # always float32
            page_size = (mat_bytes + scale_bytes + 127) // 128 * 128

            scales_raw = scales_per_cta.contiguous().view(torch.float8_e4m3fn)
            padding_size = page_size - mat_bytes - scales_raw.shape[-1]
            padding = torch.zeros(
                num_ctas, padding_size, dtype=torch.float8_e4m3fn, device=wkv_b_b.device
            )
            return torch.cat([w_bytes, scales_raw, padding], dim=-1).contiguous()

    def convert_to_bf16mma(self, weights: list[torch.Tensor]) -> torch.Tensor:
        """Convert weights to BF16 HMMA packed format: dequant -> swizzle -> pack."""
        with torch.inference_mode():
            tilert_wkv_b_weights, tilert_wkv_b_scales = weights

            # Compute head_dim_block_size (same logic as ProjoWKVb.__init__)
            wkvb_head_dim = self.model_args.qk_nope_head_dim + self.model_args.v_head_dim
            left_head_dim = wkvb_head_dim % self.model_args.block_size
            hd_block = left_head_dim if left_head_dim != 0 else self.model_args.block_size

            # Compute n_local_heads with head padding (same as convert_to_general)
            if self.model_args.n_heads % self.num_devices == 0:
                n_local_heads = self.model_args.n_heads // self.num_devices
            else:
                n_local_heads = math.ceil(self.model_args.n_heads / self.num_devices)
                if n_local_heads % 2 != 0:
                    n_local_heads += 1

            # Input from device_sharding is already v-extracted:
            #   weights: [n_local_heads, v_head_dim, kv_lora_rank] FP8
            #   scales:  [n_local_heads, v_head_dim // hd_block, kv_lora_rank // 128]
            # Dequantize using raw scales with per-block broadcast.
            v_head_dim = self.model_args.v_head_dim
            kv_lora_rank = self.model_args.kv_lora_rank
            n_block = self.model_args.block_size  # 128

            # Handle head padding (same as convert_to_general)
            w = tilert_wkv_b_weights
            s = tilert_wkv_b_scales
            if self.model_args.n_heads % self.num_devices != 0:
                n_current = w.size(0)
                if n_current < n_local_heads:
                    pad_w = torch.zeros(
                        n_local_heads - n_current, *w.shape[1:], dtype=w.dtype, device=w.device
                    )
                    w = torch.cat([w, pad_w], dim=0)
                    pad_s = torch.zeros(
                        n_local_heads - n_current, *s.shape[1:], dtype=s.dtype, device=s.device
                    )
                    s = torch.cat([s, pad_s], dim=0)

            # Expand scales to match weight dims via repeat_interleave, then multiply
            # scales: [n_local_heads, v_head_dim // hd_block, kv_lora_rank // n_block]
            s = s.float()
            s = s.repeat_interleave(hd_block, dim=1).repeat_interleave(n_block, dim=2)
            # s: [n_local_heads, v_head_dim, kv_lora_rank]
            wkv_bf16 = (w.float() * s).to(torch.bfloat16)
            # wkv_bf16: [n_heads, v_head_dim, kv_lora_rank]
            n_heads = n_local_heads

            num_ctas = 80
            rows_per_cta = (n_heads * v_head_dim) // num_ctas

            # 3. Reshape + swizzle (K dimension is kv_lora_rank=512)
            w_flat = wkv_bf16.reshape(num_ctas, rows_per_cta // 16, 16, kv_lora_rank)
            w_swizzled = ProjoWKVbWeightsConverter._swizzle_mma_16x16_for_pages(
                w_flat, kv_lora_rank, pages=1
            )
            # View as raw bytes (fp8 view for byte-level packing)
            w_bytes = w_swizzled.reshape(num_ctas, -1).contiguous().view(torch.float8_e4m3fn)

            # 4. Pack per CTA: [bf16_weights | padding], 128-byte aligned, NO scales
            mat_bytes = rows_per_cta * kv_lora_rank * 2  # bf16 = 2 bytes
            page_size = (mat_bytes + 127) // 128 * 128
            padding_size = page_size - w_bytes.shape[-1]

            if padding_size > 0:
                padding = torch.zeros(
                    num_ctas, padding_size, dtype=torch.float8_e4m3fn, device=wkv_bf16.device
                )
                return torch.cat([w_bytes, padding], dim=-1).contiguous()
            return w_bytes.contiguous()

    def convert_to_general(self, weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            tilert_wkv_b_weights, tilert_wkv_b_scales = weights

            # Input weights are already in the correct shape from device_sharding:
            # wkv_b_weights: (n_local_heads, v_head_dim, kv_lora_rank)
            # wkv_b_scales: (n_local_heads, v_head_dim // block_size, kv_lora_rank // block_size)
            wkv_b_b = tilert_wkv_b_weights.contiguous()
            wkv_b_b_scales = tilert_wkv_b_scales.contiguous()
            if self.model_args.arch_name == "glm_5":
                if wkv_b_b_scales.dtype != torch.float32:
                    print(
                        "Warning: ProjoWKVbWeightsConverter: "
                        + f"wkv_b_b_scales.dtype: {wkv_b_b_scales.dtype} "
                        + "is not float32, convert to float32."
                    )
                wkv_b_b_scales = wkv_b_b_scales.to(torch.float32)
            else:  # DS v3.2, use bfloat16 for wkv_b_b_scales
                wkv_b_b_scales = wkv_b_b_scales.to(torch.bfloat16)

            wkv_b_b = wkv_b_b.detach()
            wkv_b_b_scales = wkv_b_b_scales.detach()

            # Auto-detect Device Group B: pad heads when they don't divide evenly.
            # self.num_devices is already the PureMla GPU count (7), not total (8).
            if self.model_args.n_heads % self.num_devices != 0:
                n_target = math.ceil(self.model_args.n_heads / self.num_devices)
                if n_target % 2 != 0:
                    n_target += 1  # Round up to nearest even for kernel alignment
                n_current = wkv_b_b.size(0)
                if n_current < n_target:
                    pad_b = torch.zeros(
                        n_target - n_current,
                        *wkv_b_b.shape[1:],
                        dtype=wkv_b_b.dtype,
                        device=wkv_b_b.device,
                    )
                    wkv_b_b = torch.cat([wkv_b_b, pad_b], dim=0)
                    pad_s = torch.zeros(
                        n_target - n_current,
                        *wkv_b_b_scales.shape[1:],
                        dtype=wkv_b_b_scales.dtype,
                        device=wkv_b_b_scales.device,
                    )
                    wkv_b_b_scales = torch.cat([wkv_b_b_scales, pad_s], dim=0)
                wkv_b_b = wkv_b_b.contiguous()
                wkv_b_b_scales = wkv_b_b_scales.contiguous()

        return wkv_b_b, wkv_b_b_scales


@dataclass
class ProjoWKVbRefWeightsAlias:
    """Reference weights alias for ProjoWKVb."""

    wkv_b_weights = "self_attn.kv_b_proj.weight"
    wkv_b_scales = "self_attn.kv_b_proj.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.wkv_b_weights, self.wkv_b_scales]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ProjoWKVbTilertWeightsAlias:
    """TileRT weights alias for ProjoWKVb."""

    wkv_b_weights = "wkv_b2_weights"
    wkv_b_scales = "wkv_b2_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.wkv_b_weights, self.wkv_b_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ProjoWKVb(TileRTModule):
    """ProjoWKVb module: O projection (wkv_b) for output."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [ProjoWKVbAlgorithm.FP16MMA],
        "glm_5": [ProjoWKVbAlgorithm.FP16MMA],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ProjoWKVbRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = ProjoWKVbTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else ProjoWKVbRefWeightsAlias()
        )

        self.ref_wkv_b: torch.Tensor | None = None
        self.tilert_wkv_b_b: torch.Tensor | None = None
        self.tilert_wkv_b_b_scales: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

        # Compute padded n_local_heads when heads don't divide evenly (TP7 Device Group B)
        if self.model_args.n_heads % self.num_devices == 0:
            self.num_local_heads = self.model_args.n_heads // self.num_devices
        else:
            n_local = math.ceil(self.model_args.n_heads / self.num_devices)
            if n_local % 2 != 0:
                n_local += 1  # Round up to nearest even for kernel alignment
            self.num_local_heads = n_local

        # lora dim and quant block size
        self.wkvb_lora_rank = self.model_args.kv_lora_rank
        self.wkvb_lora_rank_qsize = self.wkvb_lora_rank // self.model_args.block_size

        self.wkvb_head_dim = self.model_args.qk_nope_head_dim + self.model_args.v_head_dim
        self.wkvb_v_head_dim = self.model_args.v_head_dim
        left_head_dim = self.wkvb_head_dim % self.model_args.block_size
        if left_head_dim != 0:
            assert self.model_args.block_size % left_head_dim == 0
            self.head_dim_block_size = left_head_dim
            self.head_dim_scale_repeat = self.model_args.block_size // self.head_dim_block_size
        else:
            self.head_dim_scale_repeat = 1
            self.head_dim_block_size = self.model_args.block_size
        self.wkvb_head_qsize = self.wkvb_head_dim // self.head_dim_block_size
        self.wkvb_v_head_qsize = self.wkvb_v_head_dim // self.head_dim_block_size

        self.compute_kernel_type = "fp16mma"

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_wkv_b_b, self.tilert_wkv_b_b_scales]

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

        if self.model_args.n_heads % self.num_devices == 0:
            # Even split: direct view into (num_devices, num_local_heads, ...)
            dev_weights = kv_b_proj_weight.view(
                self.num_devices, self.num_local_heads, self.wkvb_head_dim, self.wkvb_lora_rank
            )
            dev_scale_rows = self.num_local_heads * self.wkvb_head_dim // self.model_args.block_size
            dev_scales = kv_b_proj_weight_scale.view(
                self.num_devices, dev_scale_rows, 1, self.wkvb_lora_rank_qsize
            )
        else:
            # Padded redistribution for uneven head split (e.g. 64 heads / 7 GPUs → H=10)
            from tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqb import (
                RmsnormProjqWqbWeightsConverter,
            )

            wq_b_list, scale_list = RmsnormProjqWqbWeightsConverter._redistribute_heads(
                kv_b_proj_weight,
                kv_b_proj_weight_scale,
                n_total_heads=self.model_args.n_heads,
                n_local_heads=self.num_local_heads,
                num_devices=self.num_devices,
                qk_head_dim=self.wkvb_head_dim,
                block_size=self.model_args.block_size,
            )
            dev_weights = torch.stack(wq_b_list, dim=0).view(
                self.num_devices, self.num_local_heads, self.wkvb_head_dim, self.wkvb_lora_rank
            )
            dev_scale_rows = self.num_local_heads * self.wkvb_head_dim // self.model_args.block_size
            dev_scales = torch.stack(scale_list, dim=0).view(
                self.num_devices, dev_scale_rows, 1, self.wkvb_lora_rank_qsize
            )

        # Extract v part per head
        wkvb = dev_weights[:, :, -self.wkvb_v_head_dim :]
        wkvb_scales = (
            dev_scales.contiguous()
            .repeat(1, 1, self.head_dim_scale_repeat, 1)
            .view(
                self.num_devices,
                self.num_local_heads,
                self.wkvb_head_qsize,
                self.wkvb_lora_rank_qsize,
            )
            .contiguous()[:, :, -self.wkvb_v_head_qsize :]
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
        self.ref_wkv_b = wkv_b[:, -self.wkvb_v_head_dim :]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.init_tilert_weights_hmma(state_dict)

    def init_tilert_weights_hmma(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize with HMMA-packed weights."""
        packed = ProjoWKVbWeightsConverter(self.model_args, self.num_devices).dispatch(
            ProjoWKVbAlgorithm.FP16MMA,
            [
                state_dict[self.tilert_weights_alias.wkv_b_weights],
                state_dict[self.tilert_weights_alias.wkv_b_scales],
            ],
        )
        self.tilert_wkv_b_b = packed
        # Dummy scales — HMMA packs scales into the weight tensor.
        self.tilert_wkv_b_b_scales = torch.empty(1, dtype=torch.float8_e4m3fn, device=packed.device)
        self.compute_kernel_type = "fp16mma"

    def init_tilert_weights_hmma_bf16(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize with BF16 HMMA-packed weights (dequantized, no scales)."""
        packed = ProjoWKVbWeightsConverter(self.model_args, self.num_devices).dispatch(
            ProjoWKVbAlgorithm.BF16MMA,
            [
                state_dict[self.tilert_weights_alias.wkv_b_weights],
                state_dict[self.tilert_weights_alias.wkv_b_scales],
            ],
        )
        self.tilert_wkv_b_b = packed
        self.tilert_wkv_b_b_scales = torch.empty(1, dtype=torch.float8_e4m3fn, device=packed.device)
        self.compute_kernel_type = "bf16mma"

    def init_random_weights(self) -> None:
        # Use padded total heads when heads don't divide evenly across devices
        # (e.g. 64 heads / 7 GPUs → num_local_heads=10, padded_total=70).
        # This ensures every device's slice in init_reference_weights has enough rows.
        padded_total_heads = self.num_local_heads * self.num_devices
        wkv_b = init_func(
            torch.empty(
                padded_total_heads * self.wkvb_head_dim,
                self.wkvb_lora_rank,
                dtype=torch.float8_e4m3fn,
            )
        )
        wkv_b_scales = init_func(
            torch.empty(
                # Block quant should be applied to the original weight dimension (including head
                # dimension)
                padded_total_heads * self.wkvb_head_dim // self.model_args.block_size,
                self.wkvb_lora_rank_qsize,
                dtype=torch.float32,
            )
        )
        ref_state_dict = dict(
            zip(
                self.ref_weights_alias(),
                [wkv_b, wkv_b_scales],
            )
        )
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        self.init_tilert_weights({k: v[self.device_id] for k, v in sharded.items()})

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.output = torch.zeros(
            (batch_size, seq_len, self.num_local_heads, self.wkvb_v_head_dim),
            dtype=torch.bfloat16,
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, x_out: torch.Tensor) -> torch.Tensor:
        assert self.ref_wkv_b is not None
        return torch.einsum("bshc,hdc->bshd", x_out, self.ref_wkv_b)

    def tilert_forward(self, x_out: torch.Tensor) -> torch.Tensor:
        assert self.tilert_wkv_b_b is not None
        assert self.tilert_wkv_b_b_scales is not None
        assert self.output is not None
        assert self.profile_logs is not None
        projo_wkvb(
            x_out,
            self.tilert_wkv_b_b,
            self.tilert_wkv_b_b_scales,
            self.output,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
            compute_kernel_type=self.compute_kernel_type,
        )
        return self.output
