"""RmsnormProjqWqi operation module (IQ-only projection)."""

from dataclasses import dataclass
from enum import Enum

import torch
from einops import rearrange

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RmsnormProjqWqi",
    "RmsnormProjqWqiAlgorithm",
    "RmsnormProjqWqiWeightsConverter",
]


def rmsnorm_projq_wqi_op(
    q: torch.Tensor,
    wqi: torch.Tensor,
    wqi_scale: torch.Tensor,
    rmsnorm_gamma: torch.Tensor,
    iq: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str,
    model_arch: str,
) -> None:
    torch.ops.tilert.rmsnorm_proj_qi_op(
        q,
        wqi,
        wqi_scale,
        rmsnorm_gamma,
        iq,
        model_arch,
        algorithm,
        profile_logs,
    )


class RmsnormProjqWqiAlgorithm(Enum):
    """RmsnormProjqWqi algorithm."""

    FP16MMA = "fp16mma"


class RmsnormProjqWqiWeightsConverter(TilertWeightsConverter):
    """Weights converter: common format to TileRT format (IQ only)."""

    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args=model_args, num_devices=num_devices)

        self.block_size = self.model_args.block_size
        self.q_lora_dim = self.model_args.q_lora_rank
        self.q_lora_qdim = self.q_lora_dim // self.block_size

        self.index_n_heads = self.model_args.index_n_heads
        self.index_head_dim = self.index_n_heads * self.model_args.index_head_dim
        self.index_head_qdim = self.index_head_dim // self.block_size

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16_for_pages(
        mat_in: torch.Tensor, q_lora_rank: int, pages: int
    ) -> torch.Tensor:
        """Swizzle a 16xK matrix for the paged weight layout (K divisible by 16)."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == q_lora_rank
        pre_shape = mat_in.shape[:-2]
        k_per_page = q_lora_rank // pages
        n_k_tiles = k_per_page // 16
        mat_in = mat_in.reshape(*pre_shape, 16, pages, k_per_page).transpose(-3, -2)
        mat_in = mat_in.reshape(*pre_shape, pages, 16, n_k_tiles, 16).transpose(-3, -2)
        mat_in = RmsnormProjqWqiWeightsConverter._swizzle_mma_16x16(mat_in)
        return mat_in.contiguous()

    def _common_to_tilert_fp16mma(
        self,
        wqi: torch.Tensor,
        wqi_scales: torch.Tensor,
        rmsnorm_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common weights to the packed TileRT FP16 layout (IQ only)."""
        sms = 128
        k_per_page = 1024 if self.model_args.arch_name == "glm_5" else 512
        pages = self.q_lora_dim // k_per_page
        iq_dim_per_sm = self.index_head_dim // sms

        wqi_scales_f32 = wqi_scales.to(torch.float32)
        wqi_scales_f32 = (
            wqi_scales_f32.reshape(self.index_head_qdim, 1, self.q_lora_qdim)
            .repeat(1, self.block_size, 1)
            .reshape(self.index_head_dim, self.q_lora_qdim)
        )
        wqi_scales_f32 = wqi_scales_f32.reshape(
            sms, iq_dim_per_sm, pages, self.q_lora_qdim // pages
        ).transpose(1, 2)
        wqi_scales_f32 = wqi_scales_f32[:, :, 0, :]
        wqi_full_scales = wqi_scales_f32.contiguous().view(torch.float8_e4m3fn)

        wqi_mat = wqi.reshape(sms, iq_dim_per_sm // 16, 16, self.q_lora_dim)
        wqi_mat = RmsnormProjqWqiWeightsConverter._swizzle_mma_16x16_for_pages(
            wqi_mat, self.q_lora_dim, pages
        )
        wqi_mat = (
            wqi_mat.reshape(sms, iq_dim_per_sm // 16, pages, 16, -1)
            .transpose(1, 2)
            .reshape(sms, pages, iq_dim_per_sm, -1)
        )
        wqi_mat = wqi_mat.reshape(sms, pages, -1)

        wqi_scales_padding = torch.zeros(
            sms,
            pages,
            128 - wqi_full_scales.shape[-1],
            dtype=torch.float8_e4m3fn,
            device=wqi.device,
        )
        tilert_wqi = torch.cat([wqi_mat, wqi_full_scales, wqi_scales_padding], dim=-1).contiguous()
        tilert_wqi_scales = torch.zeros(1, dtype=torch.bfloat16)
        tilert_gamma = rmsnorm_gamma.float().detach().clone()
        return tilert_wqi, tilert_wqi_scales, tilert_gamma

    def convert_to_fp16mma(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert common-format weights to TileRT FP16 MMA layout.

        Args:
            weights: [wqi, wqi_scale, q_norm_weight].
        """
        with torch.inference_mode():
            wqi, wqi_scale, q_norm_weight = weights
            return self._common_to_tilert_fp16mma(wqi, wqi_scale, q_norm_weight)


@dataclass
class RmsnormProjqWqiRefWeightsAlias:
    """Reference (HuggingFace) weights alias for RmsnormProjqWqi."""

    rmsnorm_gamma = "self_attn.q_a_layernorm.weight"
    wqi_weights = "self_attn.indexer.wq_b.weight"
    wqi_scales = "self_attn.indexer.wq_b.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.rmsnorm_gamma, self.wqi_weights, self.wqi_scales]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class RmsnormProjqWqiTilertWeightsAlias:
    """TileRT weights alias for RmsnormProjqWqi."""

    rmsnorm_gamma = "q_rmsnorm_gamma_qi"
    wqi_weights = "wqi_weights"
    wqi_scales = "wqi_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.rmsnorm_gamma, self.wqi_weights, self.wqi_scales]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RmsnormProjqWqi(TileRTModule):
    """RmsnormProjqWqi module: RMSNorm + W_qi projection (IQ only, GLM5 v2)."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [RmsnormProjqWqiAlgorithm.FP16MMA],
        "glm_5": [RmsnormProjqWqiAlgorithm.FP16MMA],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.tilert_weights_alias = RmsnormProjqWqiTilertWeightsAlias()
        self.ref_weights_alias = RmsnormProjqWqiRefWeightsAlias()

        self.q_lora_rank = model_args.q_lora_rank
        self.index_n_heads = model_args.index_n_heads
        self.head_dim = model_args.index_head_dim
        self.index_head_dim = model_args.index_n_heads * model_args.index_head_dim

        self.block_size = model_args.block_size
        self.q_lora_qdim = self.q_lora_rank // self.block_size
        self.index_head_qdim = self.index_head_dim // self.block_size
        self.eps = model_args.eps

        self.ref_q_norm: torch.Tensor | None = None
        self.ref_wqi: torch.Tensor | None = None

        self.tilert_wqi: torch.Tensor | None = None
        self.tilert_wqi_scales: torch.Tensor | None = None
        self.tilert_q_norm_weight: torch.Tensor | None = None

        self.iq: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_q_norm_weight, self.tilert_wqi, self.tilert_wqi_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replicate IQ weights across devices (no per-head redistribution needed)."""
        gamma = (
            weights_map[self.ref_weights_alias.rmsnorm_gamma][None, ...]
            .float()
            .repeat(self.num_devices, 1)
        )
        wqi_weights = weights_map[self.ref_weights_alias.wqi_weights][None, ...].repeat(
            self.num_devices, 1, 1
        )
        wqi_scales = weights_map[self.ref_weights_alias.wqi_scales][None, ...].repeat(
            self.num_devices, 1, 1
        )
        return {
            self.tilert_weights_alias.rmsnorm_gamma: gamma,
            self.tilert_weights_alias.wqi_weights: wqi_weights,
            self.tilert_weights_alias.wqi_scales: wqi_scales,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize reference weights from common-format state dict."""
        self.ref_q_norm = state_dict[self.tilert_weights_alias.rmsnorm_gamma]
        wqi = weight_dequant(
            state_dict[self.tilert_weights_alias.wqi_weights],
            state_dict[self.tilert_weights_alias.wqi_scales],
        )
        self.ref_wqi = wqi.contiguous()

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize TileRT weights from common-format state dict."""
        weights = [
            state_dict[self.tilert_weights_alias.wqi_weights],
            state_dict[self.tilert_weights_alias.wqi_scales],
            state_dict[self.tilert_weights_alias.rmsnorm_gamma],
        ]
        assert self.algorithm is not None, "Algorithm is not set"
        self.tilert_wqi, self.tilert_wqi_scales, self.tilert_q_norm_weight = (
            RmsnormProjqWqiWeightsConverter(self.model_args, self.num_devices).dispatch(
                self.algorithm, weights
            )
        )

    def init_random_weights(self) -> None:
        """Initialize random reference and TileRT weights for testing."""
        q_norm = torch.randn(self.q_lora_rank, dtype=torch.float32)
        wqi = torch.randn(self.index_head_dim, self.q_lora_rank, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        scale_dtype = torch.float32 if self.model_args.arch_name == "glm_5" else torch.bfloat16
        wqi_scale = torch.randn(self.index_head_qdim, self.q_lora_qdim, dtype=scale_dtype)

        ref_state = {
            self.tilert_weights_alias.rmsnorm_gamma: q_norm,
            self.tilert_weights_alias.wqi_weights: wqi,
            self.tilert_weights_alias.wqi_scales: wqi_scale,
        }

        self.init_reference_weights(ref_state)
        self.init_tilert_weights(ref_state)

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """Allocate TileRT output buffers."""
        self.iq = torch.zeros(
            batch_size, seq_len, self.index_n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, q: torch.Tensor) -> torch.Tensor:
        """Reference forward: RMSNorm + W_qi_b linear projection."""
        assert self.ref_q_norm is not None
        assert self.ref_wqi is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4, 8]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        qr = torch.nn.functional.rms_norm(q.float(), [q.size(-1)], self.ref_q_norm, self.eps).to(
            q.dtype
        )

        return rearrange(torch.matmul(qr, self.ref_wqi.T), "b s (h d) -> b s h d", d=self.head_dim)

    def tilert_forward(self, q: torch.Tensor) -> torch.Tensor:
        assert self.tilert_wqi is not None
        assert self.tilert_wqi_scales is not None
        assert self.tilert_q_norm_weight is not None
        assert self.iq is not None
        assert self.profile_logs is not None

        bsz, seqlen, _ = q.shape
        if bsz != 1 or seqlen not in [1, 2, 4, 8]:
            raise ValueError(f"Invalid batch size or sequence length: bsz={bsz}, seqlen={seqlen}")

        assert self.algorithm is not None, "Algorithm is not set"

        rmsnorm_projq_wqi_op(
            q,
            self.tilert_wqi,
            self.tilert_wqi_scales,
            self.tilert_q_norm_weight,
            self.iq,
            self.profile_logs,
            self.algorithm.value,
            model_arch=self.model_args.arch_name,
        )

        return self.iq
