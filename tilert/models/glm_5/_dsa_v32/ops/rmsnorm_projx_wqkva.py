"""RMSNormProjxWqkva operation module."""

from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormProjxWqkva",
    "RMSNormProjxWqkvaAlgorithm",
]


class RMSNormProjQKVAFP8MMAWeightsConverter:
    """Weight converter: pack FP8 weights into WqkvaPagedShared layout for the FP8 MMA kernel."""

    HIDDEN_DIM = 6144
    Q_LORA_RANK = 2048
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    TOTAL_ROWS = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM
    ROWS_PER_CTA = 32
    NUM_CTAS = TOTAL_ROWS // ROWS_PER_CTA
    COLS_PER_PAGE = 1024
    NUM_PAGES = HIDDEN_DIM // COLS_PER_PAGE
    SCALES_PER_PAGE = COLS_PER_PAGE // 128
    BLOCK_SIZE = 128

    MAT_BYTES = ROWS_PER_CTA * COLS_PER_PAGE
    SCALE_OFFSET = MAT_BYTES

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def convert_to_fp8_mma_gemv(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        w_pe: torch.Tensor,
        w_pe_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
        *,
        hidden_dim: int = 6144,
        q_lora_rank: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = RMSNormProjQKVAFP8MMAWeightsConverter
        kv_lora_rank = C.KV_LORA_RANK
        qk_rope_head_dim = C.QK_ROPE_HEAD_DIM

        expected = q_lora_rank * hidden_dim
        assert wq_a.numel() == expected, f"wq_a numel {wq_a.numel()} != expected {expected}"
        expected = kv_lora_rank * hidden_dim
        assert wkv_a.numel() == expected, f"wkv_a numel {wkv_a.numel()} != expected {expected}"
        expected = qk_rope_head_dim * hidden_dim
        assert w_pe.numel() == expected, f"w_pe numel {w_pe.numel()} != expected {expected}"

        return C._pack_per_row_no_requant(
            wq_a,
            wkv_a,
            w_pe,
            wq_a_scale,
            wkv_a_scale,
            w_pe_scale,
            attn_norm_weight,
            hidden_dim=hidden_dim,
            q_lora_rank=q_lora_rank,
        )

    @staticmethod
    def _pack_per_row_no_requant(
        wq_a: torch.Tensor,
        wkv_a: torch.Tensor,
        w_pe: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        w_pe_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
        *,
        hidden_dim: int,
        q_lora_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = RMSNormProjQKVAFP8MMAWeightsConverter
        kv_lora_rank = C.KV_LORA_RANK
        qk_rope_head_dim = C.QK_ROPE_HEAD_DIM
        total_rows = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        num_ctas = total_rows // C.ROWS_PER_CTA
        num_pages = hidden_dim // C.COLS_PER_PAGE

        num_tiles = C.COLS_PER_PAGE // 32
        blk = C.BLOCK_SIZE
        num_blk_page = C.COLS_PER_PAGE // blk
        num_blk_total = hidden_dim // blk

        w_fp8 = torch.cat(
            [
                wq_a.reshape(q_lora_rank, hidden_dim),
                wkv_a.reshape(kv_lora_rank, hidden_dim),
                w_pe.reshape(qk_rope_head_dim, hidden_dim),
            ],
            dim=0,
        ).contiguous()

        def _bcast_block_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
            s = scale.to(torch.float32).reshape(-1, num_blk_total)
            return s.repeat_interleave(blk, dim=0)[:rows]

        w_scales = torch.cat(
            [
                _bcast_block_scale(wq_a_scale, q_lora_rank),
                _bcast_block_scale(wkv_a_scale, kv_lora_rank),
                _bcast_block_scale(w_pe_scale, qk_rope_head_dim),
            ],
            dim=0,
        ).clamp(min=1e-12)

        assert C.MAT_BYTES == C.SCALE_OFFSET, "Layout mismatch: scales must follow mat"

        w_bytes = w_fp8.view(torch.uint8)
        mat = w_bytes.reshape(num_ctas, C.ROWS_PER_CTA, num_pages, C.COLS_PER_PAGE)
        mat = mat.transpose(1, 2)
        mat = mat.reshape(num_ctas, num_pages, 2, 16, num_tiles, 32)
        mat = mat.transpose(3, 4)
        mat = C._swizzle_mma_16x32(mat)
        mat = mat.contiguous().reshape(num_ctas, num_pages, C.MAT_BYTES)

        sc = w_scales.reshape(num_ctas, C.ROWS_PER_CTA, num_pages, num_blk_page)
        sc = sc.permute(0, 2, 1, 3).contiguous()
        scales_bytes = (
            sc.to(torch.float32)
            .reshape(num_ctas, num_pages, C.ROWS_PER_CTA * num_blk_page)
            .view(torch.uint8)
        )
        packed = torch.cat([mat, scales_bytes], dim=-1)
        packed = packed.contiguous().reshape(-1)

        return packed.view(torch.float8_e4m3fn), attn_norm_weight.clone()


class RMSNormProjQKVAFP16MMAWeightsConverter:
    """Weight converter: pack FP16 weights for the MMA kernel."""

    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    ROWS_PER_CTA = 32
    COLS_PER_PAGE = 512
    BLOCK_SIZE = 128

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def convert_to_fp16_mma_gemv(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        w_pe: torch.Tensor,
        w_pe_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
        *,
        hidden_dim: int = 6144,
        q_lora_rank: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = RMSNormProjQKVAFP16MMAWeightsConverter
        kv_lora_rank = C.KV_LORA_RANK
        qk_rope_head_dim = C.QK_ROPE_HEAD_DIM
        cols_per_page = C.COLS_PER_PAGE
        rows_per_cta = C.ROWS_PER_CTA

        total_rows = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        num_ctas = total_rows // rows_per_cta
        num_pages = hidden_dim // cols_per_page
        num_k_tiles = cols_per_page // 16

        wq_a_f = weight_dequant(wq_a.reshape(q_lora_rank, hidden_dim), wq_a_scale)
        wkv_a_f = weight_dequant(wkv_a.reshape(kv_lora_rank, hidden_dim), wkv_a_scale)
        w_pe_f = weight_dequant(w_pe.reshape(qk_rope_head_dim, hidden_dim), w_pe_scale)
        w_float = torch.cat([wq_a_f, wkv_a_f, w_pe_f], dim=0)

        w_fp16 = w_float.to(torch.float16)

        mat = w_fp16.reshape(num_ctas, rows_per_cta, num_pages, cols_per_page)
        mat = mat.transpose(1, 2)

        mat = mat.reshape(num_ctas, num_pages, 2, 16, num_k_tiles, 16)
        mat = mat.transpose(3, 4)
        mat = C._swizzle_mma_16x16(mat)
        mat = mat.contiguous()

        mat_bytes = mat.view(torch.uint8).reshape(num_ctas, num_pages, -1)
        packed = mat_bytes.contiguous().reshape(-1)

        return packed.view(torch.float16), attn_norm_weight.clone()


class RMSNormProjQKVAW8A16MMAWeightsConverter:
    """Pack FP8 weight + block scale into packed format."""

    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    ROWS_PER_CTA = 32
    COLS_PER_PAGE = 1024
    BLOCK_SIZE = 128
    NUM_WARPS = 8
    MMA_K = 16
    M_TILES_PER_CTA = ROWS_PER_CTA // 16
    K_TILES_PER_WARP = COLS_PER_PAGE // (NUM_WARPS * MMA_K)
    SCALES_PER_PAGE = COLS_PER_PAGE // BLOCK_SIZE
    PAGE_MAT_BYTES = M_TILES_PER_CTA * K_TILES_PER_WARP * NUM_WARPS * 32 * 8
    PAGE_BYTES = PAGE_MAT_BYTES + 128

    @staticmethod
    def _permute_mma_a_fragment_16x16(tile: torch.Tensor) -> torch.Tensor:
        assert tile.shape[-2:] == (16, 16)
        pre = tile.shape[:-2]
        return (
            tile.reshape(*pre, 2, 8, 2, 4, 2)
            .permute(
                *range(len(pre)),
                len(pre) + 1,
                len(pre) + 3,
                len(pre) + 2,
                len(pre) + 0,
                len(pre) + 4,
            )
            .contiguous()
            .reshape(*pre, 32, 8)
        )

    @staticmethod
    def convert_to_w8a16_mma_gemv(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        w_pe: torch.Tensor,
        w_pe_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
        *,
        hidden_dim: int = 6144,
        q_lora_rank: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = RMSNormProjQKVAW8A16MMAWeightsConverter
        kv_lora_rank = C.KV_LORA_RANK
        qk_rope_head_dim = C.QK_ROPE_HEAD_DIM
        rows_per_cta = C.ROWS_PER_CTA
        cols_per_page = C.COLS_PER_PAGE

        w_fp8 = torch.cat(
            [
                wq_a.reshape(q_lora_rank, hidden_dim),
                wkv_a.reshape(kv_lora_rank, hidden_dim),
                w_pe.reshape(qk_rope_head_dim, hidden_dim),
            ],
            dim=0,
        ).contiguous()
        assert w_fp8.dtype == torch.float8_e4m3fn, f"expected fp8 weight, got {w_fp8.dtype}"

        scales = (
            torch.cat([wq_a_scale, wkv_a_scale, w_pe_scale], dim=0).to(torch.float32).contiguous()
        )

        total_rows = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        num_ctas = total_rows // rows_per_cta
        num_pages = hidden_dim // cols_per_page
        expected_scale_rows = (total_rows + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
        assert scales.shape == (expected_scale_rows, hidden_dim // C.BLOCK_SIZE), (
            f"scales {tuple(scales.shape)} != "
            f"{(expected_scale_rows, hidden_dim // C.BLOCK_SIZE)}"
        )

        del num_ctas, num_pages
        return C.pack_lane_major(w_fp8, scales, hidden_dim), attn_norm_weight.clone()

    @classmethod
    def pack_lane_major(
        cls, w_fp8: torch.Tensor, scales: torch.Tensor, hidden_dim: int
    ) -> torch.Tensor:
        assert w_fp8.dtype == torch.float8_e4m3fn, f"expected fp8 weight, got {w_fp8.dtype}"
        rows_per_cta = cls.ROWS_PER_CTA
        cols_per_page = cls.COLS_PER_PAGE
        total_rows = w_fp8.shape[0]
        num_ctas = total_rows // rows_per_cta
        num_pages = hidden_dim // cols_per_page
        scales = scales.to(torch.float32).contiguous()
        device = w_fp8.device
        w_bytes = w_fp8.view(torch.uint8)

        w = w_bytes.reshape(num_ctas, rows_per_cta, num_pages, cols_per_page)
        w = w.reshape(num_ctas, cls.M_TILES_PER_CTA, 16, num_pages, cols_per_page)
        w = w.permute(0, 3, 1, 2, 4).contiguous()
        w = w.reshape(
            num_ctas, num_pages, cls.M_TILES_PER_CTA, 16, cls.NUM_WARPS, cls.K_TILES_PER_WARP, 16
        )
        w = w.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
        w_lane = cls._permute_mma_a_fragment_16x16(w)
        mat_blob = w_lane.contiguous().reshape(num_ctas, num_pages, cls.PAGE_MAT_BYTES)

        cta_idx = torch.arange(num_ctas, device=device)
        scale_row = cta_idx // (cls.BLOCK_SIZE // rows_per_cta)
        cta_scales = scales[scale_row].reshape(num_ctas, num_pages, cls.SCALES_PER_PAGE)
        scale_bytes = cta_scales.contiguous().view(torch.uint8)

        out = torch.zeros(num_ctas, num_pages, cls.PAGE_BYTES, dtype=torch.uint8, device=device)
        out[:, :, : cls.PAGE_MAT_BYTES] = mat_blob
        out[:, :, cls.PAGE_MAT_BYTES : cls.PAGE_MAT_BYTES + cls.SCALES_PER_PAGE * 4] = scale_bytes
        return out.reshape(-1).contiguous().view(torch.float8_e4m3fn)


class RMSNormProjxWqkvaAlgorithm(Enum):
    """RMSNormProjxWqkva algorithm."""

    DECOUPLED = "decoupled"
    W8A16HMMA = "w8a16_hmma"


class RMSNormProjxWqkvaWeightsConverter(TilertWeightsConverter):
    """Dispatch weight converter for RMSNormProjxWqkva."""

    def __init__(self, model_args: ModelArgs, num_devices: int):
        super().__init__(model_args, num_devices)

    def convert_to_fp8_mma_gemv(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, wq_a, wq_a_scale, wkv_a, wkv_a_scale, w_pe, w_pe_scale = weights
        return RMSNormProjQKVAFP8MMAWeightsConverter.convert_to_fp8_mma_gemv(
            wq_a,
            wq_a_scale,
            wkv_a,
            wkv_a_scale,
            w_pe,
            w_pe_scale,
            gamma,
            hidden_dim=self.model_args.dim,
            q_lora_rank=self.model_args.q_lora_rank,
        )

    def convert_to_fp16_mma_gemv(
        self, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, wq_a, wq_a_scale, wkv_a, wkv_a_scale, w_pe, w_pe_scale = weights
        return RMSNormProjQKVAFP16MMAWeightsConverter.convert_to_fp16_mma_gemv(
            wq_a,
            wq_a_scale,
            wkv_a,
            wkv_a_scale,
            w_pe,
            w_pe_scale,
            gamma,
            hidden_dim=self.model_args.dim,
            q_lora_rank=self.model_args.q_lora_rank,
        )


class RMSNormProjxWqkvaRefWeightsAlias:
    """Reference weight aliases for RMSNormProjxWqkva."""

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


class RMSNormProjxWqkvaTilertWeightsAlias:
    """Tilert weight aliases for RMSNormProjxWqkva."""

    x_rmsnorm_gamma = "x_rmsnorm_gamma"
    q_a_weights = "q_a_weights"
    q_a_scales = "q_a_scales"
    kv_a_weights = "kv_a_weights"
    kv_a_scales = "kv_a_scales"
    w_pe_weights = "w_pe_weights"
    w_pe_scales = "w_pe_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.x_rmsnorm_gamma,
            self.q_a_weights,
            self.q_a_scales,
            self.kv_a_weights,
            self.kv_a_scales,
            self.w_pe_weights,
            self.w_pe_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class RMSNormProjxWqkva(TileRTModule):
    """Fused RMSNorm + GEMV(W_q_a, W_kv_a, W_pe) for Device Group B."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [
            RMSNormProjxWqkvaAlgorithm.DECOUPLED,
            RMSNormProjxWqkvaAlgorithm.W8A16HMMA,
        ],
        "glm_5": [
            RMSNormProjxWqkvaAlgorithm.DECOUPLED,
            RMSNormProjxWqkvaAlgorithm.W8A16HMMA,
        ],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: RMSNormProjxWqkvaRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = RMSNormProjxWqkvaTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RMSNormProjxWqkvaRefWeightsAlias()
        )

        self.dim = self.model_args.dim
        self.q_lora_rank = self.model_args.q_lora_rank
        self.kv_lora_rank = self.model_args.kv_lora_rank
        self.qk_rope_head_dim = self.model_args.qk_rope_head_dim
        self.block_size = self.model_args.block_size
        self.eps = self.model_args.eps
        self.algorithm = RMSNormProjxWqkvaAlgorithm.DECOUPLED

        self.ref_norm_gamma: torch.Tensor | None = None
        self.ref_wq_a: torch.Tensor | None = None
        self.ref_wkv_a: torch.Tensor | None = None
        self.ref_w_pe: torch.Tensor | None = None

        self.tilert_norm_gamma: torch.Tensor | None = None
        self.tilert_wqkva: torch.Tensor | None = None
        self.tilert_wqkva_scales = torch.zeros((1, 1), dtype=torch.bfloat16)

        self.x_rmsnorm_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_out: torch.Tensor | None = None
        self.x_rmsnorm_quant_scale_out: torch.Tensor | None = None

        self.q_out: torch.Tensor | None = None
        self.kv_out: torch.Tensor | None = None
        self.pe_cache_out: torch.Tensor | None = None
        self.cur_pos: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self.tilert_tensor_alias: list[str] = [
            "x_rmsnorm_gamma",
            "qkva_weights",
            "qkva_scales",
        ]

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_norm_gamma, self.tilert_wqkva, self.tilert_wqkva_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_layernorm_weight = (
            weights_map[self.ref_weights_alias.x_rmsnorm_gamma][None, ...]
            .float()
            .repeat(self.num_devices, 1)
        )
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
            self.tilert_weights_alias.x_rmsnorm_gamma: input_layernorm_weight,
            self.tilert_weights_alias.q_a_weights: q_a_proj_weight,
            self.tilert_weights_alias.q_a_scales: q_a_proj_weight_scale,
            self.tilert_weights_alias.kv_a_weights: kv_a_proj_weight,
            self.tilert_weights_alias.kv_a_scales: kv_a_proj_weight_scale,
            self.tilert_weights_alias.w_pe_weights: w_pe_weight,
            self.tilert_weights_alias.w_pe_scales: w_pe_weight_scale,
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        aliases = self.ref_weights_alias()
        self.ref_norm_gamma = state_dict[aliases[0]]
        self.ref_wq_a = weight_dequant(state_dict[aliases[1]], state_dict[aliases[2]])
        kv_a_mqa = weight_dequant(state_dict[aliases[3]], state_dict[aliases[4]])
        self.ref_wkv_a = kv_a_mqa[: self.kv_lora_rank, :]
        self.ref_w_pe = kv_a_mqa[self.kv_lora_rank :, :]

        assert self.ref_norm_gamma.shape[-1] == self.dim
        assert self.ref_wq_a.shape == (self.q_lora_rank, self.dim)
        assert self.ref_wkv_a.shape == (self.kv_lora_rank, self.dim)
        assert self.ref_w_pe.shape == (self.qk_rope_head_dim, self.dim)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        tilert_aliases = self.tilert_weights_alias()
        weights_list = [state_dict[alias] for alias in tilert_aliases]
        if self.algorithm == RMSNormProjxWqkvaAlgorithm.W8A16HMMA:
            gamma, wq_a, wq_a_scale, wkv_a, wkv_a_scale, w_pe, w_pe_scale = weights_list
            self.tilert_wqkva, self.tilert_norm_gamma = (
                RMSNormProjQKVAW8A16MMAWeightsConverter.convert_to_w8a16_mma_gemv(
                    wq_a,
                    wq_a_scale,
                    wkv_a,
                    wkv_a_scale,
                    w_pe,
                    w_pe_scale,
                    gamma.float(),
                    hidden_dim=self.dim,
                    q_lora_rank=self.q_lora_rank,
                )
            )
            self.tilert_norm_gamma = self.tilert_norm_gamma.float().contiguous()
        else:
            converter = RMSNormProjxWqkvaWeightsConverter(self.model_args, self.num_devices)
            self.tilert_wqkva, self.tilert_norm_gamma = converter.convert_to_fp8_mma_gemv(
                weights_list
            )
        self.tilert_wqkva_scales = torch.zeros((1,), dtype=torch.float32)

    def init_tilert_vars(self, batch_size: int, seq_len: int, max_len: int = 128) -> None:
        self.q_out = torch.zeros((batch_size, seq_len, self.q_lora_rank), dtype=torch.bfloat16)
        self.kv_out = torch.zeros((batch_size, seq_len, self.kv_lora_rank), dtype=torch.bfloat16)
        self.pe_cache_out = torch.zeros(
            (batch_size, max_len, self.qk_rope_head_dim), dtype=torch.bfloat16
        )
        self.cur_pos = torch.zeros((1,), dtype=torch.int32)
        self.x_rmsnorm_out = torch.zeros((batch_size, seq_len, self.dim), dtype=torch.bfloat16)
        self.x_rmsnorm_quant_out = torch.zeros(
            (batch_size, seq_len, self.dim), dtype=torch.float8_e4m3fn
        )
        self.x_rmsnorm_quant_scale_out = torch.zeros(
            (batch_size, seq_len, self.dim // self.block_size), dtype=torch.float32
        )
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
            torch.randn(q_scale_dim, dim_scale_dim, dtype=scale_dtype).abs(),
            torch.randn(kv_mqa_rows, self.dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            torch.randn(kv_mqa_scale_dim, dim_scale_dim, dtype=scale_dtype).abs(),
        ]
        ref_state_dict = dict(zip(self.ref_weights_alias(), tensor_list))
        self.init_reference_weights(ref_state_dict)
        self.init_tilert_weights(
            {_k: _v[self.device_id] for _k, _v in self.device_sharding(ref_state_dict).items()}
        )

    def golden_forward(
        self,
        x: torch.Tensor,
        cur_pos: int = 0,  # noqa: U100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch reference: RMSNorm -> q, kv, pe."""
        assert self.ref_norm_gamma is not None
        assert self.ref_wq_a is not None
        assert self.ref_wkv_a is not None
        assert self.ref_w_pe is not None

        x_rmsnorm = torch.nn.functional.rms_norm(
            x.float(), [x.size(-1)], self.ref_norm_gamma, self.eps
        )
        q_out = torch.matmul(x_rmsnorm.float(), self.ref_wq_a.transpose(0, 1).float())
        kv_out = torch.matmul(x_rmsnorm.float(), self.ref_wkv_a.transpose(0, 1).float())
        pe_out = torch.matmul(x_rmsnorm.float(), self.ref_w_pe.transpose(0, 1).float())
        return (
            q_out.to(torch.bfloat16),
            kv_out.to(torch.bfloat16),
            pe_out.to(torch.bfloat16),
        )

    def tilert_forward(
        self,
        x: torch.Tensor,
        cur_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.cur_pos is not None
        assert self.pe_cache_out is not None
        self.cur_pos.fill_(cur_pos)

        from tilert.models.glm_5._dsa_v32.ops.projx_wqkva import projx_wqkva as _projx_wqkva
        from tilert.models.glm_5._dsa_v32.ops.rmsnorm_quant import rmsnorm_quant as _rmsnorm_quant

        _rmsnorm_quant(
            x.to(torch.bfloat16),
            self.tilert_norm_gamma,
            self.x_rmsnorm_out,
            self.x_rmsnorm_quant_out,
            self.x_rmsnorm_quant_scale_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        _projx_wqkva(
            self.x_rmsnorm_quant_out,
            self.x_rmsnorm_quant_scale_out,
            self.tilert_wqkva,
            self.cur_pos,
            self.q_out,
            self.kv_out,
            self.pe_cache_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )

        seq_len = x.size(-2)
        pe_at_pos = self.pe_cache_out[:, cur_pos : cur_pos + seq_len, :]
        return self.q_out, self.kv_out, pe_at_pos

    def __call__(
        self,
        x: torch.Tensor,
        cur_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.golden_forward(x, cur_pos)
