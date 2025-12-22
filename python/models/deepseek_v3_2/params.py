from abc import abstractmethod

import torch

from tilert.models.utils import SwizzleMode, gen_tensor_swizzle_map_1d

__all__ = [
    "IntermediateMapper",
    "BaseParams",
    "MlaParams",
    "MLPParams",
    "MoEParams",
    "TempVars",
    "DenseLayerParamsKeys",
    "MoELayerParamsKeys",
    "CacheVars",
    "gen_down_allreduce_fp8_params",
]


DenseLayerParamsKeys = [
    # MLA params
    "x_rmsnorm_gamma",  # 0
    "qkv_wa_weights",  # 1
    "qkv_wa_scales",  # 2
    "k_weights",  # 3
    "k_bias",  # 4
    "q_rmsnorm_gamma",  # 5
    "q_wb_weights",  # 6
    "q_wb_scales",  # 7
    "id_score_weights",  # 8
    "wkv_b1_weights",  # 9
    "wkv_b1_scales",  # 10
    "kv_rmsnorm_gamma",  # 11
    "wkv_b2_weights",  # 12
    "wkv_b2_scales",  # 13
    "unproj_weights",  # 14
    "unproj_scales",  # 15
    # MLP params
    "unproj_o_gamma",  # 16
    "upgate_weights",  # 17
    "upgate_scales",  # 18
    "down_weights",  # 19
    "down_scales",  # 20
]

MoELayerParamsKeys = [
    # MLA params
    "x_rmsnorm_gamma",  # 0
    "qkv_wa_weights",  # 1
    "qkv_wa_scales",  # 2
    "k_weights",  # 3
    "k_bias",  # 4
    "q_rmsnorm_gamma",  # 5
    "q_wb_weights",  # 6
    "q_wb_scales",  # 7
    "id_score_weights",  # 8
    "wkv_b1_weights",  # 9
    "wkv_b1_scales",  # 10
    "kv_rmsnorm_gamma",  # 11
    "wkv_b2_weights",  # 12
    "wkv_b2_scales",  # 13
    "unproj_weights",  # 14
    "unproj_scales",  # 15
    # MoE params
    "unproj_o_gamma",  # 16
    "exp_proj_weights",  # 17
    "exp_bias",  # 18
    "exp_upgate_weights",  # 19
    "exp_upgate_scales",  # 20
    "exp_down_weights",  # 21
    "exp_down_scales",  # 22
]


def gen_down_allreduce_fp8_params(mat_in: torch.Tensor, mat_scale_in: torch.Tensor) -> torch.Tensor:
    """Convert tilert mat and scale to tilert-fp8 input format."""
    mat_and_scale_in = torch.zeros(
        (9, 128, (56 * 256 + 64)), dtype=torch.float8_e4m3fn, device=mat_in.device
    )
    scale_part = mat_and_scale_in[..., 56 * 256 :]
    mat_part = mat_and_scale_in[..., : 56 * 256].reshape(9, 128, 2, 56 * 8, 16)
    mat_in = mat_in.reshape(9, 128, 56, 2, 8, 16)
    mat_in = mat_in.transpose(2, 3).reshape(9, 128, 2, 56 * 8, 16)

    swizzle_map = gen_tensor_swizzle_map_1d(56, 8, SwizzleMode.SWIZZLE_128B)
    mat_part[:, :, :, swizzle_map] = mat_in

    # copy mat_scale_in to scale_part
    # scale to fp32
    mat_scale_in_fp32 = mat_scale_in.to(torch.float32).reshape(9, 128, 16)  # 7x2 + 2 zeros
    scale_part.copy_(mat_scale_in_fp32.view(dtype=torch.float8_e4m3fn))
    return mat_and_scale_in


def gen_expert_down_allreduce_fp8_params(
    mat_in: torch.Tensor, mat_scale_in: torch.Tensor
) -> torch.Tensor:
    """Convert tilert mat and scale to tilert-fp8 input format."""
    mat_and_scale_in = torch.zeros(
        (257, 128, (56 * 256 + 64)), dtype=torch.float8_e4m3fn, device=mat_in.device
    )
    scale_part = mat_and_scale_in[..., 56 * 256 :]
    mat_part = mat_and_scale_in[..., : 56 * 256].reshape(257, 128, 2, 56 * 8, 16)
    mat_in = mat_in.reshape(257, 128, 56, 2, 8, 16)
    mat_in = mat_in.transpose(2, 3).reshape(257, 128, 2, 56 * 8, 16)

    swizzle_map = gen_tensor_swizzle_map_1d(56, 8, SwizzleMode.SWIZZLE_128B)
    mat_part[:, :, :, swizzle_map] = mat_in

    # copy mat_scale_in to scale_part
    # scale to fp32
    mat_scale_in_fp32 = mat_scale_in.to(torch.float32).reshape(257, 128, 16)  # 7x2 + 2 zeros
    scale_part.copy_(mat_scale_in_fp32.view(dtype=torch.float8_e4m3fn))
    return mat_and_scale_in


def gen_unproj_o_allreduce_fp8_params(
    mat_in: torch.Tensor, mat_scale_in: torch.Tensor
) -> torch.Tensor:
    """Convert tilert mat and scale to tilert-fp8 input format."""
    mat_and_scale_in = torch.zeros(
        (128, 4, (56 * 512 + 8 * 4 * 4)), dtype=torch.float8_e4m3fn, device=mat_in.device
    )
    scale_part = mat_and_scale_in[..., 56 * 512 :].reshape(128, 4, 128)
    mat_part = mat_and_scale_in[..., : 56 * 512].reshape(128, 4, 4, 56 * 8, 16)

    mat_in = mat_in.reshape(128, 56, 4, 512)
    mat_in = (
        mat_in.transpose(1, 2)
        .reshape(128, 4, 56, 4, 128)
        .transpose(2, 3)
        .reshape(128, 4, 4, 56 * 8, 16)
    )

    swizzle_map = gen_tensor_swizzle_map_1d(56, 8, SwizzleMode.SWIZZLE_128B)
    mat_part[:, :, :, swizzle_map] = mat_in
    # 896x16
    mat_scale_in_fp32 = mat_scale_in.to(torch.float32).reshape(128, 7, 16)
    # padding to 1024x16
    zeros = torch.zeros((128, 1, 16), dtype=torch.float32, device=mat_scale_in.device)
    mat_scale_in_fp32 = torch.cat([mat_scale_in_fp32, zeros], dim=1).contiguous()
    # transpose
    mat_scale_in_fp32 = mat_scale_in_fp32.reshape(128, 8, 4, 4).transpose(1, 2).contiguous()
    scale_part.copy_(mat_scale_in_fp32.view(dtype=torch.float8_e4m3fn).reshape(128, 4, 128))

    return mat_and_scale_in.contiguous()


class IntermediateMapper:
    """Map the intermediate tensors to the corresponding variables."""

    def __init__(self, intermediate_list: list[torch.Tensor]):
        self.q = intermediate_list[0]
        self.kv = intermediate_list[1]
        self.ki = intermediate_list[2]
        self.q_nope_down = intermediate_list[3]
        self.q_pe = intermediate_list[4]
        self.iq = intermediate_list[5]
        self.iq_rt = intermediate_list[6]
        self.idx_score = intermediate_list[7]
        self.idx_logits = intermediate_list[8]
        self.idx_sels = intermediate_list[9]
        self.q_nope = intermediate_list[10]
        self.o = intermediate_list[11]
        self.o_acc = intermediate_list[12]
        self.o_lse = intermediate_list[13]
        self.o_lse_acc = intermediate_list[14]
        self.proj_o = intermediate_list[15]
        self.unproj_o = intermediate_list[16]
        self.scores = intermediate_list[17]
        self.x_mlp_in = intermediate_list[18]
        self.exp_up_gate = intermediate_list[19]
        self.sel_probs = intermediate_list[20]
        self.sel_indices = intermediate_list[21]
        self.exp_out = intermediate_list[22]
        self.x_rmsnorm = intermediate_list[23]
        self.logits_out = intermediate_list[24]
        self.token_out = intermediate_list[25]


class BaseParams:
    def __init__(self) -> None:
        self._params: list[torch.Tensor] = []

    def register_params(self, param: torch.Tensor) -> torch.Tensor:
        self._params.append(param)
        return param

    def get_params(self) -> list[torch.Tensor]:
        return self._params

    @staticmethod
    @abstractmethod
    def num_params() -> int:
        raise NotImplementedError("Subclasses must implement this method")


class MlaParams(BaseParams):
    def __init__(
        self,
        x_rmsnorm_gamma: torch.Tensor,
        qkv_wa_weights: torch.Tensor,
        qkv_wa_scales: torch.Tensor,
        k_weights: torch.Tensor,
        k_bias: torch.Tensor,
        q_rmsnorm_gamma: torch.Tensor,
        q_wb_weights: torch.Tensor,
        q_wb_scales: torch.Tensor,
        id_score_weights: torch.Tensor,
        wkv_b1_weights: torch.Tensor,
        wkv_b1_scales: torch.Tensor,
        kv_rmsnorm_gamma: torch.Tensor,
        wkv_b2_weights: torch.Tensor,
        wkv_b2_scales: torch.Tensor,
        unproj_weights: torch.Tensor,
        unproj_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.x_rmsnorm_gamma = self.register_params(x_rmsnorm_gamma)
        self.qkv_wa_weights = self.register_params(qkv_wa_weights)
        self.qkv_wa_scales = self.register_params(qkv_wa_scales)
        self.k_weights = self.register_params(k_weights)
        self.k_bias = self.register_params(k_bias)
        self.q_rmsnorm_gamma = self.register_params(q_rmsnorm_gamma)
        self.q_wb_weights = self.register_params(q_wb_weights)
        self.q_wb_scales = self.register_params(q_wb_scales)
        self.id_score_weights = self.register_params(id_score_weights)
        self.wkv_b1_weights = self.register_params(wkv_b1_weights)
        self.wkv_b1_scales = self.register_params(wkv_b1_scales)
        self.kv_rmsnorm_gamma = self.register_params(kv_rmsnorm_gamma)
        self.wkv_b2_weights = self.register_params(wkv_b2_weights)
        self.wkv_b2_scales = self.register_params(wkv_b2_scales)
        self.unproj_weights = self.register_params(unproj_weights)
        self.unproj_scales = self.register_params(unproj_scales)

    @staticmethod
    def num_params() -> int:
        return 16

    def to_dict(self, layer_id: int, device_id: int) -> dict[str, torch.Tensor]:
        return {
            f"layer_{layer_id}_x_rmsnorm_gamma_dev_{device_id}": self.x_rmsnorm_gamma.to(device_id),
            f"layer_{layer_id}_qkv_wa_weights_dev_{device_id}": self.qkv_wa_weights.to(device_id),
            f"layer_{layer_id}_qkv_wa_scales_dev_{device_id}": self.qkv_wa_scales.to(device_id),
            f"layer_{layer_id}_k_weights_dev_{device_id}": self.k_weights.to(device_id),
            f"layer_{layer_id}_k_bias_dev_{device_id}": self.k_bias.to(device_id),
            f"layer_{layer_id}_q_rmsnorm_gamma_dev_{device_id}": self.q_rmsnorm_gamma.to(device_id),
            f"layer_{layer_id}_q_wb_weights_dev_{device_id}": self.q_wb_weights.to(device_id),
            f"layer_{layer_id}_q_wb_scales_dev_{device_id}": self.q_wb_scales.to(device_id),
            f"layer_{layer_id}_id_score_weights_dev_{device_id}": self.id_score_weights.to(
                device_id
            ),
            f"layer_{layer_id}_wkv_b1_weights_dev_{device_id}": self.wkv_b1_weights.to(device_id),
            f"layer_{layer_id}_wkv_b1_scales_dev_{device_id}": self.wkv_b1_scales.to(device_id),
            f"layer_{layer_id}_kv_rmsnorm_gamma_dev_{device_id}": self.kv_rmsnorm_gamma.to(
                device_id
            ),
            f"layer_{layer_id}_wkv_b2_weights_dev_{device_id}": self.wkv_b2_weights.to(device_id),
            f"layer_{layer_id}_wkv_b2_scales_dev_{device_id}": self.wkv_b2_scales.to(device_id),
            f"layer_{layer_id}_unproj_weights_dev_{device_id}": self.unproj_weights.to(device_id),
            f"layer_{layer_id}_unproj_scales_dev_{device_id}": self.unproj_scales.to(device_id),
        }


class MLPParams(BaseParams):
    def __init__(
        self,
        unproj_o_gamma: torch.Tensor,
        upgate_weights: torch.Tensor,
        upgate_scales: torch.Tensor,
        down_weights: torch.Tensor,
        down_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unproj_o_gamma = self.register_params(unproj_o_gamma)
        self.upgate_weights = self.register_params(upgate_weights)
        self.upgate_scales = self.register_params(upgate_scales)
        self.down_weights = self.register_params(down_weights)
        self.down_scales = self.register_params(down_scales)

    @staticmethod
    def num_params() -> int:
        return 5

    def to_dict(self, layer_id: int, device_id: int) -> dict[str, torch.Tensor]:
        return {
            f"layer_{layer_id}_unproj_o_gamma_dev_{device_id}": self.unproj_o_gamma.to(device_id),
            f"layer_{layer_id}_upgate_weights_dev_{device_id}": self.upgate_weights.to(device_id),
            f"layer_{layer_id}_upgate_scales_dev_{device_id}": self.upgate_scales.to(device_id),
            f"layer_{layer_id}_down_weights_dev_{device_id}": self.down_weights.to(device_id),
            f"layer_{layer_id}_down_scales_dev_{device_id}": self.down_scales.to(device_id),
        }


class MoEParams(BaseParams):
    def __init__(
        self,
        unproj_o_gamma: torch.Tensor,
        exp_proj_weights: torch.Tensor,
        exp_bias: torch.Tensor,
        exp_upgate_weights: torch.Tensor,
        exp_upgate_scales: torch.Tensor,
        exp_down_weights: torch.Tensor,
        exp_down_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unproj_o_gamma = self.register_params(unproj_o_gamma)
        self.exp_proj_weights = self.register_params(exp_proj_weights)
        self.exp_bias = self.register_params(exp_bias)
        self.exp_upgate_weights = self.register_params(exp_upgate_weights)
        self.exp_upgate_scales = self.register_params(exp_upgate_scales)
        self.exp_down_weights = self.register_params(exp_down_weights)
        self.exp_down_scales = self.register_params(exp_down_scales)

    @staticmethod
    def num_params() -> int:
        return 7

    def to_dict(self, layer_id: int, device_id: int) -> dict[str, torch.Tensor]:
        return {
            f"layer_{layer_id}_unproj_o_gamma_dev_{device_id}": self.unproj_o_gamma.to(device_id),
            f"layer_{layer_id}_exp_proj_weights_dev_{device_id}": self.exp_proj_weights.to(
                device_id
            ),
            f"layer_{layer_id}_exp_bias_dev_{device_id}": self.exp_bias.to(device_id),
            f"layer_{layer_id}_exp_upgate_weights_dev_{device_id}": self.exp_upgate_weights.to(
                device_id
            ),
            f"layer_{layer_id}_exp_upgate_scales_dev_{device_id}": self.exp_upgate_scales.to(
                device_id
            ),
            f"layer_{layer_id}_exp_down_weights_dev_{device_id}": self.exp_down_weights.to(
                device_id
            ),
            f"layer_{layer_id}_exp_down_scales_dev_{device_id}": self.exp_down_scales.to(device_id),
        }


class MlaFp8Params(BaseParams):
    def __init__(
        self,
        unproj_o_weights_and_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unproj_o_weights_and_scales = self.register_params(unproj_o_weights_and_scales)

    @staticmethod
    def num_params() -> int:
        return 1


class MLPFp8Params(BaseParams):
    def __init__(
        self,
        upgate_weights_and_scales: torch.Tensor,
        down_weights_and_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.upgate_weights_and_scales = self.register_params(upgate_weights_and_scales)
        self.down_weights_and_scales = self.register_params(down_weights_and_scales)

    @staticmethod
    def num_params() -> int:
        return 2


class MoEFp8Params(BaseParams):
    def __init__(
        self,
        exp_upgate_weights_and_scales: torch.Tensor,
        exp_down_weights_and_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        self.exp_upgate_weights_and_scales = self.register_params(exp_upgate_weights_and_scales)
        self.exp_down_weights_and_scales = self.register_params(exp_down_weights_and_scales)

    @staticmethod
    def num_params() -> int:
        return 2


class TempVars(BaseParams):
    def __init__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        ki: torch.Tensor,
        q_nope_down: torch.Tensor,
        q_pe: torch.Tensor,
        iq: torch.Tensor,
        iq_rt: torch.Tensor,
        idx_score: torch.Tensor,
        idx_logits: torch.Tensor,
        idx_sels: torch.Tensor,
        q_nope: torch.Tensor,
        o: torch.Tensor,
        o_acc: torch.Tensor,
        o_lse: torch.Tensor,
        o_lse_acc: torch.Tensor,
        proj_o: torch.Tensor,
        unproj_o: torch.Tensor,
        scores: torch.Tensor,
        x_mlp_in: torch.Tensor,
        exp_up_gate: torch.Tensor,
        sel_probs: torch.Tensor,
        sel_indices: torch.Tensor,
        exp_out: torch.Tensor,
        x_rmsnorm: torch.Tensor,
        logits_out: torch.Tensor,
        token_out: torch.Tensor,
    ) -> None:
        super().__init__()
        self.q = self.register_params(q)
        self.kv = self.register_params(kv)
        self.ki = self.register_params(ki)
        self.q_nope_down = self.register_params(q_nope_down)
        self.q_pe = self.register_params(q_pe)
        self.iq = self.register_params(iq)
        self.iq_rt = self.register_params(iq_rt)
        self.idx_score = self.register_params(idx_score)
        self.idx_logits = self.register_params(idx_logits)
        self.idx_sels = self.register_params(idx_sels)
        self.q_nope = self.register_params(q_nope)
        self.o = self.register_params(o)
        self.o_acc = self.register_params(o_acc)
        self.o_lse = self.register_params(o_lse)
        self.o_lse_acc = self.register_params(o_lse_acc)
        self.proj_o = self.register_params(proj_o)
        self.unproj_o = self.register_params(unproj_o)
        self.scores = self.register_params(scores)
        self.x_mlp_in = self.register_params(x_mlp_in)
        self.exp_up_gate = self.register_params(exp_up_gate)
        self.sel_probs = self.register_params(sel_probs)
        self.sel_indices = self.register_params(sel_indices)
        self.exp_out = self.register_params(exp_out)
        self.x_rmsnorm = self.register_params(x_rmsnorm)
        self.logits_out = self.register_params(logits_out)
        self.token_out = self.register_params(token_out)

    @staticmethod
    def num_params() -> int:
        return 26

    def tot_size_in_bytes_aligned(self, aligned_size: int) -> int:
        tot_size: int = 0
        for param in self._params:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            tot_size += aligned_param_size
        return tot_size

    def generate_params_with_continuous_storage(
        self, device: torch.device, aligned_size: int = 1024
    ) -> list[torch.Tensor]:
        tot_size = self.tot_size_in_bytes_aligned(aligned_size)
        cloned_params = []
        large_tensor = torch.zeros(tot_size, device=device, dtype=torch.uint8)
        offset = 0
        for param in self._params:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            cloned_params.append(
                large_tensor[offset : offset + param.nbytes].view(param.dtype).view(param.shape)
            )
            offset += aligned_param_size
        return cloned_params


class CacheVars(BaseParams):
    def __init__(
        self,
        k_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
    ) -> None:
        super().__init__()
        self.k_cache = self.register_params(k_cache)
        self.kv_cache = self.register_params(kv_cache)
        self.pe_cache = self.register_params(pe_cache)

    @staticmethod
    def num_params() -> int:
        return 3


class LLMHeadParams(BaseParams):
    """LLM Head Parameters"""

    def __init__(
        self,
        hidden_rms_gamma: torch.Tensor,
        head_proj_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.hidden_rms_gamma = self.register_params(hidden_rms_gamma)
        self.head_proj_weights = self.register_params(head_proj_weights)

    @staticmethod
    def num_params() -> int:
        return 2

    def to_dict(self, layer_id: int, device_id: int) -> dict[str, torch.Tensor]:
        return {
            f"layer_{layer_id}_model.norm.weight_dev_{device_id}": self.hidden_rms_gamma.to(
                device_id
            ),
            f"layer_{layer_id}_lm_head.weight_dev_{device_id}": self.head_proj_weights.to(
                device_id
            ),
        }
