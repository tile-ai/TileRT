from abc import abstractmethod

import torch

from tilert.models.deepseek_v3_2.model_args import ModelArgs as ModelArgsV3_2
from tilert.models.utils import SwizzleMode, gen_tensor_swizzle_map_1d, precompute_freqs_cis

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


class PlaceHolderParams(BaseParams):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def num_params() -> int:
        return 0


class EmbeddingParams(BaseParams):
    def __init__(
        self,
        embedding: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        super().__init__()
        self.embedding = self.register_params(embedding)
        self.freqs_cis = self.register_params(freqs_cis)

    @staticmethod
    def num_params() -> int:
        return 1

    def to_dict(self, device_id: int) -> dict[str, torch.Tensor]:
        return {"model.embed_tokens.weight": self.embedding.to(device_id)}


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


class MTPPreprocessParams(BaseParams):
    def __init__(
        self,
        embedding_rmsnorm_gamma: torch.Tensor,
        hidden_rmsnorm_gamma: torch.Tensor,
        eh_proj_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        self.embedding_rmsnorm_gamma = self.register_params(embedding_rmsnorm_gamma)
        self.hidden_rmsnorm_gamma = self.register_params(hidden_rmsnorm_gamma)
        self.eh_proj_weights = self.register_params(eh_proj_weights)

    @staticmethod
    def num_params() -> int:
        return 3

    def to_dict(self, layer_id: int, device_id: int) -> dict[str, torch.Tensor]:
        return {
            f"layer_{layer_id}_embedding_rmsnorm_gamma_dev_{device_id}": (
                self.embedding_rmsnorm_gamma.to(device_id)
            ),
            f"layer_{layer_id}_hidden_rmsnorm_gamma_dev_{device_id}": self.hidden_rmsnorm_gamma.to(
                device_id
            ),
            f"layer_{layer_id}_eh_proj_weights_dev_{device_id}": self.eh_proj_weights.to(device_id),
        }


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
        embedding_rmsnorm: torch.Tensor,
        hidden_rmsnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        x_tensor: torch.Tensor,
        rope_freqs: torch.Tensor,
        cur_pos: torch.Tensor,
        token_id: torch.Tensor,
        last_hidden_states: torch.Tensor,
        draft_tokens: torch.Tensor,
        predicted_tokens: torch.Tensor,
        predicted_hidden: torch.Tensor,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
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
        self.embedding_rmsnorm = self.register_params(embedding_rmsnorm)
        self.hidden_rmsnorm = self.register_params(hidden_rmsnorm)
        self.eh_proj = self.register_params(eh_proj)
        self.x_tensor = self.register_params(x_tensor)
        self.rope_freqs = self.register_params(rope_freqs)
        self.cur_pos = self.register_params(cur_pos)
        self.token_id = self.register_params(token_id)
        self.last_hidden_states = self.register_params(last_hidden_states)
        self.draft_tokens = self.register_params(draft_tokens)
        self.predicted_tokens = self.register_params(predicted_tokens)
        self.predicted_hidden = self.register_params(predicted_hidden)
        self.accepted_tokens = self.register_params(accepted_tokens)
        self.next_draft_tokens = self.register_params(next_draft_tokens)

    @staticmethod
    def num_params() -> int:
        return 39

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


class Dsa671BModelInitializer:
    """DSA with MTP e2e model for DeepSeek v3.2"""

    # TODO: These parameters should be carefully checked
    BATCH_SIZE = 1
    MAX_SEQ_LEN = 4
    NUM_HEADS = 16
    NUM_KI_HEADS = 64

    MAX_OPS = 2048
    MAX_SEL_TOKENS = 2048
    MAX_CTX_LEN = 16384
    NUM_DENSE_LAYERS = 3
    NUM_MOE_LAYERS = 58
    NUM_LAYERS = NUM_DENSE_LAYERS + NUM_MOE_LAYERS

    HIDDEN_SIZE = 7168
    PE_LORA_DIM = 64
    Q_NOPE_DIM = 128

    Q_DIM = 1536
    KV_CACHE_DIM = 512
    PE_CACHE_DIM = 64
    KI_CACHE_DIM = 128
    Q_PE_DIM = 512
    V_HEAD_DIM = 128

    N_ROUTED_EXPERTS = 256
    N_ACTIVATE_EXPERTS = 8
    N_TOTAL_EXPERTS = N_ACTIVATE_EXPERTS + 1
    EXP_DIMS = 256

    FULL_VOCAB_SIZE = 129280
    VOCAB_SIZE = FULL_VOCAB_SIZE // 8  # 16160

    def __init__(
        self,
        device: torch.device,
        max_seq_len: int | None = None,
        max_ctx_len: int | None = None,
        with_weight_conversion: bool = True,
        with_mtp: bool = False,
    ) -> None:
        super().__init__()

        self.device = device
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.MAX_SEQ_LEN
        self.max_ctx_len = max_ctx_len if max_ctx_len is not None else self.MAX_CTX_LEN
        self.with_weight_conversion = with_weight_conversion
        self.with_mtp = with_mtp

        self.bf16_desc = {"dtype": torch.bfloat16, "device": device}
        self.fp16_desc = {"dtype": torch.float16, "device": device}
        self.fp32_desc = {"dtype": torch.float32, "device": device}
        self.uint64_desc = {"dtype": torch.uint64, "device": device}
        self.int32_desc = {"dtype": torch.int32, "device": device}
        self.uint8_desc = {"dtype": torch.uint8, "device": device}

        self.mtp_params_sidx = 0

    def register_weights_and_scales(
        self, dim1: int, dim2: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = 128
        weights_dims = (dim1, dim2)
        weights = torch.randn(weights_dims, **self.bf16_desc).to(torch.float8_e4m3fn)
        scales = torch.randn((dim1, dim2 // block_size), **self.bf16_desc)
        return weights, scales

    def init_llm_head_params(self) -> LLMHeadParams:
        from tilert.models.preprocess.weight_utils import RMSNormHeadProjWeightsConverter

        hidden_rms_gamma_shape = (self.HIDDEN_SIZE,)
        head_proj_weights_shape = (self.VOCAB_SIZE, self.HIDDEN_SIZE)

        hidden_rms_gamma = torch.randn(hidden_rms_gamma_shape, **self.fp32_desc)
        head_proj_weights = torch.randn(head_proj_weights_shape, **self.bf16_desc)

        if self.with_weight_conversion:
            # Apply weight conversion for LLM head
            head_proj_weights = (
                RMSNormHeadProjWeightsConverter.tilert_to_tilert_native_bf16_warp_gemv(
                    head_proj_weights
                )
            )

        return LLMHeadParams(
            hidden_rms_gamma,
            head_proj_weights,
        )

    def init_embedding_params(self) -> EmbeddingParams:
        embedding = torch.randn(self.FULL_VOCAB_SIZE, self.HIDDEN_SIZE, **self.bf16_desc)
        freqs_cis = precompute_freqs_cis(ModelArgsV3_2())
        freqs_cis = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1)
        return EmbeddingParams(embedding, freqs_cis.to(self.device))

    def init_mla_params(self) -> MlaParams:
        from tilert.models.preprocess.weight_utils import (
            RMSNormProjQAKVAKIWeightsConverter,
        )

        qkv_dim = self.Q_DIM + self.KV_CACHE_DIM + self.PE_LORA_DIM + 128
        x_rmsnorm_gamma_shape = (self.HIDDEN_SIZE,)
        q_wb_shape = ((self.PE_LORA_DIM + self.Q_NOPE_DIM + 512) * self.NUM_HEADS, self.Q_DIM)
        wkv_b1_shape = (self.NUM_HEADS, self.KV_CACHE_DIM, self.V_HEAD_DIM)
        wkv_b2_shape = (self.NUM_HEADS, self.V_HEAD_DIM, self.KV_CACHE_DIM)
        wkv_b2_scales_shape = (self.NUM_HEADS, self.V_HEAD_DIM // 128, self.KV_CACHE_DIM // 128)
        unproj_w_shape = (self.HIDDEN_SIZE, self.NUM_HEADS * self.V_HEAD_DIM)
        unproj_scales_shape = (896, self.NUM_HEADS * self.V_HEAD_DIM // 128)

        x_rmsnorm_gamma = torch.randn(x_rmsnorm_gamma_shape, **self.fp32_desc)
        qkv_wa_weights, _ = self.register_weights_and_scales(qkv_dim, self.HIDDEN_SIZE)
        qkv_wa_scales = torch.randn((130, 64), **self.bf16_desc)
        k_weights = torch.randn(128, **self.fp32_desc)
        k_bias = torch.randn(128, **self.fp32_desc)
        q_rmsnorm_gamma = torch.randn(self.Q_DIM, **self.fp32_desc)
        q_wb_weights, _ = self.register_weights_and_scales(*q_wb_shape)
        q_wb_scales = torch.randn((448, 12), **self.bf16_desc)
        id_score_weights = torch.randn(64, self.HIDDEN_SIZE, **self.bf16_desc)
        wkv_b1_weights = torch.randn(wkv_b1_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        wkv_b1_scales = torch.randn((16, 8, 1), **self.bf16_desc)
        kv_rmsnorm_gamma = torch.randn(self.KV_CACHE_DIM, **self.fp32_desc)
        wkv_b2_weights = torch.randn(wkv_b2_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        wkv_b2_scales = torch.randn(wkv_b2_scales_shape, **self.bf16_desc)
        unproj_weights = torch.randn(unproj_w_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        unproj_scales = torch.randn(unproj_scales_shape, **self.bf16_desc)

        if self.with_weight_conversion:
            # Apply weight conversion for MLA qkv_wa weights
            # Convert tilert format -> common format -> tilert native bf16 warp gemv format
            common_weights = RMSNormProjQAKVAKIWeightsConverter.tilert_to_common(
                qkv_wa_weights, qkv_wa_scales, x_rmsnorm_gamma
            )
            qkv_wa_weights, x_rmsnorm_gamma = (
                RMSNormProjQAKVAKIWeightsConverter.common_to_tilert_native_bf16_warp_gemv(
                    *common_weights
                )
            )

        return MlaParams(
            x_rmsnorm_gamma=x_rmsnorm_gamma,
            qkv_wa_weights=qkv_wa_weights,
            qkv_wa_scales=qkv_wa_scales,
            k_weights=k_weights,
            k_bias=k_bias,
            q_rmsnorm_gamma=q_rmsnorm_gamma,
            q_wb_weights=q_wb_weights,
            q_wb_scales=q_wb_scales,
            id_score_weights=id_score_weights,
            wkv_b1_weights=wkv_b1_weights,
            wkv_b1_scales=wkv_b1_scales,
            kv_rmsnorm_gamma=kv_rmsnorm_gamma,
            wkv_b2_weights=wkv_b2_weights,
            wkv_b2_scales=wkv_b2_scales,
            unproj_weights=unproj_weights,
            unproj_scales=unproj_scales,
        )

    def init_mlp_params(self) -> MLPParams:
        exp_upgate_w_shape = (9, self.EXP_DIMS * 2, self.HIDDEN_SIZE)
        exp_upgate_s_shape = (9, self.EXP_DIMS * 2 // 128, 64)
        exp_down_w_shape = (9, self.HIDDEN_SIZE, self.EXP_DIMS)
        exp_down_s_shape = (9, 1024, self.EXP_DIMS // 128)

        unproj_o_gamma = torch.randn(self.HIDDEN_SIZE, **self.fp32_desc)
        upgate_weights = torch.randn(exp_upgate_w_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        upgate_scales = torch.randn(exp_upgate_s_shape, **self.bf16_desc)
        down_weights = torch.randn(exp_down_w_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        down_scales = torch.randn(exp_down_s_shape, **self.bf16_desc)

        return MLPParams(
            unproj_o_gamma,
            upgate_weights,
            upgate_scales,
            down_weights,
            down_scales,
        )

    def init_moe_params(self) -> MoEParams:
        from tilert.models.preprocess.weight_utils import (
            ExpertSelectUpGateSiLUWeightsConverter,
        )

        exp_ug_w_shape = (self.N_ROUTED_EXPERTS + 1, self.EXP_DIMS * 2, self.HIDDEN_SIZE)
        exp_upgate_s_shape = (self.N_ROUTED_EXPERTS + 1, self.EXP_DIMS * 2 // 128, 64)
        exp_down_w_shape = (self.N_ROUTED_EXPERTS + 1, self.HIDDEN_SIZE, self.EXP_DIMS)
        exp_down_s_shape = (self.N_ROUTED_EXPERTS + 1, 1024, self.EXP_DIMS // 128)

        unproj_o_gamma = torch.randn(self.HIDDEN_SIZE, **self.fp32_desc)
        exp_proj_weights = torch.randn((self.N_ROUTED_EXPERTS, self.HIDDEN_SIZE), **self.bf16_desc)
        exp_bias = torch.randn(self.N_ROUTED_EXPERTS, **self.fp32_desc)
        exp_upgate_weights = torch.randn(exp_ug_w_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        exp_upgate_scales = torch.randn(exp_upgate_s_shape, **self.bf16_desc)
        exp_down_weights = torch.randn(exp_down_w_shape, **self.fp16_desc).to(torch.float8_e4m3fn)
        exp_down_scales = torch.randn(exp_down_s_shape, **self.bf16_desc)

        if self.with_weight_conversion:
            # Apply weight conversion for MOE exp_upgate weights
            exp_upgate_weights = ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm_mma(
                exp_upgate_weights, exp_upgate_scales
            )

        return MoEParams(
            unproj_o_gamma,
            exp_proj_weights,
            exp_bias,
            exp_upgate_weights,
            exp_upgate_scales,
            exp_down_weights,
            exp_down_scales,
        )

    def init_mtp_preprocess_params(self) -> MTPPreprocessParams:
        """Initialize MTP preprocess parameters with random values."""
        embedding_rmsnorm_gamma = torch.randn(self.HIDDEN_SIZE, **self.fp32_desc)
        hidden_rmsnorm_gamma = torch.randn(self.HIDDEN_SIZE, **self.fp32_desc)
        eh_proj_weights = torch.randn((128, 7, 56, 256), **self.bf16_desc)
        return MTPPreprocessParams(
            embedding_rmsnorm_gamma,
            hidden_rmsnorm_gamma,
            eh_proj_weights,
        )

    def acquire_params(self) -> list[torch.Tensor]:
        params = []

        for _ in range(self.NUM_DENSE_LAYERS):
            params.extend(self.init_mla_params().get_params())
            params.extend(self.init_mlp_params().get_params())

        for _ in range(self.NUM_MOE_LAYERS):
            params.extend(self.init_mla_params().get_params())
            params.extend(self.init_moe_params().get_params())

        params.extend(self.init_llm_head_params().get_params())
        params.extend(self.init_embedding_params().get_params())

        if self.with_mtp:
            self.mtp_params_sidx = len(params)
            params.extend(self.init_embedding_params().get_params())
            params.extend(self.init_mtp_preprocess_params().get_params())
            params.extend(self.init_mla_params().get_params())
            params.extend(self.init_moe_params().get_params())
            params.extend(self.init_llm_head_params().get_params())

        return params

    def acquire_temp_vars(self, seq_len: int | None = None) -> TempVars:
        """Acquire temporary variables for the model.

        Args:
            seq_len: Sequence length for temp vars. If None, uses self.max_seq_len.

        Returns:
            TempVars object containing all temporary tensors.
        """
        seq_len = seq_len if seq_len is not None else self.max_seq_len
        BATCH_SEQ = (self.BATCH_SIZE, seq_len)

        q = torch.zeros(*BATCH_SEQ, self.Q_DIM, **self.bf16_desc)
        kv = torch.zeros(*BATCH_SEQ, self.KV_CACHE_DIM, **self.bf16_desc)
        q_pe = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, self.PE_LORA_DIM, **self.bf16_desc)
        ki = torch.zeros(*BATCH_SEQ, self.KI_CACHE_DIM, **self.bf16_desc)
        q_nope_down = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, self.V_HEAD_DIM, **self.bf16_desc)
        q_nope = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, self.Q_PE_DIM, **self.bf16_desc)
        iq = torch.zeros(*BATCH_SEQ, self.NUM_KI_HEADS, self.KI_CACHE_DIM, **self.bf16_desc)
        iq_rt = torch.zeros(*BATCH_SEQ, self.NUM_KI_HEADS, self.KI_CACHE_DIM, **self.bf16_desc)
        idx_score = torch.zeros(*BATCH_SEQ, self.NUM_KI_HEADS, **self.bf16_desc)
        idx_logits = torch.zeros(*BATCH_SEQ, self.max_ctx_len, **self.fp32_desc)
        idx_sels = torch.zeros(*BATCH_SEQ, self.MAX_SEL_TOKENS, **self.int32_desc)
        o = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, self.KV_CACHE_DIM, **self.bf16_desc)
        o_acc = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, 32, self.KV_CACHE_DIM, **self.fp32_desc)
        o_lse = torch.empty(*BATCH_SEQ, self.NUM_HEADS, **self.fp32_desc)
        o_lse_acc = torch.empty(*BATCH_SEQ, self.NUM_HEADS, 32, **self.fp32_desc)
        proj_o = torch.zeros(*BATCH_SEQ, self.NUM_HEADS, self.V_HEAD_DIM, **self.bf16_desc)
        unproj_o = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        scores = torch.zeros(*BATCH_SEQ, self.N_ROUTED_EXPERTS, **self.fp32_desc)
        x_mlp_in = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        exp_up_gate = torch.zeros(*BATCH_SEQ, self.N_TOTAL_EXPERTS, self.EXP_DIMS, **self.bf16_desc)
        sel_probs = torch.zeros(*BATCH_SEQ, self.N_ACTIVATE_EXPERTS, **self.fp32_desc)
        sel_indices = torch.zeros(*BATCH_SEQ, self.N_ACTIVATE_EXPERTS, **self.int32_desc)
        exp_out = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        x_rmsnorm = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        logits_out = torch.zeros(*BATCH_SEQ, self.VOCAB_SIZE, **self.fp32_desc)
        token_out = torch.zeros(*BATCH_SEQ, 1, **self.int32_desc)

        embedding_rmsnorm = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        hidden_rmsnorm = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        eh_proj = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        x_tensor = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        rope_freqs = torch.zeros(*BATCH_SEQ, self.PE_CACHE_DIM, **self.fp32_desc)
        cur_pos = torch.zeros(self.BATCH_SIZE, **self.int32_desc)
        token_id = torch.zeros(*BATCH_SEQ, 1, **self.int32_desc)
        last_hidden_states = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)

        draft_tokens = torch.zeros(*BATCH_SEQ, **self.int32_desc)
        predicted_tokens = torch.zeros(*BATCH_SEQ, 1, **self.int32_desc)
        predicted_hidden = torch.zeros(*BATCH_SEQ, self.HIDDEN_SIZE, **self.bf16_desc)
        accepted_tokens = torch.zeros(self.BATCH_SIZE, **self.int32_desc)
        next_draft_tokens = torch.zeros(*BATCH_SEQ, **self.int32_desc)

        return TempVars(
            q,
            kv,
            ki,
            q_nope_down,
            q_pe,
            iq,
            iq_rt,
            idx_score,
            idx_logits,
            idx_sels,
            q_nope,
            o,
            o_acc,
            o_lse,
            o_lse_acc,
            proj_o,
            unproj_o,
            scores,
            x_mlp_in,
            exp_up_gate,
            sel_probs,
            sel_indices,
            exp_out,
            x_rmsnorm,
            logits_out,
            token_out,
            embedding_rmsnorm,
            hidden_rmsnorm,
            eh_proj,
            x_tensor,
            rope_freqs,
            cur_pos,
            token_id,
            last_hidden_states,
            draft_tokens,
            predicted_tokens,
            predicted_hidden,
            accepted_tokens,
            next_draft_tokens,
        )

    def acquire_cache_vars(self, num_layers: int | None = None) -> list[torch.Tensor]:
        """Acquire cache variables for the model.

        Args:
            num_layers: Number of layers to create cache for. If None, uses NUM_LAYERS.

        Returns:
            List of cache tensors (3 tensors per layer: k_cache, kv_cache, pe_cache).
        """
        num_layers = num_layers if num_layers is not None else self.NUM_LAYERS
        if self.with_mtp:
            num_layers += 1

        BATCH_CTX = (self.BATCH_SIZE, self.max_ctx_len)
        cache_vars = []
        for _ in range(num_layers):
            cache_vars.extend(
                [
                    torch.zeros(*BATCH_CTX, self.KI_CACHE_DIM, **self.bf16_desc),
                    torch.zeros(*BATCH_CTX, self.KV_CACHE_DIM, **self.bf16_desc),
                    torch.zeros(*BATCH_CTX, self.PE_CACHE_DIM, **self.bf16_desc),
                ]
            )
        return cache_vars

    def acquire_single_layer_cache_vars(self) -> list[torch.Tensor]:
        """Acquire cache variables for a single layer.

        Returns:
            List of 3 cache tensors: k_cache, kv_cache, pe_cache.
        """
        return self.acquire_cache_vars(num_layers=1)

    def acquire_misc_vars(self) -> list[torch.Tensor]:
        return [
            torch.zeros(self.MAX_OPS, 148, 16, **self.uint64_desc),
            torch.zeros(self.MAX_OPS, 128, **self.uint8_desc),
            torch.zeros(self.MAX_OPS, 8, **self.uint8_desc),
        ]

    def get_mtp_all_vars(self) -> list[torch.Tensor]:
        return [
            self.acquire_params()[self.mtp_params_sidx :],
            self.acquire_temp_vars().get_params(),
            self.acquire_cache_vars()[-3:],
            # Potential issue: Reallocate misc vars for MTP
            self.acquire_misc_vars(),
        ]
