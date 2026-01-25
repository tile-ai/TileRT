"""DSA show hands for deepseek v3.2."""

import glob
import json
import math
import os
import sys
import threading
import time
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs as ModelArgsV3_2
from tilert.models.deepseek_v3_2.params import (
    BaseParams,
    CacheVars,
    DenseLayerParamsKeys,
    Dsa671BModelInitializer,
    IntermediateMapper,
    MoELayerParamsKeys,
    TempVars,
)
from tilert.models.preprocess.weight_utils import (
    DownAllreduceWeightsConverter,
    ExpertSelectUpGateSiLUWeightsConverter,
    RMSNormHeadProjWeightsConverter,
    RMSNormProjQAKVAKIWeightsConverter,
    RMSNormUpGateSiLUWeightsConverter,
    UnProjOAllreduceWeightsConverter,
)
from tilert.models.utils import precompute_freqs_cis
from tilert.tilert_init import tilert_init
from tilert.utils import get_profile_log_tensor

__all__ = [
    "ShowHandsGenerator",
]

# MTP layer ID constant
MTP_LAYER_ID = 61

# MTP params keys order (for layer 61)
MTPPreprocessParamsKeys = [
    "embedding_rmsnorm_gamma",
    "hidden_rmsnorm_gamma",
    "eh_proj_weights",
]

MTPMlaParamsKeys = [
    "x_rmsnorm_gamma",
    "qkv_wa_weights",
    "qkv_wa_scales",
    "k_weights",
    "k_bias",
    "q_rmsnorm_gamma",
    "q_wb_weights",
    "q_wb_scales",
    "id_score_weights",
    "wkv_b1_weights",
    "wkv_b1_scales",
    "kv_rmsnorm_gamma",
    "wkv_b2_weights",
    "wkv_b2_scales",
    "unproj_weights",
    "unproj_scales",
]

MTPMoeParamsKeys = [
    "unproj_o_gamma",
    "exp_proj_weights",
    "exp_bias",
    "exp_upgate_weights",
    "exp_upgate_scales",
    "exp_down_weights",
    "exp_down_scales",
]


def stats_time(time_list: list[float], title: str) -> None:
    if len(time_list) > 0:
        avg_time = sum(time_list) / len(time_list)
        std_dev = math.sqrt(sum((x - avg_time) ** 2 for x in time_list) / len(time_list))
        logger.info(title)
        logger.info(f"--Average time taken to generate token: {avg_time * 1000:.4f} ms")
        logger.info(f"--Standard deviation of time: {std_dev * 1000:.4f} ms")
        logger.info(f"--Effective tokens per second: {1 / avg_time:.4f}")


DeviceResult = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]


def dsa_show_hands_prepare_money(
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    cache_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
    forward_max_seq_len: int,
) -> Any:
    """Prepare money for show hands"""
    return torch.ops.tilert.dsa_show_hands_prepare_money(
        params, temp_vars, cache_vars, profile_logs, forward_max_seq_len
    )


def dsa_show_hands(token_id: torch.Tensor) -> Any:
    """Show hands with native MT"""
    return torch.ops.tilert.dsa_show_hands(token_id)


def dsa_show_hands_reset(placeholder: torch.Tensor) -> Any:
    """Reset show one hand"""
    return torch.ops.tilert.dsa_show_hands_reset(placeholder)


def dsa_show_hands_go_home(placeholder: torch.Tensor) -> Any:
    """Go home"""
    return torch.ops.tilert.dsa_show_hands_go_home(placeholder)


# Put dirty conversion code here.
# TODO: better way to handle conversion.
def _convert_weights_on_demand(
    state_dicts: dict[str, torch.Tensor],
    skip_mtp_layer: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert weights on demand.

    Args:
        state_dicts: Dictionary of weights to convert.
        skip_mtp_layer: If True, skip layer 61 (MTP layer) weights.
    """
    res_dicts = {}
    for key, value in state_dicts.items():
        # Skip layer 61 (MTP layer) MLA/MoE/preprocess weights if requested,
        # but NOT lm_head and model.norm.weight (used by main model head)
        if (
            skip_mtp_layer
            and f"layer_{MTP_LAYER_ID}_" in key
            and "lm_head.weight" not in key
            and "model.norm.weight" not in key
        ):
            res_dicts[key] = value
            continue

        if "qkv_wa_weights" in key:  # first op
            weight_key = key
            scale_key = key.replace("qkv_wa_weights", "qkv_wa_scales")
            gamma_key = key.replace("qkv_wa_weights", "x_rmsnorm_gamma")
            common_weights = RMSNormProjQAKVAKIWeightsConverter.tilert_to_common(
                state_dicts[weight_key],
                state_dicts[scale_key],
                state_dicts[gamma_key],
            )
            conv_weights = (
                RMSNormProjQAKVAKIWeightsConverter.common_to_tilert_native_bf16_warp_gemv(
                    *common_weights
                )
            )
            res_dicts[key] = conv_weights[0]
        elif "unproj_weights" in key:  # unprojo_allreduce op
            weight_key = key
            scale_key = key.replace("unproj_weights", "unproj_scales")
            weights, scales = UnProjOAllreduceWeightsConverter.tilert_to_tilert_112sm_mma(
                state_dicts[weight_key],
                state_dicts[scale_key],
            )
            res_dicts[weight_key] = weights
            res_dicts[scale_key] = scales
            state_dicts[weight_key] = None
        elif "unproj_scales" in key:  # skip unprojo_allreduce op:: scales
            pass
        elif "exp_upgate_weights" in key:  # expert select up gate silu op
            weight_key = key
            scale_key = key.replace("exp_upgate_weights", "exp_upgate_scales")
            weights_and_scales = ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm_mma(
                state_dicts[weight_key], state_dicts[scale_key]
            )
            res_dicts[key] = weights_and_scales
            state_dicts[weight_key] = None

        elif "upgate_weights" in key:  # rmsnorm up gate silu op
            weight_key = key
            scale_key = key.replace("upgate_weights", "upgate_scales")
            weights_and_scales = RMSNormUpGateSiLUWeightsConverter.tilert_to_tilert_144sm_mma(
                state_dicts[weight_key],
                state_dicts[scale_key],
            )
            res_dicts[key] = weights_and_scales
            state_dicts[weight_key] = None
        elif "down_weights" in key:  # expert down allreduce op
            weight_key = key
            scale_key = key.replace("down_weights", "down_scales")
            weights_swizzled, scales = DownAllreduceWeightsConverter.tilert_to_tilert_mma(
                state_dicts[weight_key],
                state_dicts[scale_key],
            )
            res_dicts[weight_key] = weights_swizzled
            res_dicts[scale_key] = scales
            state_dicts[weight_key] = None
        elif "lm_head.weight" in key:  # head projection weights
            weights = RMSNormHeadProjWeightsConverter.tilert_to_tilert_native_bf16_warp_gemv(
                state_dicts[key]
            )
            res_dicts[key] = weights
            state_dicts[key] = None
        else:
            res_dicts[key] = value

    return res_dicts


def _convert_mtp_weights_on_demand(
    state_dicts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert MTP layer weights on demand to TileRT optimized format.

    This applies conversions specifically for MTP layer weights (layer 61).
    Only processes keys that contain 'layer_61_'.
    Note: lm_head and model.norm.weight are reused from main model (already converted).
    """
    res_dicts = {}
    for key, value in state_dicts.items():
        # Only process layer 61 (MTP layer) weights
        if f"layer_{MTP_LAYER_ID}_" not in key:
            res_dicts[key] = value
            continue

        # Skip lm_head and model.norm.weight - they're reused from main model
        # and already converted by _convert_weights_on_demand
        if "lm_head.weight" in key or "model.norm.weight" in key:
            res_dicts[key] = value
            continue

        if "qkv_wa_weights" in key:  # first op
            weight_key = key
            scale_key = key.replace("qkv_wa_weights", "qkv_wa_scales")
            gamma_key = key.replace("qkv_wa_weights", "x_rmsnorm_gamma")
            common_weights = RMSNormProjQAKVAKIWeightsConverter.tilert_to_common(
                state_dicts[weight_key],
                state_dicts[scale_key],
                state_dicts[gamma_key],
            )
            conv_weights = (
                RMSNormProjQAKVAKIWeightsConverter.common_to_tilert_native_bf16_warp_gemv(
                    *common_weights
                )
            )
            res_dicts[key] = conv_weights[0]
        elif "unproj_weights" in key:  # unproj_o_allreduce op
            weight_key = key
            scale_key = key.replace("unproj_weights", "unproj_scales")
            weights, scales = UnProjOAllreduceWeightsConverter.tilert_to_tilert_112sm_mma(
                state_dicts[weight_key],
                state_dicts[scale_key],
            )
            res_dicts[weight_key] = weights
            res_dicts[scale_key] = scales
        elif "unproj_scales" in key:  # skip - already processed with unproj_weights
            pass
        elif "exp_upgate_weights" in key:  # expert select up gate silu op
            weight_key = key
            scale_key = key.replace("exp_upgate_weights", "exp_upgate_scales")
            weights_and_scales = ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm_mma(
                state_dicts[weight_key], state_dicts[scale_key]
            )
            res_dicts[key] = weights_and_scales
        elif "exp_down_weights" in key:  # expert down allreduce op
            weight_key = key
            scale_key = key.replace("exp_down_weights", "exp_down_scales")
            weights_swizzled, scales = DownAllreduceWeightsConverter.tilert_to_tilert_mma(
                state_dicts[weight_key],
                state_dicts[scale_key],
            )
            res_dicts[weight_key] = weights_swizzled
            res_dicts[scale_key] = scales
        else:
            res_dicts[key] = value

    return res_dicts


class ShowHandsDSALayer(TileRTModule):
    """Show hands DSA for deepseek v3.2."""

    NUM_DENSE_LAYERS = 3
    NUM_MOE_LAYERS = 58
    NUM_LAYERS = NUM_DENSE_LAYERS + NUM_MOE_LAYERS

    def __init__(
        self,
        max_seq_len: int,
        model_path: str = "",
        with_weight_conversion: bool = True,
        with_mtp: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = 7168
        self.forward_max_seq_len = 4  # max supported seq_len per forward
        self.batch_size = 1
        self.num_heads = 16

        self.q_dim = 1536
        self.kv_dim = 512
        self.k_pe_dim = 64

        self.q_pe_lora_dim = 64
        self.q_pe_dim = 512
        self.q_nope_dim = 128

        self.v_head_dim = 128

        self.n_routed_experts = 256
        self.n_activate_experts = 8
        self.exp_dims = 256

        self.max_seq_len = max_seq_len

        self.vocab_size_full = 129280
        self.vocab_size = self.vocab_size_full // 8  # 16160

        self.num_devices = 8

        self.model_path = model_path
        self.with_weight_conversion = with_weight_conversion
        self.with_mtp = with_mtp

        self.multi_devices_results: list[DeviceResult | None] = [None] * torch.cuda.device_count()

        self.kv_cache = torch.zeros(
            self.batch_size, self.max_seq_len, self.kv_dim, dtype=torch.bfloat16, device="cuda:0"
        )
        self.pe_cache = torch.zeros(
            self.batch_size, self.max_seq_len, self.k_pe_dim, dtype=torch.bfloat16, device="cuda:0"
        )
        self.k_cache = torch.zeros(
            self.batch_size, self.max_seq_len, 128, dtype=torch.bfloat16, device="cuda:0"
        )

        self.placeholder = torch.zeros(1, 1, dtype=torch.int32, device="cpu")

    def _get_num_cache_layers(self) -> int:
        """Return number of cache layers. Override in subclass for MTP."""
        return self.NUM_LAYERS

    def _prepare_money(
        self,
        params: list[torch.Tensor],
        intermediates: list[torch.Tensor],
        caches: list[torch.Tensor],
        profile_logs: torch.Tensor,
    ) -> None:
        """Prepare money for show hands. Override in subclass for MTP."""
        dsa_show_hands_prepare_money(
            params, intermediates, caches, profile_logs, self.forward_max_seq_len
        )

    def _show_hands(self, token_id: torch.Tensor) -> Any:
        """Run show hands forward. Override in subclass for MTP."""
        return dsa_show_hands(token_id.cpu())

    def _reset_sequence_impl(self) -> None:
        """Reset sequence implementation. Override in subclass for MTP."""
        dsa_show_hands_reset(self.placeholder)

    def _cleanup_impl(self) -> None:
        """Cleanup implementation. Override in subclass for MTP."""
        dsa_show_hands_go_home(self.placeholder)

    def golden_forward(self) -> None:
        raise NotImplementedError("golden_forward not implemented")

    def tilert_forward(self) -> None:
        raise NotImplementedError("tilert_forward not implemented")

    def to_tilert_weights(self) -> BaseParams:
        raise NotImplementedError("to_tilert_weights not implemented")

    def get_mla_moe_layer_params_dict(
        self, layer_id: int, device: torch.device, dev_attrs: dict
    ) -> dict[str, torch.Tensor]:
        del dev_attrs
        dsa_671b_model = Dsa671BModelInitializer(
            torch.device(device),
            with_weight_conversion=self.with_weight_conversion,
        )
        return {
            **dsa_671b_model.init_mla_params().to_dict(layer_id, device),
            **dsa_671b_model.init_moe_params().to_dict(layer_id, device),
        }

    def get_mla_mlp_layer_params_dict(
        self, layer_id: int, device: torch.device, dev_attrs: dict
    ) -> dict[str, torch.Tensor]:
        del dev_attrs
        dsa_671b_model = Dsa671BModelInitializer(
            torch.device(device),
            with_weight_conversion=self.with_weight_conversion,
        )
        return {
            **dsa_671b_model.init_mla_params().to_dict(layer_id, device),
            **dsa_671b_model.init_mlp_params().to_dict(layer_id, device),
        }

    def get_llm_head_layer_params_dict(
        self, layer_id: int, device: torch.device, dev_attrs: dict
    ) -> dict[str, torch.Tensor]:
        del dev_attrs
        dsa_671b_model = Dsa671BModelInitializer(
            torch.device(device),
            with_weight_conversion=self.with_weight_conversion,
        )
        return {**dsa_671b_model.init_llm_head_params().to_dict(layer_id, device)}

    def get_temp_vars(self, device: torch.device, dev_attrs: dict) -> TempVars:
        del dev_attrs
        dsa_671b_model = Dsa671BModelInitializer(
            torch.device(device),
            with_weight_conversion=self.with_weight_conversion,
            with_mtp=self.with_mtp,
        )
        return dsa_671b_model.acquire_temp_vars()

    def get_cache_vars(self, device: torch.device, dev_attrs: dict) -> CacheVars:
        del dev_attrs
        return CacheVars(
            torch.zeros_like(self.k_cache).to(device),
            torch.zeros_like(self.kv_cache).to(device),
            torch.zeros_like(self.pe_cache).to(device),
        )

    def _gen_freqs_cis(self) -> torch.Tensor:
        freqs_cis = precompute_freqs_cis(ModelArgsV3_2())
        return torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1)

    def get_dev_id(self, weight_name: str) -> int:
        line_splits = weight_name.split("_dev_")
        if len(line_splits) == 2:
            return int(line_splits[1])

        return -1

    def get_weight_files(self, weight_map: dict[str, str], device_id: int) -> list[str]:
        """Get the weight files for the given device."""
        weight_files = []  # to preserve the order of weight files
        weight_files_set_ = set()  # to avoid duplicate weight files
        for weight in weight_map:
            dev_id = self.get_dev_id(weight)
            if dev_id == -1 or dev_id != device_id:
                continue

            weight_file = weight_map[weight]
            if weight_file in weight_files_set_:
                continue
            weight_files_set_.add(weight_file)
            weight_files.append(weight_file)

        return weight_files

    def load_embedding_weights(self, model_path: str, device_id: int) -> torch.Tensor:
        """Load the embedding weights for the given device."""
        # Look up the embedding file from index.json instead of hardcoding
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_file, encoding="utf-8") as f:
            weights_index = json.load(f)
        weight_map = weights_index["weight_map"]

        embed_key = "model.embed_tokens.weight"
        if embed_key not in weight_map:
            raise ValueError(f"Embedding weight {embed_key} not found in index.json")

        embed_weights_file = weight_map[embed_key]
        embed_weights_file_path = os.path.join(model_path, embed_weights_file)
        state_dict = load_file(embed_weights_file_path, device=f"cuda:{device_id}")
        return state_dict[embed_key]

    def get_total_shards(self, model_path: str) -> int:
        """Get the total number of shards by counting safetensors files in the directory."""
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        return len(safetensors_files)

    def _init_weights(self, model_path: str | None) -> None:
        """Load the model weights from the given path or generate random weights."""

        def _load_state_dicts(model_path: str, dev_attrs: dict) -> dict[str, torch.Tensor]:
            device_id = dev_attrs["device"]
            index_file = "model.safetensors.index.json"
            with open(os.path.join(model_path, index_file), encoding="utf-8") as f:
                weights_index = json.load(f)
            weight_map = weights_index["weight_map"]

            weight_files = self.get_weight_files(weight_map, device_id)
            state_dicts = {}
            for weight_file in weight_files:
                state_dict = load_file(
                    os.path.join(model_path, weight_file), device=f"cuda:{device_id}"
                )
                logger.info(f"Loaded weights from {weight_file} for device {device_id}")
                state_dicts.update(state_dict)
            embed_weights = self.load_embedding_weights(model_path, device_id)
            state_dicts["model.embed_tokens.weight"] = embed_weights
            return state_dicts

        def _gen_state_dicts_with_random_weights(dev_attrs: dict) -> dict[str, torch.Tensor]:
            device_id = dev_attrs["device"]
            state_dicts = {}
            for layer_id in range(self.NUM_DENSE_LAYERS):
                state_dicts.update(
                    self.get_mla_mlp_layer_params_dict(layer_id, device_id, dev_attrs)
                )
            for layer_id in range(3, 3 + self.NUM_MOE_LAYERS):
                state_dicts.update(
                    self.get_mla_moe_layer_params_dict(layer_id, device_id, dev_attrs)
                )
            state_dicts.update(self.get_llm_head_layer_params_dict(61, device_id, dev_attrs))
            dsa_671b_model = Dsa671BModelInitializer(
                torch.device(device_id),
                with_weight_conversion=self.with_weight_conversion,
            )
            state_dicts.update(dsa_671b_model.init_embedding_params().to_dict(device_id))
            return state_dicts

        def __load_weights(device_id: int, model_path: str | None) -> None:
            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            state_dicts = {}
            dev_attrs = {
                "device": device_id,
                "dtype": torch.bfloat16,
            }
            start_time = time.time()
            with torch.cuda.device(device_id):
                intermediates.extend(
                    self.get_temp_vars(
                        device_id, dev_attrs
                    ).generate_params_with_continuous_storage(device_id)
                )
                num_cache_layers = self._get_num_cache_layers()
                for _ in range(num_cache_layers):
                    caches.extend(self.get_cache_vars(device_id, dev_attrs).get_params())
                logger.info(f"Created intermediates and caches for device {device_id}")

                load_from_path = bool(model_path and os.path.exists(model_path))
                if load_from_path:
                    assert model_path is not None  # Type narrowing for mypy
                    state_dicts = _load_state_dicts(model_path, dev_attrs)
                else:
                    state_dicts = _gen_state_dicts_with_random_weights(dev_attrs)
                # Do necessary weight conversions only for loaded weights.
                # Skip MTP layer (layer 61) conversion here if with_mtp is True,
                # as it will be handled separately.
                if load_from_path:
                    state_dicts = _convert_weights_on_demand(
                        state_dicts, skip_mtp_layer=self.with_mtp
                    )

                for layer_id in range(self.NUM_DENSE_LAYERS):
                    # Each layer has its dedicated cache
                    for param in DenseLayerParamsKeys:
                        key_name = f"layer_{layer_id}_{param}_dev_{device_id}"
                        if key_name not in state_dicts:
                            raise ValueError(f"Weight {key_name} not found")
                        params.append(state_dicts[key_name])

                for layer_id in range(3, 3 + self.NUM_MOE_LAYERS):
                    # Each layer has its dedicated cache
                    for param in MoELayerParamsKeys:
                        key_name = f"layer_{layer_id}_{param}_dev_{device_id}"
                        if key_name not in state_dicts:
                            raise ValueError(f"Weight {key_name} not found")
                        params.append(state_dicts[key_name])

                # heads
                head = f"layer_61_lm_head.weight_dev_{device_id}"
                head_norm = f"layer_61_model.norm.weight_dev_{device_id}"
                if head not in state_dicts:
                    raise ValueError(f"Weight {head} not found")
                if head_norm not in state_dicts:
                    raise ValueError(f"Weight {head_norm} not found")
                params.append(state_dicts[head_norm])
                params.append(state_dicts[head])

                # embed_weights = self.load_embedding_weights(total_shards, device_id)
                params.append(state_dicts["model.embed_tokens.weight"])

                # RoPE frequencies
                freqs_cis = self._gen_freqs_cis()
                params.extend([freqs_cis.to(device_id)])

                # Add MTP-specific params when with_mtp is True
                if self.with_mtp:
                    if load_from_path:
                        # Load real MTP weights from state_dicts
                        # Convert MTP-specific weights (layer 61)
                        state_dicts = _convert_mtp_weights_on_demand(state_dicts)

                        # MTP params order (matching C++ register_mtp.cu):
                        # 1. LlmPreprocessModule: 2 (embedding + freqs_cis)
                        # 2. MtpPreProcessLayer: 3 (embedding_rmsnorm_gamma,
                        #    hidden_rmsnorm_gamma, eh_proj_weights)
                        # 3. MoeBlock: 23 (MLA 16 + MOE 7)
                        # 4. LlmHeadModule: 2 (hidden_rms_gamma, head_proj_weights)

                        # 1. Embedding params (2)
                        params.append(state_dicts["model.embed_tokens.weight"])
                        mtp_freqs_cis = self._gen_freqs_cis()
                        params.append(mtp_freqs_cis.to(device_id))

                        # 2. MTP preprocess params (3)
                        for key in MTPPreprocessParamsKeys:
                            full_key = f"layer_{MTP_LAYER_ID}_{key}_dev_{device_id}"
                            if full_key not in state_dicts:
                                raise ValueError(f"MTP weight {full_key} not found")
                            params.append(state_dicts[full_key])

                        # 3. MLA params (16) + MOE params (7) = MoeBlock (23)
                        for key in MTPMlaParamsKeys:
                            full_key = f"layer_{MTP_LAYER_ID}_{key}_dev_{device_id}"
                            if full_key not in state_dicts:
                                raise ValueError(f"MTP weight {full_key} not found")
                            params.append(state_dicts[full_key])

                        for key in MTPMoeParamsKeys:
                            full_key = f"layer_{MTP_LAYER_ID}_{key}_dev_{device_id}"
                            if full_key not in state_dicts:
                                raise ValueError(f"MTP weight {full_key} not found")
                            params.append(state_dicts[full_key])

                        # 4. LLM head params (2)
                        mtp_head_norm = f"layer_{MTP_LAYER_ID}_model.norm.weight_dev_{device_id}"
                        mtp_head = f"layer_{MTP_LAYER_ID}_lm_head.weight_dev_{device_id}"
                        if mtp_head_norm not in state_dicts:
                            raise ValueError(f"MTP weight {mtp_head_norm} not found")
                        if mtp_head not in state_dicts:
                            raise ValueError(f"MTP weight {mtp_head} not found")
                        params.append(state_dicts[mtp_head_norm])
                        params.append(state_dicts[mtp_head])

                        logger.info(f"Loaded real MTP weights for device {device_id}")
                    else:
                        # Use random weights for MTP
                        dsa_671b_model = Dsa671BModelInitializer(
                            torch.device(device_id),
                            with_weight_conversion=self.with_weight_conversion,
                            with_mtp=True,
                        )
                        # MTP needs: embedding, mtp_preprocess, mla, moe, llm_head params
                        params.extend(dsa_671b_model.init_embedding_params().get_params())
                        params.extend(dsa_671b_model.init_mtp_preprocess_params().get_params())
                        params.extend(dsa_671b_model.init_mla_params().get_params())
                        params.extend(dsa_671b_model.init_moe_params().get_params())
                        params.extend(dsa_671b_model.init_llm_head_params().get_params())

                profile_logs = get_profile_log_tensor(device=device_id, num_max_insts=65536)
                result = (intermediates, caches, params, profile_logs)
                self.multi_devices_results[device_id] = result

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = (
                f"{minutes} minutes {seconds} seconds" if minutes > 0 else f"{seconds} seconds"
            )
            logger.info(f"Completed loading weights for device {device_id} in {time_str}")

        threads = []
        exceptions: list[Exception | None] = [None] * self.num_devices
        for device_id in range(self.num_devices):

            def _runner(dev_id: int) -> None:
                try:
                    __load_weights(dev_id, model_path)
                except Exception as exc:  # pragma: no cover - surfaced after join
                    exceptions[dev_id] = exc

            thread = threading.Thread(target=_runner, args=(device_id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for device_id, exc in enumerate(exceptions):
            if exc is not None:
                raise RuntimeError(f"Failed to initialize device {device_id}: {exc}") from exc

        # Prepare money for all devices
        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                self._prepare_money(params, intermediates, caches, profile_logs)

    def from_pretrained(self, model_path: str) -> None:
        """Load the model weights from the given path."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        self._init_weights(model_path)

    def init_random_weights(self) -> None:
        """Generate random weights."""
        self._init_weights(None)

    def forward(
        self,
        token_id: torch.Tensor,
    ) -> list[DeviceResult]:
        self._show_hands(token_id)
        return [self._get_device_result(device_id) for device_id in range(self.num_devices)]

    def reset_sequence(self) -> None:
        self._reset_sequence_impl()

    def cleanup(self) -> None:
        self._cleanup_impl()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception as e:
            print(f"Exception during cleanup: {e}", file=sys.stderr)

    def _get_device_result(self, device_id: int) -> DeviceResult:
        device_result = self.multi_devices_results[device_id]
        if device_result is None:
            raise RuntimeError(f"Device {device_id} is not initialized")
        return device_result


class ShowHandsGenerator:
    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
        with_mtp: bool = False,
    ):
        """Initialize the ShowHandsGenerator.

        Args:
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 100.
            temperature: Temperature for sampling. Defaults to 1.0.
            model_weights_dir: Path of the model weights directory.
            with_mtp: Whether to use MTP (Multi-Token Prediction) for speculative decoding.
        """
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.with_mtp = with_mtp

        self.config = ModelArgsV3_2()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_weights_dir)
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size = 1  # fixed batch size to 1 for now

        self.default_device = torch.device("cuda:0")

        if with_mtp:
            from tilert.models.deepseek_v3_2.dsa_mtp_e2e_show_hands import ShowHandsDsaMtpE2eLayer

            self.decode_layer = ShowHandsDsaMtpE2eLayer(
                max_seq_len=self.config.max_seq_len,
            )
            self.mtp_seq_len = self.decode_layer.MTP_SEQ_LEN  # 4
        else:
            self.decode_layer = ShowHandsDSALayer(
                max_seq_len=self.config.max_seq_len,
                model_path=self.model_weights_dir,
            )

    def init(self) -> None:
        """Initialize the ShowHandsGenerator."""
        tilert_init()

    def cleanup(self) -> None:
        """Cleanup the ShowHandsGenerator."""
        self.decode_layer.cleanup()

    def init_random_weights(self) -> None:
        """Random initialize the weights."""
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        self.decode_layer.from_pretrained(self.model_weights_dir)

    @torch.inference_mode()
    def generate(self, prompt: str, print_log: bool = True) -> tuple[str, list[float], list[int]]:
        """Main function to load the model and perform single sequence generation.

        Returns:
            Tuple of (result_text, time_list, accepted_counts).
            accepted_counts is empty for non-MTP mode.
        """
        if self.with_mtp:
            return self._generate_with_mtp(prompt, print_log)
        result, time_list = self._generate_without_mtp(prompt, print_log)
        return result, time_list, []  # Empty accepted_counts for non-MTP

    def _generate_without_mtp(self, prompt: str, print_log: bool = True) -> tuple[str, list[float]]:
        """Standard generation without MTP."""
        prompt_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )
        prompt_mask = tokens != -1

        prev_pos = 0
        finished = torch.tensor(
            [False] * self.batch_size, dtype=torch.bool, device=self.default_device
        )

        time_list = []
        for cur_pos_val in range(1, total_len):
            start_time = time.time()
            multi_devices_results = self.decode_layer.forward(tokens[0, prev_pos])
            end_time = time.time()
            time_list.append(end_time - start_time)

            intermediates, *_ = multi_devices_results[0]
            intermediates_mapper = IntermediateMapper(list(intermediates[-TempVars.num_params() :]))
            next_token = intermediates_mapper.token_out[0][0]  # only the first token

            # replace the next token with the prompt token if the prompt mask is True
            next_token = torch.where(
                prompt_mask[0, cur_pos_val], tokens[0, cur_pos_val], next_token
            )
            tokens[0, cur_pos_val] = next_token
            finished |= torch.logical_and(~prompt_mask[0, cur_pos_val], next_token == self.eos_id)
            prev_pos = cur_pos_val
            if cur_pos_val >= prompt_len:
                decoded_tokens = self.tokenizer.decode(
                    [next_token.item()], skip_special_tokens=True
                )
                if print_log:
                    print(decoded_tokens, end="", flush=True)

            if finished.all():
                break

        if print_log:
            print("\n")
            logger.info(f"--Number of tokens generated: {len(time_list)}")

            # skip the first several samples to avoid the warmup effect
            stats_time(time_list[5:], "==== Performance ====")
            print("\n")

        # Reset sequence after generation, i.e. reset the cur_pos to 0 internally
        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            if self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id)]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return f"{decoded_tokens[0]}\n" if decoded_tokens else "", time_list

    def _generate_with_mtp(
        self, prompt: str, print_log: bool = True
    ) -> tuple[str, list[float], list[int]]:
        """Generation with MTP (Multi-Token Prediction) speculative decoding."""
        prompt_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        # Output tokens buffer
        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        prefill_time_list = []
        decode_time_list = []
        decode_accepted_counts = []  # Only track decode phase for statistics
        cur_pos = 0  # Current position in the output sequence

        # Prefill phase: process prompt tokens in chunks
        # Process prompt tokens in chunks of mtp_seq_len (with overlap)
        while cur_pos < prompt_len - 1:
            draft_end = min(cur_pos + self.mtp_seq_len, prompt_len)
            draft_tokens = tokens[0, cur_pos:draft_end].clone()
            actual_token_count = draft_tokens.shape[0]

            # Pad if needed (use last token for padding)
            if actual_token_count < self.mtp_seq_len:
                pad_token = draft_tokens[-1].item()
                padding = torch.full(
                    (self.mtp_seq_len - actual_token_count,),
                    pad_token,
                    dtype=torch.long,
                    device=self.default_device,
                )
                draft_tokens = torch.cat([draft_tokens, padding])

            draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)

            # Tell GPU how many tokens are valid (for cur_pos advancement)
            self.decode_layer.set_prefill_valid_tokens(actual_token_count)

            start_time = time.time()
            self.decode_layer.forward(draft_tokens)
            end_time = time.time()
            prefill_time_list.append(end_time - start_time)

            # Advance cur_pos by (actual_token_count - 1) to maintain overlap
            # This ensures cur_pos ends at prompt_len - 1 after all chunks
            cur_pos += actual_token_count - 1

        # Decode phase: speculative decoding
        # Set prefill_valid_tokens to 0 to switch to decode mode
        self.decode_layer.set_prefill_valid_tokens(0)

        finished = False
        while cur_pos < total_len - 1 and not finished:
            # Get next_draft_tokens from previous iteration
            # (or use last prompt tokens for first decode)
            if cur_pos == prompt_len - 1:
                # First decode iteration: use last prompt token repeated as placeholder drafts
                # We can't use [t6, t7, t8, t9] because that would apply wrong RoPE positions
                # (cur_pos=9 means positions 9,10,11,12, but t6 should be at position 6)
                last_token = tokens[0, prompt_len - 1].item()
                draft_tokens = torch.full(
                    (self.mtp_seq_len,),
                    last_token,
                    dtype=torch.long,
                    device=self.default_device,
                )
                draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)
            else:
                # Use next_draft_tokens from previous iteration
                draft_tokens = self.decode_layer.get_next_draft_tokens(0).reshape(
                    1, self.mtp_seq_len
                )

            start_time = time.time()
            self.decode_layer.forward(draft_tokens)
            end_time = time.time()
            decode_time_list.append(end_time - start_time)

            num_accepted = self.decode_layer.get_num_accepted(0)
            # Use predicted_tokens for output (not next_draft_tokens which is for next iteration)
            predicted_tokens = self.decode_layer.get_predicted_tokens(0).flatten()
            decode_accepted_counts.append(num_accepted)

            # Add accepted tokens to output
            num_output_tokens = num_accepted
            for i in range(num_output_tokens):
                if cur_pos + 1 + i >= total_len:
                    break
                new_token = int(predicted_tokens[i].item())
                tokens[0, cur_pos + 1 + i] = new_token

                # Print generated token
                if cur_pos + 1 + i >= prompt_len and print_log:
                    decoded_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
                    print(decoded_text, end="", flush=True)

                # Check for EOS
                if new_token == self.eos_id:
                    finished = True
                    break

            cur_pos += num_accepted

        if print_log:
            print("\n")
            total_tokens = sum(decode_accepted_counts)
            logger.info(f"--Number of forward calls (decode): {len(decode_accepted_counts)}")
            logger.info(f"--Total tokens generated: {total_tokens}")
            if len(decode_accepted_counts) > 0:
                avg_accepted = sum(decode_accepted_counts) / len(decode_accepted_counts)
                min_accepted = min(decode_accepted_counts)
                max_accepted = max(decode_accepted_counts)
                logger.info(
                    f"--Accepted tokens per call: mean={avg_accepted:.2f}, "
                    f"min={min_accepted}, max={max_accepted}"
                )

            # Calculate correct TPS accounting for MTP's multiple tokens per call
            if len(decode_time_list) > 5:
                total_decode_time = sum(decode_time_list[5:])  # skip warmup
                tokens_after_warmup = (
                    sum(decode_accepted_counts[5:])
                    if len(decode_accepted_counts) > 5
                    else total_tokens
                )
                effective_tps = (
                    tokens_after_warmup / total_decode_time if total_decode_time > 0 else 0
                )
                avg_time_ms = total_decode_time / len(decode_time_list[5:]) * 1000
                logger.info(f"--Avg forward time: {avg_time_ms:.2f}ms")
                logger.info(f"--Effective TPS (with MTP): {effective_tps:.2f} tokens/s")

            print("\n")

        # Reset sequence after generation
        self.decode_layer.reset_sequence()

        # Extract completion tokens
        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            # Remove -1 padding and tokens after EOS
            toks = [t for t in toks if t != -1]
            if self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id)]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return (
            f"{decoded_tokens[0]}\n" if decoded_tokens else "",
            decode_time_list,
            decode_accepted_counts,
        )
