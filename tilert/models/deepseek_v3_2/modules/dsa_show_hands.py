"""DSA show hands for deepseek v3.2."""

import glob
import json
import os
import sys
import threading
import time
from typing import Any

import torch
from safetensors.torch import load_file

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs as ModelArgsV3_2
from tilert.models.deepseek_v3_2.params import (
    CacheVars,
    DenseLayerParamsKeys,
    LLMHeadParams,
    MlaParams,
    MLPParams,
    MoELayerParamsKeys,
    MoEParams,
    TempVars,
)
from tilert.models.utils import precompute_freqs_cis
from tilert.tests.utils import get_profile_log_tensor

__all__ = [
    "ShowHandsDSALayer",
]

DeviceResult = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]


def dsa_show_hands_prepare_money(
    enable_fused_op: bool,
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    cache_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
) -> Any:
    """Prepare money for show hands"""
    return torch.ops.tilert.dsa_show_hands_prepare_money(
        enable_fused_op, params, temp_vars, cache_vars, profile_logs
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


class ShowHandsDSALayer(TileRTModule):
    """Show hands DSA for deepseek v3.2."""

    NUM_DENSE_LAYERS = 3
    NUM_MOE_LAYERS = 58
    NUM_LAYERS = NUM_DENSE_LAYERS + NUM_MOE_LAYERS

    def __init__(
        self,
        max_seq_len: int,
        model_path: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = 7168
        self.seq_len = 1
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

        self.multi_devices_results: list[DeviceResult | None] = [None] * torch.cuda.device_count()

        self.kv_cache = torch.zeros(
            self.batch_size,
            self.max_seq_len,
            self.kv_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        self.pe_cache = torch.zeros(
            self.batch_size,
            self.max_seq_len,
            self.k_pe_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        self.k_cache = torch.zeros(
            self.batch_size,
            self.max_seq_len,
            128,
            dtype=torch.bfloat16,
            device="cuda:0",
        )

        self.placeholder = torch.zeros(1, 1, dtype=torch.int32, device="cpu")

    def golden_forward(self) -> None:
        raise NotImplementedError("golden_forward not implemented")

    def tilert_forward(self) -> None:
        raise NotImplementedError("tilert_forward not implemented")

    def to_tilert_weights(self) -> None:
        raise NotImplementedError("to_tilert_weights not implemented")

    def register_weights_and_scales(
        self, dim1: int, dim2: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = 128
        weights_dims = (dim1, dim2)
        weights = torch.randn(weights_dims, dtype=torch.float16, device=device).to(
            torch.float8_e4m3fn
        )
        scales = torch.randn(
            (dim1, dim2 // block_size),
            dtype=torch.bfloat16,
            device=device,
        )
        return weights, scales

    def init_mla_params(self, device: torch.device, dev_attrs: dict) -> MlaParams:
        mat_attrs = {
            "device": device,
            "dtype": torch.float16,
        }

        qkv_dim = self.q_dim + self.kv_dim + self.k_pe_dim + 128
        x_rmsnorm_gamma_shape = (self.hidden_size,)
        q_wb_shape = (
            (self.q_pe_lora_dim + self.q_nope_dim + 512) * self.num_heads,
            self.q_dim,
        )
        wkv_b1_shape = (self.num_heads, self.q_pe_dim, self.v_head_dim)
        wkv_b2_shape = (self.num_heads, self.v_head_dim, self.kv_dim)
        wkv_b2_scales_shape = (
            self.num_heads,
            self.v_head_dim // 128,
            self.kv_dim // 128,
        )
        unproj_w_shape = (self.hidden_size, self.num_heads * self.v_head_dim)

        x_rmsnorm_gamma = torch.randn(x_rmsnorm_gamma_shape, dtype=torch.float32, device=device)
        qkv_wa_weights, _ = self.register_weights_and_scales(qkv_dim, self.hidden_size, device)
        qkv_wa_scales = torch.randn((130, 64), dtype=torch.bfloat16, device=device)
        k_weights = torch.randn(128, dtype=torch.float32, device=device)
        k_bias = torch.randn(128, dtype=torch.float32, device=device)
        q_rmsnorm_gamma = torch.randn(self.q_dim, dtype=torch.float32, device=device)
        q_wb_weights, _ = self.register_weights_and_scales(*q_wb_shape, device)
        q_wb_scales = torch.randn((448, 12), dtype=torch.bfloat16, device=device)
        id_score_weights = torch.randn(64, self.hidden_size, dtype=torch.bfloat16, device=device)
        wkv_b1_weights = torch.randn(wkv_b1_shape, **mat_attrs).to(torch.float8_e4m3fn)
        wkv_b1_scales = torch.randn((16, 8, 1), dtype=torch.bfloat16, device=device)
        kv_rmsnorm_gamma = torch.randn(self.kv_dim, dtype=torch.float32, device=device)
        wkv_b2_weights = torch.randn(wkv_b2_shape, **mat_attrs).to(torch.float8_e4m3fn)
        wkv_b2_scales = torch.randn(wkv_b2_scales_shape, **dev_attrs)
        unproj_weights = torch.randn(unproj_w_shape, **mat_attrs).to(torch.float8_e4m3fn)
        unproj_scales = torch.randn((896, self.num_heads * self.v_head_dim // 128), **dev_attrs)

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

    def init_mlp_params(self, device: torch.device, dev_attrs: dict) -> MLPParams:
        mat_attrs = {
            "device": device,
            "dtype": torch.float16,
        }

        exp_upgate_w_shape = (9, self.exp_dims * 2, self.hidden_size)
        exp_upgate_s_shape = (9, self.exp_dims * 2 // 128, 64)
        exp_down_w_shape = (9, self.hidden_size, self.exp_dims)
        exp_down_s_shape = (9, 1024, self.exp_dims // 128)

        unproj_o_gamma = torch.randn(self.hidden_size, dtype=torch.float32, device=device)
        upgate_weights = torch.randn(exp_upgate_w_shape, **mat_attrs).to(torch.float8_e4m3fn)
        upgate_scales = torch.randn(exp_upgate_s_shape, **dev_attrs)
        down_weights = torch.randn(exp_down_w_shape, **mat_attrs).to(torch.float8_e4m3fn)
        down_scales = torch.randn(exp_down_s_shape, **dev_attrs)

        return MLPParams(
            unproj_o_gamma,
            upgate_weights,
            upgate_scales,
            down_weights,
            down_scales,
        )

    def init_moe_params(self, device: torch.device, dev_attrs: dict) -> MoEParams:
        mat_attrs = {
            "device": device,
            "dtype": torch.float16,
        }

        exp_upgate_w_shape = (
            self.n_routed_experts + 1,
            self.exp_dims * 2,
            self.hidden_size,
        )
        exp_upgate_s_shape = (self.n_routed_experts + 1, self.exp_dims * 2 // 128, 64)
        exp_down_w_shape = (self.n_routed_experts + 1, self.hidden_size, self.exp_dims)
        exp_down_s_shape = (self.n_routed_experts + 1, 1024, self.exp_dims // 128)

        unproj_o_gamma = torch.randn(self.hidden_size, dtype=torch.float32, device=device)
        exp_proj_weights = torch.randn((self.n_routed_experts, self.hidden_size), **dev_attrs)
        exp_bias = torch.randn(self.n_routed_experts, dtype=torch.float32, device=device)
        exp_upgate_weights = torch.randn(exp_upgate_w_shape, **mat_attrs).to(torch.float8_e4m3fn)
        exp_upgate_scales = torch.randn(exp_upgate_s_shape, **dev_attrs)
        exp_down_weights = torch.randn(exp_down_w_shape, **mat_attrs).to(torch.float8_e4m3fn)
        exp_down_scales = torch.randn(exp_down_s_shape, **dev_attrs)

        return MoEParams(
            unproj_o_gamma,
            exp_proj_weights,
            exp_bias,
            exp_upgate_weights,
            exp_upgate_scales,
            exp_down_weights,
            exp_down_scales,
        )

    def init_llm_head_params(self, device: torch.device, dev_attrs: dict) -> LLMHeadParams:
        del dev_attrs
        hidden_rms_gamma_shape = (self.hidden_size,)
        head_proj_weights_shape = (self.vocab_size, self.hidden_size)

        hidden_rms_gamma = torch.randn(hidden_rms_gamma_shape, dtype=torch.float32, device=device)
        head_proj_weights = torch.randn(
            head_proj_weights_shape, dtype=torch.bfloat16, device=device
        )
        return LLMHeadParams(
            hidden_rms_gamma,
            head_proj_weights,
        )

    def get_mla_moe_layer_params(self, device: torch.device, dev_attrs: dict) -> list[torch.Tensor]:
        return [
            *self.init_mla_params(device, dev_attrs).get_params(),
            *self.init_moe_params(device, dev_attrs).get_params(),
        ]

    def get_mla_mlp_layer_params(self, device: torch.device, dev_attrs: dict) -> list[torch.Tensor]:
        return [
            *self.init_mla_params(device, dev_attrs).get_params(),
            *self.init_mlp_params(device, dev_attrs).get_params(),
        ]

    def get_llm_head_layer_params(
        self, device: torch.device, dev_attrs: dict
    ) -> list[torch.Tensor]:
        return [
            *self.init_llm_head_params(device, dev_attrs).get_params(),
        ]

    def get_freqs_cis_layer_params(
        self, device: torch.device, dev_attrs: dict
    ) -> list[dict[str, torch.Tensor]]:
        return [
            *self.init_freqs_cis_params(device, dev_attrs).get_params(),
        ]

    def get_temp_vars(self, device: torch.device, dev_attrs: dict) -> TempVars:
        q = torch.zeros(self.batch_size, self.seq_len, self.q_dim, **dev_attrs)
        kv = torch.zeros(self.batch_size, self.seq_len, self.kv_dim, **dev_attrs)
        q_pe = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.q_pe_lora_dim,
            **dev_attrs,
        )
        ki = torch.zeros(self.batch_size, self.seq_len, 128, **dev_attrs)
        q_nope_down = torch.zeros(
            self.batch_size, self.seq_len, self.num_heads, self.v_head_dim, **dev_attrs
        )
        q_nope = torch.zeros(
            self.batch_size, self.seq_len, self.num_heads, self.q_pe_dim, **dev_attrs
        )
        iq = torch.zeros(self.batch_size, self.seq_len, 64, 128, **dev_attrs)
        iq_rt = torch.zeros(self.batch_size, self.seq_len, 64, 128, **dev_attrs)
        idx_score = torch.zeros(self.batch_size, self.seq_len, 64, **dev_attrs)
        idx_logits = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.max_seq_len,
            dtype=torch.float32,
            device=device,
        )
        idx_sels = torch.zeros(self.batch_size, 2048, dtype=torch.int32, device=device)
        o = torch.zeros(self.batch_size, self.seq_len, self.num_heads, self.kv_dim, **dev_attrs)
        o_acc = torch.zeros(
            self.batch_size,
            self.num_heads,
            128,
            self.kv_dim,
            dtype=torch.float32,
            device=device,
        )
        o_lse = torch.empty(self.batch_size, self.num_heads, dtype=torch.float32, device=device)
        o_lse_acc = torch.empty(
            self.batch_size, self.num_heads, 128, dtype=torch.float32, device=device
        )
        proj_o = torch.zeros(
            self.batch_size, self.seq_len, self.num_heads, self.v_head_dim, **dev_attrs
        )
        unproj_o = torch.zeros(self.batch_size, self.seq_len, self.hidden_size, **dev_attrs)
        scores = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.n_routed_experts,
            dtype=torch.float32,
            device=device,
        )
        x_mlp_in = torch.zeros(self.batch_size, self.seq_len, self.hidden_size, **dev_attrs)
        exp_up_gate = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.n_activate_experts + 1,
            self.exp_dims,
            **dev_attrs,
        )
        sel_probs = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.n_activate_experts,
            dtype=torch.float32,
            device=device,
        )
        sel_indices = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.n_activate_experts,
            dtype=torch.int32,
            device=device,
        )
        exp_out = torch.zeros(self.batch_size, self.seq_len, self.hidden_size, **dev_attrs)
        x_rmsnorm = torch.zeros(self.batch_size, self.seq_len, self.hidden_size, **dev_attrs)
        logits_out = torch.zeros(
            self.batch_size, self.vocab_size, dtype=torch.float32, device=device
        )
        token_out = torch.zeros(self.batch_size, 1, dtype=torch.int32, device=device)

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
        )

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

    def init_random_weights(self) -> None:
        """Random initialize the weights."""
        logger.info("Random initializing weights...")

        def __init_weights(device_id: int) -> None:
            dev_attrs = {
                "device": device_id,
                "dtype": torch.bfloat16,
            }
            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []

            with torch.cuda.device(device_id):
                # shared
                intermediates.extend(self.get_temp_vars(device_id, dev_attrs).get_params())
                for _ in range(self.NUM_DENSE_LAYERS):
                    # Each layer has its dedicated cache
                    caches.extend(self.get_cache_vars(device_id, dev_attrs).get_params())
                    params.extend(self.get_mla_mlp_layer_params(device_id, dev_attrs))

                for _ in range(self.NUM_MOE_LAYERS):
                    caches.extend(self.get_cache_vars(device_id, dev_attrs).get_params())
                    params.extend(self.get_mla_moe_layer_params(device_id, dev_attrs))

                # heads, embedding, and RoPE frequencies
                params.extend(self.get_llm_head_layer_params(device_id, dev_attrs))
                params.extend([torch.randn(self.vocab_size_full, self.hidden_size, **dev_attrs)])
                freqs_cis = self._gen_freqs_cis()
                params.extend([freqs_cis.to(device_id)])

                profile_logs = get_profile_log_tensor(device=device_id, num_max_insts=65536)
                self.multi_devices_results[device_id] = (
                    intermediates,
                    caches,
                    params,
                    profile_logs,
                )

        threads = []
        for device_id in range(self.num_devices):
            thread = threading.Thread(target=__init_weights, args=(device_id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        # Prepare money for all devices
        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                dsa_show_hands_prepare_money(
                    True,  # enable fused op
                    params,
                    intermediates,
                    caches,
                    profile_logs,
                )

    def load_embedding_weights(self, total_shards: int, device_id: int) -> torch.Tensor:
        """Load the embedding weights for the given device."""
        # the first shard is for embedding
        weight_prefix = "model.safetensors-00001-of"
        embed_weights_file = f"{weight_prefix}-{total_shards:05d}.safetensors"
        embed_weights_file_path = os.path.join(self.model_path, embed_weights_file)
        state_dict = load_file(embed_weights_file_path, device=f"cuda:{device_id}")
        return state_dict["model.embed_tokens.weight"]

    def get_total_shards(self, model_path: str) -> int:
        """Get the total number of shards by counting safetensors files in the directory."""
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        return len(safetensors_files)

    def from_pretrained(self, model_path: str) -> None:
        """Load the model weights from the given path."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        total_shards = self.get_total_shards(model_path)

        def __load_weights(device_id: int) -> None:
            index_file = "model.safetensors.index.json"
            with open(os.path.join(self.model_path, index_file)) as f:
                weights_index = json.load(f)
            weight_map = weights_index["weight_map"]

            weight_files = self.get_weight_files(weight_map, device_id)

            dev_attrs = {
                "device": device_id,
                "dtype": torch.bfloat16,
            }

            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            state_dicts = {}

            start_time = time.time()
            with torch.cuda.device(device_id):
                intermediates.extend(self.get_temp_vars(device_id, dev_attrs).get_params())
                for _ in range(self.NUM_LAYERS):
                    caches.extend(self.get_cache_vars(device_id, dev_attrs).get_params())
                logger.info(f"Created intermediates and caches for device {device_id}")

                for weight_file in weight_files:
                    state_dict = load_file(
                        os.path.join(self.model_path, weight_file),
                        device=f"cuda:{device_id}",
                    )
                    logger.info(f"Loaded weights from {weight_file} for device {device_id}")
                    state_dicts.update(state_dict)

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

                embed_weights = self.load_embedding_weights(total_shards, device_id)
                params.extend([embed_weights])

                # RoPE frequencies
                freqs_cis = self._gen_freqs_cis()
                params.extend([freqs_cis.to(device_id)])

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
        for device_id in range(self.num_devices):
            thread = threading.Thread(target=__load_weights, args=(device_id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        # Prepare money for all devices
        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                dsa_show_hands_prepare_money(
                    True,  # enable fused op
                    params,
                    intermediates,
                    caches,
                    profile_logs,
                )

    def forward(
        self,
        token_id: torch.Tensor,
    ) -> list[DeviceResult]:
        dsa_show_hands(token_id.cpu())
        return [self._get_device_result(device_id) for device_id in range(self.num_devices)]

    def reset_sequence(self) -> None:
        dsa_show_hands_reset(self.placeholder)

    def cleanup(self) -> None:
        dsa_show_hands_go_home(self.placeholder)

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
