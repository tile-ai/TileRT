"""DSA show hands for deepseek v3.2."""

import json
import os
import sys
import threading
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.modules.dsa import Dsa
from tilert.models.glm_5._dsa_v32.modules.mtp import MTP
from tilert.models.glm_5._dsa_v32.temp_var_indices import Idx, validate_temp_vars_layout
from tilert.models.utils import precompute_freqs_cis
from tilert.utils import get_profile_log_tensor

__all__ = ["ShowHandsDSALayer", "_extract_ffn_ops", "_get_moe_weight_keys"]


DeviceResult = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]


def _mark_weights_initialized(module: TileRTModule) -> None:
    """Recursively mark a module and all sub-ops as having initialized tilert weights."""
    module.is_tilert_weights_init = True
    if hasattr(module, "exec_seq"):
        for op in module.exec_seq:
            _mark_weights_initialized(op)


def _extract_ffn_ops(dsa: "Dsa") -> list:
    """Extract Moe/Mlp op objects from a Dsa's layer blocks.

    Returns a list of length n_layers where each element is a Moe or Mlp instance.
    """
    from tilert.models.glm_5._dsa_v32.modules.mlp import MlpBlock
    from tilert.models.glm_5._dsa_v32.modules.moe import MoeBlock

    ffn_ops = []
    for block in dsa.exec_seq:
        if isinstance(block, MoeBlock):
            op = block.moe
            _mark_weights_initialized(op)
            ffn_ops.append(op)
        elif isinstance(block, MlpBlock):
            op = block.mlp
            _mark_weights_initialized(op)
            ffn_ops.append(op)

    assert (
        len(ffn_ops) == dsa.model_args.n_layers
    ), f"Expected {dsa.model_args.n_layers} FFN ops, got {len(ffn_ops)}"
    return ffn_ops


def _get_moe_weight_keys(dsa: "Dsa") -> set[str]:
    """Get state_dict keys that belong exclusively to MOE/MLP ops in this Dsa."""
    from tilert.models.glm_5._dsa_v32.modules.mlp import MlpBlock
    from tilert.models.glm_5._dsa_v32.modules.moe import MoeBlock

    moe_keys: set[str] = set()
    mla_keys: set[str] = set()
    for block, prefix, suffix in zip(dsa.exec_seq, dsa.prefix_seq, dsa.suffix_seq):
        if isinstance(block, (MoeBlock, MlpBlock)):
            ffn = block.moe if isinstance(block, MoeBlock) else block.mlp
            for alias in ffn.get_tilert_weights_alias():
                moe_keys.add(f"{prefix}{alias}{suffix}")
            for alias in block.mla.get_tilert_weights_alias():
                mla_keys.add(f"{prefix}{alias}{suffix}")
    return moe_keys - mla_keys


def dsa_show_hands_prepare_money(
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    cache_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
    forward_max_seq_len: int,
    with_mtp: bool = False,
    is_glm5: bool = False,
) -> Any:
    """Prepare money for show hands"""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_prepare_money{glm5_flag}"
    if mtp_flag:
        return getattr(torch.ops.tilert, func_name)(params, temp_vars, cache_vars, profile_logs)
    return getattr(torch.ops.tilert, func_name)(
        params, temp_vars, cache_vars, profile_logs, forward_max_seq_len
    )


def dsa_show_hands(token_id: torch.Tensor, with_mtp: bool = False, is_glm5: bool = False) -> Any:
    """Show hands with native MT"""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)(token_id)


def dsa_show_hands_reset(with_mtp: bool = False, is_glm5: bool = False) -> Any:
    """Reset show one hand"""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_reset{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)()


def dsa_show_hands_go_home(with_mtp: bool = False, is_glm5: bool = False) -> Any:
    """Go home"""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_go_home{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)()


def dsa_show_hands_set_sampling_seed(
    seed: int, with_mtp: bool = False, is_glm5: bool = False
) -> Any:
    """Set the sampling seed (request-level, fixed for the entire request).

    Args:
        seed: The sampling seed value.
    """
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_set_sampling_seed{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)(seed)


def dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(
    num_valid_tokens: int, is_glm5: bool = False
) -> Any:
    """Set the number of valid (non-padding) tokens for prefill mode.

    This controls how many tokens are copied from draft_tokens to predicted_tokens
    during prefill. Should be called before forward() when the chunk has padding.

    Args:
        num_valid_tokens: Number of valid tokens in the chunk (1-4).
    """
    mtp_flag = "_mtp_e2e"
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_set_prefill_valid_tokens{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)(num_valid_tokens)


def dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token: int, is_glm5: bool = False) -> Any:
    """Set the extra token for MTP[0] shifted input during prefill.

    Args:
        token: The extra prompt token id (int32).
    """
    mtp_flag = "_mtp_e2e"
    glm5_flag = "_glm5" if is_glm5 else ""
    func_name = f"dsa{mtp_flag}_show_hands_set_prefill_mtp_extra_token{glm5_flag}"
    return getattr(torch.ops.tilert, func_name)(token)


class ShowHandsDSALayer:
    """Show hands DSA for deepseek v3.2."""

    def __init__(
        self,
        model_args: ModelArgs,
        model_path: str = "",
        with_weight_conversion: bool = True,
        with_mtp: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
    ) -> None:
        validate_temp_vars_layout()
        print(f"Model args: {model_args.arch_name}")
        for k_arg, v_arg in model_args.__dict__.items():
            print(f" - {k_arg}: {v_arg}")
        self.model_args = model_args
        self.is_glm5 = self.model_args.arch_name == "glm_5"
        assert self.model_args.arch_name in ["deepseek_v3_2", "glm_5"]

        self.num_devices = 8
        self.forward_max_seq_len = 4

        self.model_path = model_path
        self.with_weight_conversion = with_weight_conversion
        self.with_mtp = with_mtp

        self.multi_devices_results: list[DeviceResult | None] = [None] * torch.cuda.device_count()
        self._dsa_objects: list[Dsa | None] = [None] * torch.cuda.device_count()

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp

    def _gen_freqs_cis(self) -> torch.Tensor:
        freqs_cis = precompute_freqs_cis(self.model_args)
        return torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1)

    def load_device_weights(
        self,
        model_path: str,
        device_id: int,
        extra_keys: list,
        skip_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        index_file = "model.safetensors.index.json"
        with open(os.path.join(model_path, index_file), encoding="utf-8") as f:
            weights_index = json.load(f)
        weight_file_map = weights_index["weight_map"]

        weights_list = [_k for _k in weight_file_map.keys() if _k.endswith(f"dev_{device_id}")]
        weights_list = [*weights_list, *extra_keys]

        if skip_keys:
            weights_list = [k for k in weights_list if k not in skip_keys]

        target_files = set()
        for weight_key in weights_list:
            weight_file = weight_file_map[weight_key]
            target_files.add(weight_file)

        state_dicts = {}
        weights_set = set(weights_list)
        for weight_file in target_files:
            filepath = os.path.join(model_path, weight_file)
            if skip_keys:
                logger.info(
                    f"Selectively loading weights from {weight_file} for device {device_id}"
                )
                with safe_open(filepath, framework="pt", device=f"cuda:{device_id}") as f:
                    for key in f.keys():
                        if key in weights_set:
                            state_dicts[key] = f.get_tensor(key)
                torch.cuda.empty_cache()
            else:
                logger.info(f"Loading weights from {weight_file} for device {device_id}")
                state_dict = load_file(filepath, device=f"cuda:{device_id}")
                state_dicts.update(state_dict)
                del state_dict
                torch.cuda.empty_cache()

        state_dicts["freqs_cis"] = self._gen_freqs_cis().to(device_id)
        return state_dicts

    def update_sampling_config(
        self, temperature: float, top_p: float, top_k: int, use_topp: bool = True
    ) -> None:
        """Update sampling config, re-capturing CUDA graphs if parameters changed."""
        new_config = (temperature, top_p, top_k, use_topp)
        current_config = (self.temperature, self.top_p, self.top_k, self.use_topp)
        if new_config == current_config:
            return

        print(
            f"Recapturing CUDA graphs: "
            f"temperature={temperature}, top_p={top_p}, top_k={top_k}, use_topp={use_topp}"
        )

        if self.with_mtp:
            dsa_show_hands_go_home(True, self.is_glm5)
            dsa_show_hands_go_home(False, self.is_glm5)
        else:
            dsa_show_hands_go_home(False, self.is_glm5)

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp

        for device_id in range(self.num_devices):
            result = self.multi_devices_results[device_id]
            if result is not None:
                intermediates = result[0]
                intermediates[Idx.SAMPLING_CONFIG].copy_(
                    torch.tensor(
                        [temperature, top_p, float(top_k), 1.0 if use_topp else 0.0],
                        dtype=torch.float32,
                        device=f"cuda:{device_id}",
                    )
                )

        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                dsa_show_hands_prepare_money(
                    params,
                    intermediates,
                    caches,
                    profile_logs,
                    self.forward_max_seq_len,
                    self.with_mtp,
                    self.is_glm5,
                )
                if self.with_mtp:
                    dsa_show_hands_prepare_money(
                        params[: self._base_params_count],
                        intermediates,
                        caches[: self._base_caches_count],
                        profile_logs,
                        self.forward_max_seq_len,
                        False,
                        self.is_glm5,
                    )

    @staticmethod
    def tot_size_in_bytes_aligned(temp_vars: list[torch.Tensor], aligned_size: int) -> int:
        tot_size: int = 0
        for param in temp_vars:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            tot_size += aligned_param_size
        return tot_size

    def generate_params_with_continuous_storage(
        self, temp_vars: list[torch.Tensor], device: torch.device, aligned_size: int = 1024
    ) -> list[torch.Tensor]:
        tot_size = self.tot_size_in_bytes_aligned(temp_vars, aligned_size)
        cloned_params = []
        large_tensor = torch.zeros(tot_size, device=device, dtype=torch.uint8)
        offset = 0
        for param in temp_vars:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            cloned_params.append(
                large_tensor[offset : offset + param.nbytes].view(param.dtype).view(param.shape)
            )
            offset += aligned_param_size
        return cloned_params

    def _init_weights(
        self,
        model_path: str | None,
        cached_ffn_ops_per_device: dict[int, list] | None = None,
        skip_keys_per_device: dict[int, set[str]] | None = None,
    ) -> None:
        """Load the model weights from the given path or generate random weights.

        Args:
            model_path: Path to the model weights directory.
            cached_ffn_ops_per_device: Optional dict mapping device_id to cached FFN ops.
                When provided, these ops are injected into the Dsa and their weights
                are not re-loaded from disk.
            skip_keys_per_device: Optional dict mapping device_id to safetensors keys
                to skip during loading. Used together with cached_ffn_ops_per_device.
        """
        self._v2_p2p: dict = {}

        def __load_weights(device_id: int, model_path: str | None) -> None:
            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            state_dicts = {}
            start_time = time.time()
            with torch.cuda.device(device_id):
                assert model_path is not None
                skip_keys = (
                    skip_keys_per_device.get(device_id)
                    if skip_keys_per_device is not None
                    else None
                )
                state_dicts = self.load_device_weights(
                    model_path,
                    device_id,
                    [
                        "model.embed_tokens.weight",
                        f"layer_{self.model_args.n_layers}_lm_head.weight_dev_{device_id}",
                        f"layer_{self.model_args.n_layers}_model.norm.weight_dev_{device_id}",
                    ],
                    skip_keys=skip_keys,
                )

                cached_ffn_ops = (
                    cached_ffn_ops_per_device.get(device_id)
                    if cached_ffn_ops_per_device is not None
                    else None
                )
                dsa = Dsa(
                    self.model_args,
                    device_id,
                    self.num_devices,
                    cached_ffn_ops=cached_ffn_ops,
                )
                dsa.init_tilert_weights(state_dicts)
                self._dsa_objects[device_id] = dsa
                params.extend(dsa.get_weights_list())
                caches.extend(dsa.get_cache_vars())

                if device_id == 0:
                    self._v2_p2p[device_id] = {
                        "peer_bufs": dsa.v2_peer_bufs,
                    }
                else:
                    self._v2_p2p[device_id] = {
                        "ll_buf": dsa.v2_ll_buf,
                    }
                intermediates.extend(
                    self.generate_params_with_continuous_storage(
                        dsa.get_temp_vars(
                            1,
                            self.forward_max_seq_len,
                            {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "top_k": self.top_k,
                                "use_topp": self.use_topp,
                            },
                        ),
                        device_id,
                    )
                )

                sampling_config = intermediates[Idx.SAMPLING_CONFIG]
                sampling_config.copy_(
                    torch.tensor(
                        [
                            self.temperature,
                            self.top_p,
                            float(self.top_k),
                            1.0 if self.use_topp else 0.0,
                        ],
                        dtype=torch.float32,
                        device=device_id,
                    )
                )

                base_params_count = len(params)
                base_caches_count = len(caches)

                if self.with_mtp:
                    from tilert.models.glm_5._dsa_v32.modules.mla_v2 import (
                        PureMlaV2,
                        SparseSelectMlaV2,
                    )

                    mtp_kwargs: dict = {}
                    mtp_kwargs["mla_cls"] = SparseSelectMlaV2 if device_id == 0 else PureMlaV2
                    mtp_kwargs["mla_num_devices"] = 1 if device_id == 0 else self.num_devices - 1
                    if device_id == 0:
                        mtp_kwargs["mla_kwargs"] = {
                            "peer_bufs": dsa.v2_peer_bufs,
                        }
                    else:
                        mtp_kwargs["mla_kwargs"] = {"ll_buf": dsa.v2_ll_buf}
                    mtp = MTP(self.model_args, device_id, self.num_devices, **mtp_kwargs)
                    mtp.init_tilert_weights(state_dicts)
                    params.extend(mtp.get_weights_list())
                    caches.extend(mtp.get_cache_vars())
                    logger.info(f"Loaded real MTP weights for device {device_id}")

                profile_logs = get_profile_log_tensor(device=device_id, num_max_insts=65536)
                result = (intermediates, caches, params, profile_logs)
                self.multi_devices_results[device_id] = result
                self._base_params_count = base_params_count
                self._base_caches_count = base_caches_count

            del state_dicts
            torch.cuda.empty_cache()
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

        if self._v2_p2p:
            gpu0 = self._v2_p2p[0]
            peer_bufs_cpu = torch.zeros(self.num_devices - 1, dtype=torch.int64)
            for i in range(self.num_devices - 1):
                dev_id = i + 1
                peer_bufs_cpu[i] = self._v2_p2p[dev_id]["ll_buf"].data_ptr()
            gpu0["peer_bufs"].copy_(peer_bufs_cpu)
            logger.info(
                "V2 P2P exchange complete: peer_bufs (ll_buf)=%s",
                [hex(int(x)) for x in peer_bufs_cpu],
            )

        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                dsa_show_hands_prepare_money(
                    params,
                    intermediates,
                    caches,
                    profile_logs,
                    self.forward_max_seq_len,
                    self.with_mtp,
                    self.is_glm5,
                )
                if self.with_mtp:
                    dsa_show_hands_prepare_money(
                        params[: self._base_params_count],
                        intermediates,
                        caches[: self._base_caches_count],
                        profile_logs,
                        self.forward_max_seq_len,
                        False,
                        self.is_glm5,
                    )

    def from_pretrained(self, model_path: str) -> None:
        """Load the model weights from the given path."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        self._init_weights(model_path)

    def from_pretrained_with_cache(
        self,
        model_path: str,
        cached_ffn_ops_per_device: dict[int, list],
        skip_keys_per_device: dict[int, set[str]],
    ) -> None:
        """Load weights with cached MOE/MLP ops."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        self._init_weights(
            model_path,
            cached_ffn_ops_per_device=cached_ffn_ops_per_device,
            skip_keys_per_device=skip_keys_per_device,
        )

    def init_random_weights(self) -> None:
        """Generate random weights."""
        self._init_weights(None)

    def forward(
        self,
        token_id: torch.Tensor,
        with_mtp: bool | None = None,
    ) -> list[DeviceResult]:
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        dsa_show_hands(token_id.cpu(), active_mtp, self.is_glm5)
        return [self._get_device_result(device_id) for device_id in range(self.num_devices)]

    def set_sampling_seed(self, seed: int, with_mtp: bool | None = None) -> None:
        """Set the sampling seed for top-p sampling.

        The seed is fixed for the entire request. Position provides per-step variation.

        Args:
            seed: The sampling seed value.
            with_mtp: Override MTP mode for this call. Defaults to self.with_mtp.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        dsa_show_hands_set_sampling_seed(seed, active_mtp, self.is_glm5)

    def reset_sequence(self) -> None:
        if self.with_mtp:
            dsa_show_hands_reset(True, self.is_glm5)
            dsa_show_hands_reset(False, self.is_glm5)
        else:
            dsa_show_hands_reset(False, self.is_glm5)

    def cleanup(self) -> None:
        if self.with_mtp:
            dsa_show_hands_go_home(True, self.is_glm5)
            dsa_show_hands_go_home(False, self.is_glm5)
        else:
            dsa_show_hands_go_home(False, self.is_glm5)

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

    def set_prefill_valid_tokens(self, num_valid_tokens: int) -> None:
        """Set the number of valid tokens for prefill mode.

        This controls how many tokens are copied from draft_tokens to predicted_tokens
        during prefill. Should be called before forward() when the chunk has padding.

        Args:
            num_valid_tokens: Number of valid tokens in the chunk (1-4).
        """
        dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(num_valid_tokens, self.is_glm5)

    def set_prefill_mtp_extra_token(self, token: int) -> None:
        """Set the extra token for MTP[0] shifted input during prefill.

        Args:
            token: The prompt token at (cur_pos + mtp_seq_len).
        """
        dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token, self.is_glm5)

    def get_next_draft_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get next_draft_tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            next_draft_tokens tensor of shape [1, MTP_SEQ_LEN].
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.NEXT_DRAFT_TOKENS]

    def get_num_accepted(self, device_id: int = 0) -> int:
        """Get number of accepted tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            Number of accepted tokens.
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return int(intermediates[Idx.ACCEPTED_TOKENS][0].item())

    def get_predicted_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get predicted_tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            predicted_tokens tensor containing main model predictions.
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.PREDICTED_TOKENS]

    def get_logits(self, device_id: int = 0) -> torch.Tensor:
        """Get logits from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size] (FP32).
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.LOGITS_OUT]

    def get_top_n_logprobs(self, device_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-N log-probabilities and token IDs from the top_p kernel.

        Args:
            device_id: Device ID to get results from.

        Returns:
            Tuple of (log_probs, token_ids):
              - log_probs: [batch, seq_len, 256] FP32
              - token_ids: [batch, seq_len, 256] INT32
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return (
            intermediates[Idx.TOP_N_LOG_PROBS],
            intermediates[Idx.TOP_N_INDICES],
        )

    def get_token_logprob(self, device_id: int = 0) -> torch.Tensor:
        """Get log-probability of the sampled token (from TOP_P_SCORES).

        Args:
            device_id: Device ID to get results from.

        Returns:
            Tensor of shape [batch, seq_len] (FP32).
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.TOP_P_SCORES]

    def set_logprobs_enabled(self, enabled: bool) -> None:
        """Enable or disable logprobs export in the top_p kernel.

        Args:
            enabled: True to enable logprobs export, False to disable.
        """
        flag_val = 1 if enabled else 0
        for device_id in range(self.num_devices):
            intermediates, _, _, _ = self._get_device_result(device_id)
            intermediates[Idx.LOGPROBS_FLAG].fill_(flag_val)
