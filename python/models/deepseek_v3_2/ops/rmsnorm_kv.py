"""RMSNormKV operation module."""

from dataclasses import dataclass

import torch

from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "rmsnorm_kv",
    "KVRMSNorm",
    "KVRMSNormRefWeightsAlias",
    "KVRMSNormTilertWeightsAlias",
]


def rmsnorm_kv(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cur_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the RMSNormKV operation.

    Args:
        kv: Input tensor.
        gamma: Weight tensor.
        cur_pos: Current position tensor.
        kv_cache: Output tensor.
        profile_logs: Profile logs tensor.
    """
    torch.ops.tilert.rmsnorm_kv_op(kv, gamma, cur_pos, kv_cache, profile_logs)


@dataclass
class KVRMSNormRefWeightsAlias:
    """Reference weights alias for KVRMSNorm."""

    kv_norm_weight = "self_attn.kv_a_layernorm.weight"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.kv_norm_weight]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class KVRMSNormTilertWeightsAlias:
    """TileRT weights alias for KVRMSNorm."""

    kv_norm_gamma = "kv_rmsnorm_gamma"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.kv_norm_gamma]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class KVRMSNorm(TileRTModule):
    """KVRMSNorm module: RMSNorm on KV tensor with in-place write to kv_cache."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: KVRMSNormRefWeightsAlias | None = None,
        tilert_weights_alias: KVRMSNormTilertWeightsAlias | None = None,
        layer_idx: int = 0,
        golden_weights_dir: str = "",
        tilert_weights_dir: str = "",
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
            layer_idx=layer_idx,
            golden_weights_dir=golden_weights_dir,
            tilert_weights_dir=tilert_weights_dir,
        )

        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else KVRMSNormTilertWeightsAlias()
        )
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else KVRMSNormRefWeightsAlias()
        )

        self.kv_lora_rank = self.model_args.kv_lora_rank
        self.eps = self.model_args.eps

        self.ref_norm_gamma: torch.Tensor | None = None
        self.tilert_kv_norm_weight: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_kv_norm_weight]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: replicate gamma for each device.

        Args:
            weights_map: Map from ref weight alias to tensor.

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        gamma = weights_map[self.ref_weights_alias.kv_norm_weight][None, ...].repeat(
            self.num_devices, 1
        )
        return {self.tilert_weights_alias.kv_norm_gamma: gamma}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize reference weights from state dict."""
        self.ref_norm_gamma = state_dict[self.ref_weights_alias.kv_norm_weight].contiguous()
        assert (
            self.ref_norm_gamma.shape[-1] == self.kv_lora_rank
        ), f"kv_norm weight shape must be ({self.kv_lora_rank},), got {self.ref_norm_gamma.shape}"

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize TileRT weights from state dict."""
        gamma = state_dict[self.tilert_weights_alias.kv_norm_gamma]
        self.tilert_kv_norm_weight = gamma.float().detach().clone().contiguous()

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """Allocate TileRT profiling buffer."""
        del batch_size, seq_len
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def init_random_weights(self) -> None:
        """Initialize random reference and TileRT weights for testing."""
        ref_state_dict = {
            self.ref_weights_alias.kv_norm_weight: torch.randn(
                self.kv_lora_rank, dtype=torch.float32
            ),
        }
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        self.init_tilert_weights({k: v[self.device_id] for k, v in sharded.items()})

    def golden_forward(
        self, kv: torch.Tensor, kv_cache: torch.Tensor, start_pos: int, bsz: int, seqlen: int
    ) -> None:
        """Reference forward: RMSNorm and write to kv_cache."""
        assert self.ref_norm_gamma is not None
        end_pos = start_pos + seqlen
        out = torch.nn.functional.rms_norm(
            kv.float(), [kv.size(-1)], self.ref_norm_gamma, self.eps
        ).to(kv.dtype)
        kv_cache[:bsz, start_pos:end_pos].copy_(out)

    def tilert_forward(
        self, kv: torch.Tensor, kv_cache: torch.Tensor, start_pos: int, bsz: int, seqlen: int
    ) -> None:
        del seqlen
        assert self.tilert_kv_norm_weight is not None
        assert self.profile_logs is not None
        cur_pos = torch.tensor([start_pos], dtype=torch.int32, device=kv.device)
        rmsnorm_kv(kv, self.tilert_kv_norm_weight, cur_pos, kv_cache[:bsz], self.profile_logs)
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )

    def __call__(
        self, kv: torch.Tensor, kv_cache: torch.Tensor, start_pos: int, bsz: int, seqlen: int
    ) -> None:
        if self.flag_enable_tilert:
            return self.tilert_forward(kv, kv_cache, start_pos, bsz, seqlen)
        return self.golden_forward(kv, kv_cache, start_pos, bsz, seqlen)
