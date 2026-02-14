"""ProjxWis operation module."""

from dataclasses import dataclass

import torch

from tilert.models.base import TileRTModule
from tilert.models.common import init_func
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "projx_wis",
    "ProjxWis",
    "ProjxWisRefWeightsAlias",
    "ProjxWisTilertWeightsAlias",
]


def projx_wis(
    x_in: torch.Tensor,
    w: torch.Tensor,
    output: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Define the ProjxWis operation.

    Args:
        x_in: Input tensor.
        w: Weight tensor.
        output: Output tensor.
        profile_logs: Profile logs tensor.
    """
    if x_in.shape[-1] == 7168:
        torch.ops.tilert.proj_w_op(x_in, w, output, profile_logs)
    elif x_in.shape[-1] == 6144:
        torch.ops.tilert.proj_w_glm5_op(x_in, w, output, profile_logs)


@dataclass
class ProjxWisRefWeightsAlias:
    """Reference weights alias for ProjxWis."""

    w_weights = "self_attn.indexer.weights_proj.weight"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [self.w_weights]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ProjxWisTilertWeightsAlias:
    """TileRT weights alias for ProjxWis."""

    w_weights = "id_score_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.w_weights]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ProjxWis(TileRTModule):
    """ProjxWis module: linear projection for indexer score weights."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ProjxWisRefWeightsAlias | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = ProjxWisTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else ProjxWisRefWeightsAlias()
        )

        # Backward compatibility: expose list for load_weights_for_layer etc.
        self.ref_tensor_alias = self.ref_weights_alias.ref_tensor_alias

        self.ref_w: torch.Tensor | None = None
        self.tilert_w: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None

        self.dim = model_args.dim
        self.index_n_heads = model_args.index_n_heads

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_w]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: replicate weight for each device.

        Args:
            weights_map: Map from ref weight alias to tensor.

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.
        """
        w = weights_map[self.ref_weights_alias.w_weights][None, ...].repeat(self.num_devices, 1, 1)
        return {self.tilert_weights_alias.w_weights: w}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        w = state_dict[self.ref_weights_alias.w_weights]
        self.ref_w = w.detach().clone().to(torch.bfloat16)
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tilert_w = state_dict[self.tilert_weights_alias.w_weights].detach().clone()
        self.is_tilert_weights_init = True

    def init_random_weights(self) -> None:
        ref_w = init_func(torch.empty(self.index_n_heads, self.dim, dtype=torch.bfloat16))
        ref_state_dict = dict(zip(self.ref_weights_alias(), [ref_w]))
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        self.init_tilert_weights({k: v[self.device_id] for k, v in sharded.items()})

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        self.output = torch.zeros((batch_size, seq_len, self.index_n_heads), dtype=torch.bfloat16)
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.ref_w is not None
        return torch.nn.functional.linear(x_norm, self.ref_w)

    def tilert_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.tilert_w is not None
        assert self.output is not None
        assert self.profile_logs is not None
        projx_wis(x_norm, self.tilert_w, self.output, self.profile_logs)
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.output
