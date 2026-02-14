"""EHProjAllReduce operation module."""

from dataclasses import dataclass
from enum import Enum

# import torch.nn.functional as F
import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "eh_proj_allreduce",
    "EHProjAllReduceTilertWeightsAlias",
]


def eh_proj_allreduce(
    vec_in_enorm: torch.Tensor,
    vec_in_hnorm: torch.Tensor,
    w_eh: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
) -> None:
    """
    Fused operation of EHProj and allreduce.

    Args:
        vec_in_enorm: Input tensor of shape (1, seq_len, 7168).
        vec_in_hnorm: Input tensor of shape (1, seq_len, 7168).
        w_eh: Input tensor of shape (7168, 1792) or (128, 7, 56, 256).
        flag: Input tensor.
        vec_out: Output tensor of shape (1, seq_len, 7168).
        profile_logs: Profile logs tensor. This is a 1D tensor of shape
                      (num_sms,) to store the profile logs of the eh_proj_allreduce
                      operation, where num_sms is the number of SMs on the
                      device.
    """
    dim = vec_in_enorm.shape[-1]
    if dim == 7168:
        func_call = torch.ops.tilert.eh_proj_allreduce_op
    elif dim == 6144:
        func_call = torch.ops.tilert.eh_proj_allreduce_glm5_op
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    func_call(vec_in_enorm, vec_in_hnorm, w_eh, flag, vec_out, profile_logs)


class EHProjAllReduceAlgorithm(Enum):
    """EHProjAllReduce algorithm"""

    GENERAL = "general"


class EHProjAllReduceWeightsConverter(TilertWeightsConverter):
    """EHProj weights converter"""

    def convert_to_general(self, weights_list: list[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        Convert the weights to general format.

        Args:
            weights_list: List of weights.

        Returns:
            Tuple of weights.
        """
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2" or args.arch_name == "glm_5"
        dim = args.dim
        num_sms = 128
        dim_per_sm = dim // num_sms
        in_dim = dim * 2
        in_dim_per_device = in_dim // self.num_devices
        stages = in_dim_per_device // 256

        with torch.inference_mode():
            (proj_weights,) = weights_list
            proj_weights = proj_weights.reshape(num_sms, dim_per_sm, stages, 256)
            proj_weights = proj_weights.transpose(1, 2)
            return (proj_weights.contiguous(),)


@dataclass
class EHProjAllReduceTilertWeightsAlias:
    """TileRT weights alias for EHProjAllReduce."""

    eh_proj_weights = "eh_proj_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.eh_proj_weights]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class EHProjAllReduce(TileRTModule):
    """EHProjAllReduce module"""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        algorithm: EHProjAllReduceAlgorithm = EHProjAllReduceAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim

        self.algorithm = algorithm

        # reference weights
        self.ref_proj: torch.Tensor | None = None

        # tilert weights
        self.tilert_proj: torch.Tensor | None = None

        # tilert vars
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self.tilert_weights_alias = EHProjAllReduceTilertWeightsAlias()

        # for device sharding, corresponding to the output of device_sharding
        # and input of tilert_forward
        self.tensor_alias: list[str] = [
            "eh_proj_weights",
        ]

        # reference tensor aliases
        self.ref_tensor_alias: list[str] = [
            "eh_proj.weight",
        ]

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return self.tilert_weights_alias.tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_proj]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str | None = None,  # e.g. model.layers.{layer_id}
    ) -> tuple[torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_dict: Dictionary of weights.
            key_prefix: Key prefix.
        Returns:
            Tuple of weights.
        """
        eh_proj_key = "eh_proj.weight"
        if key_prefix is not None:
            eh_proj_key = f"{key_prefix}.eh_proj.weight"

        eh_proj_weight = weights_dict[eh_proj_key]
        in_dim = eh_proj_weight.shape[1]
        out_dim = eh_proj_weight.shape[0]
        in_dim_per_device = in_dim // self.num_devices
        eh_proj_weight = eh_proj_weight.reshape(out_dim, self.num_devices, in_dim_per_device)
        eh_proj_weight = eh_proj_weight.transpose(0, 1)
        return (eh_proj_weight.contiguous(),)

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        key_prefix: str | None = None,
        device_id: int = 0,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary.
            device_id: Device ID.
        """
        sharded_list = self.device_sharding(state_dict, key_prefix)

        eh_proj_weight = sharded_list[0][device_id]

        self.ref_proj = eh_proj_weight

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.
        """
        assert self.algorithm is not None
        (self.tilert_proj,) = EHProjAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tensor_alias])

    def init_tilert_vars(self, batch_size: int, seq_len: int, device_id: int = 0) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        # tilert vars
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int = 0) -> None:
        """Initialize the random weights."""
        proj_weights = torch.randn(
            self.dim, self.dim * 2, dtype=torch.bfloat16, device=f"cuda:{device_id}"
        )

        tensor_list = [
            proj_weights,
        ]
        state_dict = dict(zip(self.ref_tensor_alias, tensor_list))

        self.init_reference_weights(state_dict, None, device_id)
        sharded_list = self.device_sharding(state_dict, None)
        sharded_state_dict = {
            alias: sharded_list[i][device_id] for i, alias in enumerate(self.tensor_alias)
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        vec_in_enorm: torch.Tensor,
        vec_in_hnorm: torch.Tensor,
        device_id: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for the down-project module.

        Args:
            vec_in_enorm: Input vector of shape (1, seq_len, 7168).
            vec_in_hnorm: Input vector of shape (1, seq_len, 7168).

        Returns:
            Output tensor.
        """
        assert self.ref_proj is not None
        bsz = vec_in_enorm.shape[0]
        assert bsz == 1

        vec_in_concat = torch.cat([vec_in_enorm, vec_in_hnorm], dim=-1)
        dim_per_device = (self.dim * 2) // self.num_devices
        vec_in_slice = vec_in_concat[
            ..., dim_per_device * device_id : dim_per_device * device_id + dim_per_device
        ]
        return vec_in_slice @ self.ref_proj.T

    def tilert_forward(
        self,
        vec_in_enorm: torch.Tensor,
        vec_in_hnorm: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        assert self.hidden_out is not None
        eh_proj_allreduce(
            vec_in_enorm, vec_in_hnorm, self.tilert_proj, flag, self.hidden_out, self.profile_logs
        )
        return self.hidden_out
