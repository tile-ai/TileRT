"""PaddedAllReduceAdd operation module."""

from enum import Enum

import torch

from tilert.models.base import TileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "padded_allreduce_add",
    "PaddedAllReduceAdd",
]


def padded_allreduce_add(
    partial_buf: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """Padded AllReduce + residual add for Device Group A (GPU 0).

    GPU 0 contributes zeros to the 8-GPU AllReduce, then adds the residual.

    Args:
        partial_buf: Zero-filled partial buffer [1, L, hidden_dim] bf16.
        x_in: Residual input [1, L, hidden_dim] bf16.
        flag: AllReduce sync flag.
        vec_out: Output tensor [1, L, hidden_dim] bf16.
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("deepseek_v3_2" or "glm_5").
        compute_kernel_type: Compute kernel type ("bf16").
    """
    torch.ops.tilert.padded_allreduce_add_op(
        partial_buf, x_in, flag, vec_out, profile_logs, model_arch, compute_kernel_type
    )


class PaddedAllReduceAddAlgorithm(Enum):
    """PaddedAllReduceAdd algorithm."""

    BF16 = "bf16"


class PaddedAllReduceAdd(TileRTModule):
    """PaddedAllReduceAdd module — zero-partial AllReduce + residual add."""

    _SUPPORTED_ALGORITHMS = {
        "deepseek_v3_2": [PaddedAllReduceAddAlgorithm.BF16],
        "glm_5": [PaddedAllReduceAddAlgorithm.BF16],
    }

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int = 0,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.dim = self.model_args.dim

        self.partial_buf: torch.Tensor | None = None

        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_var_init = False

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """Allocate output buffer and persistent zero-filled partial buffer.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        self.partial_buf = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{self.device_id}")
        self.is_var_init = True

    def golden_forward(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        """Golden reference: allreduce(zeros) + x_in = x_in (single-GPU).

        On a single GPU, allreduce of zeros returns zeros, so output = x_in.

        Args:
            x_in: Residual input [1, L, hidden_dim].

        Returns:
            Output tensor (copy of x_in).
        """
        return x_in.clone()

    def tilert_forward(
        self,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        """Run TileRT kernel forward.

        Args:
            x_in: Residual input [1, L, hidden_dim].
            flag: AllReduce sync flag.

        Returns:
            Output tensor [1, L, hidden_dim].
        """
        assert self.hidden_out is not None
        assert self.partial_buf is not None
        assert self.profile_logs is not None
        padded_allreduce_add(
            self.partial_buf,
            x_in,
            flag,
            self.hidden_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in)
