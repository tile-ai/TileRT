"""PaddedAllReduceAdd operation module."""

from enum import Enum

import torch

from tilert.models.base import TileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
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
    """Padded AllReduce + residual add for Device Group A (GPU 0)."""
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
        """Allocate output buffer and persistent zero-filled partial buffer."""
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
        """Golden reference: allreduce(zeros) + x_in = x_in (single-GPU)."""
        return x_in.clone()

    def tilert_forward(
        self,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        """Run TileRT kernel forward."""
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
        if self.flag_enable_profiling_log:
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in)
