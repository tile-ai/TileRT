"""rmsnorm_raw op wrapper."""

import torch

_EPS_FOR_DIM = {2048: 1e-06, 6144: 1e-05, 8192: 1e-06}
SUPPORTED_DIMS = tuple(_EPS_FOR_DIM)


def rmsnorm_raw(
    hidden_in: torch.Tensor, gamma: torch.Tensor, hidden_out: torch.Tensor
) -> torch.Tensor:
    torch.ops.tilert.rmsnorm_raw_op(hidden_in, gamma, hidden_out)
    return hidden_out


class RMSNormRaw:

    def __init__(self, dim: int):
        assert dim in SUPPORTED_DIMS, f"dim must be one of {SUPPORTED_DIMS}"
        self.dim = dim
        self.eps = _EPS_FOR_DIM[dim]

    def golden_forward(self, hidden_in: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(self, hidden_in: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        hidden_out = torch.empty_like(hidden_in)
        return rmsnorm_raw(hidden_in, gamma, hidden_out)
