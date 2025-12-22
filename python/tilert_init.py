"""Tilert init operation module."""

import torch

__all__ = [
    "tilert_init",
    "tilert_force_init",
]


def tilert_init(
    placeholder: torch.Tensor | None = None,
) -> None:
    """Tilert init operation.

    Args:
        placeholder: torch.Tensor,
            A placeholder tensor.
    """
    if placeholder is None:
        placeholder = torch.zeros(0).to(torch.device("cuda"))
    torch.ops.tilert.tilert_init_op(
        placeholder,
    )


def tilert_force_init(
    placeholder: torch.Tensor | None = None,
) -> None:
    """Tilert force init operation.

    Args:
        placeholder: torch.Tensor,
            A placeholder tensor.
    """
    if placeholder is None:
        placeholder = torch.zeros(0).to(torch.device("cuda"))
    torch.ops.tilert.tilert_force_init_op(
        placeholder,
    )
