"""Tilert init operation module."""

import torch

__all__ = [
    "tilert_init",
    "tilert_force_init",
]


def tilert_init() -> None:
    """Tilert init operation."""
    torch.ops.tilert.tilert_init_op()


def tilert_force_init() -> None:
    """Tilert force init operation."""
    torch.ops.tilert.tilert_force_init_op()
