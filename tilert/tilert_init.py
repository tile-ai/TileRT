"""Runtime warm-up entry points."""

import torch

__all__ = ["tilert_init", "tilert_force_init"]


def _has_op(name: str) -> bool:
    return hasattr(torch.ops.tilert, name)


def tilert_init() -> None:
    if _has_op("tilert_init_op"):
        torch.ops.tilert.tilert_init_op()
        return
    torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}")
    torch.cuda.synchronize()


def tilert_force_init() -> None:
    if _has_op("tilert_force_init_op"):
        torch.ops.tilert.tilert_force_init_op()
        return
    tilert_init()
