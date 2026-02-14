"""TileRT Python package."""

import ctypes
import logging
from pathlib import Path
from typing import Any

import torch

if not hasattr(torch, "ops"):
    raise RuntimeError("PyTorch is required but torch.ops is not available")

from .__version__ import __version__


def init_logging() -> logging.Logger:
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
    )
    return logging.getLogger(__name__)


logger = init_logging()


def _load_library(filename: str) -> Any:
    """Load the C++ library.

    Args:
        filename: Name of the library file.

    Returns:
        Any: The loaded library.

    Raises:
        RuntimeError: If the library cannot be loaded.
    """
    lib_path = Path(__file__).parent / filename

    try:
        torch.ops.load_library(str(lib_path))
        return lib_path
    except Exception as e:
        raise RuntimeError(f"Failed to load library from {lib_path}") from e


_load_library("libtilert.so")


from . import models  # noqa: E402
from .models import deepseek_v3_2  # noqa: E402
from .tilert_init import tilert_init  # noqa: E402

__all__ = [
    "logger",
    "tilert_init",
    "models",
    "deepseek_v3_2",
    "__version__",
]
