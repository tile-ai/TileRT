"""TileRT Python package.

Two backend libraries ship with TileRT — one per model family:

  - ``libtilert_dsv32.so``  (DeepSeek-V3.2)
  - ``libtilert_glm5.so``   (GLM-5)

They are NOT loaded at import time. The caller selects a backend via
``load_backend(model_type)`` (done automatically by ``tilert.generate``).
Only one backend may be loaded per process — both register the ``tilert``
torch-op namespace. Run DSv3.2 and GLM-5 in separate processes.
"""

import ctypes
import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch

if not hasattr(torch, "ops"):
    raise RuntimeError("PyTorch is required but torch.ops is not available")

try:
    __version__ = pkg_version("tilert")
except PackageNotFoundError:
    __version__ = "0.0.0"


def init_logging() -> logging.Logger:
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
    )
    return logging.getLogger(__name__)


logger = init_logging()

_BACKENDS = {
    "deepseek_v3_2": "libtilert_dsv32.so",
    "glm5": "libtilert_glm5.so",
}

_loaded_backend: str | None = None


def load_backend(model_type: str) -> None:
    """Load the backend for ``model_type`` (lazy, once per process).

    DeepSeek-V3.2 and GLM-5 ship as separate libraries; the matching one is
    loaded on first use. Loading a second, different backend in the same
    process raises (both libraries define the ``tilert`` op namespace).
    """
    global _loaded_backend
    so_name = _BACKENDS.get(model_type)
    if so_name is None:
        raise ValueError(f"Unknown model_type {model_type!r}. Supported: {sorted(_BACKENDS)}")
    if _loaded_backend is not None:
        if _loaded_backend != so_name:
            raise RuntimeError(
                f"TileRT backend '{_loaded_backend}' already loaded; cannot load "
                f"'{so_name}' in the same process. Run {model_type} in a fresh process."
            )
        return
    pkg_dir = Path(__file__).parent
    lib_path = pkg_dir / so_name
    if not lib_path.exists():
        fallback = pkg_dir / "libtilert.so"
        if not fallback.exists():
            raise RuntimeError(f"Backend library not found: {lib_path}.")
        lib_path = fallback
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_LAZY)
    torch.ops.load_library(str(lib_path))
    _loaded_backend = so_name
    logger.info(
        "Loaded TileRT backend %s (%s) for model_type=%s", so_name, lib_path.name, model_type
    )


from .tilert_init import tilert_init  # noqa: E402

__all__ = [
    "logger",
    "load_backend",
    "tilert_init",
    "__version__",
]
