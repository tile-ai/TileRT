"""TileRT: tile-based runtime for ultra-low-latency LLM inference."""

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
    level_name = os.environ.get("TILERT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
    )
    return logging.getLogger(__name__)


logger = init_logging()
_BACKENDS = {
    "deepseek_v3_2": "libtilert_dsv32.so",
    "glm5": "libtilert_glm5.so",
    "glm5_2_rocm": "libtilert_glm52_rocm.so",
}
_TORCH_FOR_LIB = {
    "libtilert_dsv32.so": ("2.11", False),
    "libtilert_glm5.so": ("2.11", False),
    "libtilert_glm52_rocm.so": ("2.12", True),
}


def _check_torch(so_name: str) -> None:
    wanted = _TORCH_FOR_LIB.get(so_name)
    if wanted is None:
        return
    version, needs_rocm = wanted
    is_rocm = getattr(torch.version, "hip", None) is not None
    if is_rocm != needs_rocm:
        raise RuntimeError(
            f"{so_name} needs a {('ROCm' if needs_rocm else 'CUDA')} build of torch; this interpreter has torch {torch.__version__}"
        )
    if not torch.__version__.startswith(f"{version}."):
        raise RuntimeError(
            f"{so_name} was built against torch {version}; this interpreter has torch {torch.__version__}"
        )


_loaded_backend: str | None = None


def load_backend(model_type: str) -> None:
    global _loaded_backend
    so_name = _BACKENDS.get(model_type)
    if so_name is None:
        raise ValueError(f"Unknown model_type {model_type!r}. Supported: {sorted(_BACKENDS)}")
    if _loaded_backend is not None:
        if _loaded_backend != so_name:
            raise RuntimeError(
                f"TileRT backend '{_loaded_backend}' already loaded; cannot load '{so_name}' in the same process. Run {model_type} in a fresh process."
            )
        return
    _check_torch(so_name)
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
    logger.info("Loaded TileRT backend %s for model_type=%s", lib_path.name, model_type)


from .tilert_init import tilert_init  # noqa: E402

__all__ = ["logger", "load_backend", "tilert_init", "__version__"]
