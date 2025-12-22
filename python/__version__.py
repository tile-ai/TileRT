from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

try:
    __version__ = pkg_version("tilert")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(
            root=str(Path(__file__).resolve().parents[1]),
            relative_to=__file__,
        )
    except Exception:
        __version__ = "0.0.0"

__all__ = ["__version__"]
