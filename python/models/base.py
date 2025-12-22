"""Base classes for deepseek v3."""

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from tilert import logger
from tilert.models.deepseek_config import get_rank, get_world_size
from tilert.models.preprocess import WeightLoader
from tilert.utils import get_profile_log_tensor

__all__ = [
    "TileRTModule",
]


class TileRTModule(nn.Module, ABC):
    """Base class for all TileRT modules.

    This class serves as an abstract base for implementing TileRT modules.
    All module classes should inherit from this class and implement their
    own forward method.
    """

    def __init__(
        self,
        op_name: str = "",
        golden_weights_dir: str = "",
        tilert_weights_dir: str = "",
        layer_idx: int = 0,
        compute_kernel_type: str = "bf16",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the operation.

        Args:
            op_name: Optional operation name. Defaults to class name.
            golden_weights_dir: Optional path to golden weights directory.
            tilert_weights_path: Optional path to tilert weights directory.
            layer_idx: Layer index.
            compute_kernel_type: Compute kernel type, bf16 by default; fp8 is also supported.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.layer_idx = layer_idx

        self.flag_enable_tilert = False

        if compute_kernel_type not in ["bf16", "fp8"]:
            raise ValueError(
                f"Invalid compute kernel type: {compute_kernel_type}, must be one of bf16, fp8."
            )
        self.compute_kernel_type = compute_kernel_type

        self.flag_enable_profiling_log = False
        self.flag_enable_external_profiling_log = False

        self.op_name = type(self).__name__ if op_name == "" else op_name
        self.profile_log_dir = "profile_logs"

        self.golden_weights_dir = golden_weights_dir
        self.tilert_weights_dir = tilert_weights_dir

        self.weight_loader = WeightLoader(
            layer_idx=layer_idx,
            golden_weights_dir=golden_weights_dir,
            tilert_weights_dir=tilert_weights_dir,
        )

        self.profile_logs = get_profile_log_tensor()

        # members for debugging
        self.temp_dir = os.path.join(os.path.expanduser("~"), ".cache", "tilert")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.tmp_vars: dict[str, torch.Tensor] = {}

    def register_weights(self, weights_config: dict[str, dict[str, Any]]) -> None:
        """Register weights configuration.

        Args:
            weights_config: Dictionary mapping weight names to their configurations.
        """
        self.weight_loader.register_weights(weights_config)

    def load_weights(self, device_id: int = 0) -> None:
        """Load weights from the weights path."""
        golden_weights_path = self.weight_loader.get_weight_file_path(
            device_id=device_id, is_tilert=False
        )
        self.weight_loader.load_weights(weights_path=golden_weights_path, device_id=device_id)

    def load_tilert_weights(self, device_id: int = 0) -> None:
        """Load tilert weights from the weights path."""
        tilert_weights_path = self.weight_loader.get_weight_file_path(
            device_id=device_id, is_tilert=True
        )
        self.weight_loader.load_tilert_weights(
            weights_path=tilert_weights_path, device_id=device_id
        )

    def get_weight(self, name: str, from_tilert: bool = False) -> Any:
        """Get a weight by name.

        Args:
            name: Weight name.
            from_tilert: Whether to get the weight from tilert.
        """
        return self.weight_loader.get_weight(name, from_tilert)

    def wrap_var_name(self, var_name: str) -> str:
        """Wrap the variable name.

        Args:
            var_name: Variable name.
        """
        return f"layer_{self.layer_idx}_{var_name}"

    def register_tmp_var(self, var_name: str, var_tensor: torch.Tensor) -> None:
        """Register a temporary variable for debugging.

        Args:
            var_name: Variable name.
            var_tensor: Variable.
        """
        self.tmp_vars[self.wrap_var_name(var_name)] = var_tensor

    def register_tmp_vars(self, var_dict: dict[str, torch.Tensor]) -> None:
        """Register a list of temporary variables for debugging.

        Args:
            var_dict: Dictionary of variable names and variables.
        """
        for var_name, tensor in var_dict.items():
            self.register_tmp_var(var_name, tensor)

    def dump_tmp_vars(
        self, tmp_vars: dict[str, torch.Tensor] | None = None, save_dir: str = ""
    ) -> None:
        """Dump variables to the profile log file.

        Args:
            tensor_vars: Dictionary of variable names and tensors.
            save_dir: Directory to save the variables.
        """
        if tmp_vars is None:
            tmp_vars = self.tmp_vars
        save_dir = self.temp_dir if save_dir == "" else save_dir
        os.makedirs(save_dir, exist_ok=True)

        for tensor_name in tmp_vars:
            logger.info(f"Saving variable {tensor_name} to {save_dir}")
            torch.save(tmp_vars[tensor_name], os.path.join(save_dir, f"{tensor_name}.pt"))

    def get_profile_log_path(self) -> str:
        """Get the path to the profile log file.

        Returns:
            Path to the profile log file.
        """
        return os.path.join(self.profile_log_dir, f"{self.op_name}.xlsx")

    def get_external_profile_log_path(self) -> str:
        """Get the path to the external profile log file.

        Returns:
            Path to the external profile log file.
        """
        return os.path.join(self.profile_log_dir, f"{self.op_name}.json")

    def world_size(self) -> int:
        """Get the world size.

        Returns:
            World size.
        """
        return int(get_world_size())

    def rank(self) -> int:
        """Get the rank.

        Returns:
            Rank.
        """
        return int(get_rank())

    @abstractmethod
    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Golden forward pass.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        del args, kwargs
        raise NotImplementedError("Golden forward not implemented")

    @abstractmethod
    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: U100
        """Tilert forward method to be implemented by subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del args, kwargs
        raise NotImplementedError("Tilert forward not implemented")

    @abstractmethod
    def to_tilert_weights(self, *args: Any, **kwargs: Any) -> None:
        """Convert weights to tilert.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del args, kwargs
        raise NotImplementedError("Convert weights to tilert not implemented")

    def enable_profiling_log(self, enable: bool = True) -> None:
        """Enable profiling log for this module and all submodules.

        Args:
            enable: Whether to enable profiling.
        """
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable profiling for {module.__class__.__name__}")
                module.flag_enable_profiling_log = enable

    def enable_external_profiling_log(self, enable: bool = True) -> None:
        """Enable external profiling log for this module and all submodules.

        Args:
            enable: Whether to enable external profiling.
        """
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable external profiling for {module.__class__.__name__}")
                module.flag_enable_external_profiling_log = enable

    def enable_tilert(self, enable: bool = True) -> None:  # type: ignore
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable tilert for {module.__class__.__name__}")
                module.flag_enable_tilert = enable
                if enable:
                    module.to_tilert_weights()
