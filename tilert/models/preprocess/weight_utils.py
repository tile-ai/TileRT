"""Weight loading and preprocessing utilities."""

import os
from typing import Any

import torch

from tilert import logger
from tilert.models.deepseek_config import get_world_size

__all__ = [
    "print_weights_info",
    "WeightLoader",
]


def print_weights_info(weights_path: str) -> None:
    """Print the information of the weights."""
    try:
        weights = torch.load(
            weights_path,
            map_location="cuda",
            weights_only=True,
        )
        print("Successfully loaded weights. Available keys:")
        for key in weights.keys():
            print(f"  - {key}, shape: {weights[key].shape}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise


class WeightLoader:
    """Weight loader for TileRT models."""

    def __init__(
        self,
        layer_idx: int = 0,
        golden_weights_dir: str = "",
        tilert_weights_dir: str = "",
    ) -> None:
        """Initialize the weight loader.

        Args:
            layer_idx: Layer index.
            golden_weights_dir: Path to golden weights directory.
            tilert_weights_dir: Path to tilert weights directory.
        """
        self.layer_idx = layer_idx
        self.golden_weights_dir = golden_weights_dir
        self.tilert_weights_dir = tilert_weights_dir

        self.weights_loaded_golden = False
        self.weights_dict_golden: dict[str, dict[str, Any]] = {}

        self.weights_loaded_tilert = False
        self.weights_dict_tilert: dict[str, dict[str, Any]] = {}

    def get_weight_file_path(self, device_id: int = 0, is_tilert: bool = False) -> str:
        """Get the weight file path for a given layer.

        Args:
            device_id: Device id.
            is_tilert: Whether the weights are for tilert.
        """
        if is_tilert:
            return os.path.join(
                self.tilert_weights_dir,
                f"tilert_deepseek_v32.layer_{self.layer_idx}.dev_{device_id}.weights.pt",
            )

        return os.path.join(
            self.golden_weights_dir,
            f"deepseek_v32.layer_{self.layer_idx}.weights.pt",
        )

    def get_weight_prefix(self) -> str:
        """Get the weight file prefix for a given layer."""
        return f"model.layers.{self.layer_idx}."

    def register_weights(
        self, weights_config: dict[str, dict[str, Any]], is_tilert: bool = False
    ) -> None:
        """Register weights configuration.

        Args:
            weights_config: Dictionary mapping weight names to their configurations.
                Each configuration should have 'shape', 'dtype', and 'data' keys.
            is_tilert: Whether the weights are for tilert.
        """
        if is_tilert:
            self.weights_dict_tilert.update(weights_config)
        else:
            self.weights_dict_golden.update(weights_config)

    def check_shape(
        self,
        data_shape: torch.Size,
        config_shape: tuple[int, ...],
        split_method: str = "no_split",
    ) -> None:
        """Check if the shape of the data is the same as the shape in the weights configuration.

        Args:
            data_shape: Shape of the data tensor.
            config_shape: Expected shape from the configuration.

        Raises:
            ValueError: If the shapes don't match.
        """
        data_shape = tuple(data_shape)
        config_shape = tuple(config_shape)

        if split_method == "row_split":
            new_shape = (data_shape[0], data_shape[1] // get_world_size())
        elif split_method == "column_split":
            new_shape = (data_shape[0] // get_world_size(), data_shape[1])
        elif split_method == "no_split":
            new_shape = data_shape
        else:
            raise ValueError(f"Invalid split method: {split_method}")

        if new_shape != config_shape:
            raise ValueError(f"Shape mismatch: got {new_shape}, expected {config_shape}")

    def load_weights(self, weights_path: str, device_id: int = 0) -> None:
        """Load weights from the weights path.

        Args:
            weights_path: Path to weights file.
            device_id: Device id.
        """
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights path {weights_path} does not exist")

        # TODO(ying): Enhance the error handling for weights loading.
        device = torch.device(f"cuda:{device_id}")
        weights = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )

        for key in self.weights_dict_golden:
            weight_name = self.get_weight_prefix() + key
            if weight_name not in weights:
                raise ValueError(f"Weight {weight_name} not found in weights file")

            data = weights[weight_name]
            logger.info(f"Loaded weight {weight_name} with shape {data.shape}")

            item = self.weights_dict_golden[key]
            split_method = item.get("split_method", "no_split")
            self.check_shape(data.shape, item["shape"], split_method)

            if split_method == "row_split":
                split_size = data.shape[1] // get_world_size()
                start_idx = device_id * split_size
                end_idx = start_idx + split_size
                data = data[:, start_idx:end_idx]
            elif split_method == "column_split":
                split_size = data.shape[0] // get_world_size()
                start_idx = device_id * split_size
                end_idx = start_idx + split_size
                data = data[start_idx:end_idx, :]
            elif split_method == "no_split":
                pass
            else:
                raise ValueError(f"Invalid split method: {split_method}")

            if isinstance(item["data"], torch.Tensor):
                item["data"].copy_(data)
            else:
                item["data"] = data

        self.weights_loaded_golden = True

    def load_tilert_weights(self, weights_path: str, device_id: int = 0) -> None:
        """Load tilert weights from the weights path.

        Args:
            weights_path: Path to weights file.
            device_id: Device id.
        """
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights path {weights_path} does not exist")

        device = torch.device(f"cuda:{device_id}")
        weights = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )

        for key in self.weights_dict_tilert:
            if key not in weights:
                raise ValueError(f"Weight {key} not found in weights file")

            data = weights[key]
            logger.info(f"Loaded weight {key} with shape {data.shape}")
            if isinstance(self.weights_dict_tilert[key]["data"], torch.Tensor):
                self.check_shape(tuple(data.shape), self.weights_dict_tilert[key]["shape"])
                self.weights_dict_tilert[key]["data"].copy_(data)
            else:
                self.weights_dict_tilert[key]["data"] = data

        self.weights_loaded_tilert = True

    def get_weight(self, name: str, from_tilert: bool = False) -> Any:
        """Get a weight by name.

        Args:
            name: Weight name.

        Returns:
            Weight data.

        Raises:
            ValueError: If weight is not found or not loaded.
        """
        weight_dict = self.weights_dict_tilert if from_tilert else self.weights_dict_golden

        if name not in weight_dict:
            raise ValueError(f"Weight {name} not registered")

        if from_tilert:
            if not self.weights_loaded_tilert:
                raise ValueError("Tilert weights not loaded. Call load_tilert_weights first.")
        elif not self.weights_loaded_golden:
            raise ValueError("Golden weights not loaded. Call load_weights first.")

        return weight_dict[name]["data"]
