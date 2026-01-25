"""Weight loading and preprocessing utilities."""

import os
from typing import Any

import torch

from tilert import logger
from tilert.models.deepseek_config import get_world_size

__all__ = [
    "print_weights_info",
    "WeightLoader",
    "DownAllreduceWeightsConverter",
    "RMSNormProjQAKVAKIWeightsConverter",
    "RMSNormHeadProjWeightsConverter",
    "ExpertSelectUpGateSiLUWeightsConverter",
    "RMSNormUpGateSiLUWeightsConverter",
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
        self, data_shape: torch.Size, config_shape: tuple[int, ...], split_method: str = "no_split"
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


class RMSNormProjQAKVAKIWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def tilert_to_common(
        tilert_wqkv_a: torch.Tensor,
        tilert_wqkv_a_scales: torch.Tensor,
        tilert_attn_norm_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Convert tilert weights to common weights.

        Args:
            tilert_wqkv_a: Tilert weight tensor.
            tilert_wqkv_a_scales: Tilert weight scale tensor.
            tilert_attn_norm_weight: Tilert attention norm weight tensor.
        Returns:
            tuple: Common weights.
        """
        wq_a = tilert_wqkv_a[:1536]  # 1536, 7168
        wkv_a = tilert_wqkv_a[1536 : 1536 + 576]  # 576, 7168
        wk = tilert_wqkv_a[1536 + 576 :]  # 128, 7168

        wqkv_a_scales_0 = tilert_wqkv_a_scales[:128, :].reshape(16, 8, 64)
        wqkv_a_scales_0 = wqkv_a_scales_0[:, 0, :].reshape(16, 64)
        wqkv_a_scales_1 = tilert_wqkv_a_scales[128:129, :]  # 1, 64
        wqkv_a_scales_2 = tilert_wqkv_a_scales[129:, :]  # 1, 64
        wqkv_a_scales_swizzled = torch.cat(
            [wqkv_a_scales_0, wqkv_a_scales_1, wqkv_a_scales_2], dim=0
        )
        wqkv_scales = torch.zeros(
            (18, 56), dtype=torch.bfloat16, device=tilert_wqkv_a_scales.device
        )

        for i in range(64):
            if ((i % 8) * 8 + i // 8) < 56:
                wqkv_scales[:, ((i % 8) * 8 + i // 8)] = wqkv_a_scales_swizzled[:, i]
        wq_a_scale = wqkv_scales[:12, :]  # 12, 56
        wkv_a_scale = wqkv_scales[12:17, :]  # 5, 56
        wk_scale = wqkv_scales[17:, :]  # 1, 56

        attn_norm_weight = tilert_attn_norm_weight
        return wq_a, wq_a_scale, wkv_a, wkv_a_scale, wk, wk_scale, attn_norm_weight

    @staticmethod
    def common_to_tilert_native_bf16_warp_gemv(
        wq_a: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_a: torch.Tensor,
        wkv_a_scale: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        attn_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert common weights to weights for tilert native bf16 warp gemv op.

        Args:
            wq_a: Common weight tensor.
            wq_a_scale: Common weight scale tensor.
            wkv_a: Common weight tensor.
            wkv_a_scale: Common weight scale tensor.
            wk: Common weight tensor.
            wk_scale: Common weight scale tensor.
            attn_norm_weight: Common attention norm weight tensor.
        Returns:
            tuple: Tilert weights for native bf16 warp gemv op.
        """
        wq_a_scale = wq_a_scale.reshape((12, 56, 1)).repeat(1, 1, 128).reshape((12, 1, 7168))
        wq_a_scale = wq_a_scale.repeat(1, 128, 1).reshape((1536, 7168))
        wkv_a_scale = wkv_a_scale.reshape((5, 56, 1)).repeat(1, 1, 128).reshape((5, 1, 7168))
        wkv_a_scale = wkv_a_scale.repeat(1, 128, 1).reshape((-1, 7168))
        wkv_a_scale = wkv_a_scale[:576]
        wk_scale = wk_scale.reshape((1, 56, 1)).repeat(1, 1, 128).reshape((1, 1, 7168))
        wk_scale = wk_scale.repeat(1, 128, 1).reshape((128, 7168))
        wq_a = wq_a.reshape((1536, 7168)).float() * wq_a_scale.float()
        wkv_a = wkv_a.reshape((576, 7168)).float() * wkv_a_scale.float()
        wk = wk.reshape((128, 7168)).float() * wk_scale.float()
        # concatenate the weights
        weights = torch.cat([wq_a, wkv_a, wk], dim=0)
        assert weights.shape == (1536 + 576 + 128, 7168)

        weights = weights.reshape(140, 16, 7, 1024)
        weights = weights.transpose(1, 2)  # 140, 7, 16, 1024
        return weights.to(torch.bfloat16).contiguous(), attn_norm_weight.clone()


class ExpertSelectUpGateSiLUWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def tilert_to_tilert_144sm(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str | None = None
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
            mma_type: MMA type, None,"16x32" or "16x16"
        Returns:
            tilert_144sm weights and scales
        """
        exp_num = mat_in.shape[0]
        assert mat_in.shape == (exp_num, 512, 7168)
        assert mat_scale_in.shape == (exp_num, 4, 64)
        weights_trt = mat_in.reshape(exp_num, 128, 4, 7168)
        weights_w1 = weights_trt[:, :, :2].reshape(exp_num, 256, 7168)
        weights_w3 = weights_trt[:, :, 2:].reshape(exp_num, 256, 7168)
        # to 16x1024 blocks
        weights_w1 = weights_w1.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        weights_w3 = weights_w3.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        if mma_type == "16x32":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)
        elif mma_type == "16x16":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)

        weights = torch.cat([weights_w1, weights_w3], dim=3)
        assert weights.shape == (exp_num, 16, 7, 32, 1024)
        weights = weights.reshape(exp_num, 16, 7, 32 * 1024)

        # For scales, first unswizzle
        scales_unswizzled = torch.zeros(exp_num, 4, 56)
        for i in range(64):
            if ((i % 8) * 8 + i // 8) < 56:
                scales_unswizzled[..., ((i % 8) * 8 + i // 8)] = mat_scale_in[..., i]
        scales_unswizzled = scales_unswizzled.reshape(exp_num, 2, 2, 56)

        scales_w1 = scales_unswizzled[:, :, :1].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w1 = scales_w1.transpose(2, 3)
        scales_w3 = scales_unswizzled[:, :, 1:].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w3 = scales_w3.transpose(2, 3)
        scales = torch.cat([scales_w1, scales_w3], dim=3)
        assert scales.shape == (exp_num, 16, 7, 2, 8)
        scales = (
            scales.reshape(exp_num, 16, 7, 2 * 8).to(torch.bfloat16).view(dtype=torch.float8_e4m3fn)
        )
        weights_and_scales = torch.zeros(
            exp_num, 16, 7, 32 * 1024 + 128, dtype=torch.float8_e4m3fn, device=mat_in.device
        )
        weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
        weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + 32].copy_(scales)
        return weights_and_scales

    @staticmethod
    def tilert_to_tilert_144sm_mma(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str = "16x32"
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm_mma input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
        Returns:
            tilert_144sm weights and scales
        """
        return ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm(
            mat_in, mat_scale_in, mma_type
        )


class RMSNormHeadProjWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def tilert_to_tilert_native_bf16_warp_gemv(
        tilert_weight_in: torch.Tensor,
    ) -> torch.Tensor:
        """Convert TILERT weights to TILERT native bf16 warp gemv weights."""
        weights = tilert_weight_in.reshape(1010, 16, 7, 1024)
        weights = weights.transpose(1, 2).reshape(7070, 16, 1024)
        return weights.contiguous()


class RMSNormUpGateSiLUWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def tilert_to_tilert_144sm(
        mat_in: torch.Tensor,
        mat_scale_in: torch.Tensor,
        mma_type: str | None = None,
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
        Returns:
            tilert_144sm weights and scales
        """
        return ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm(
            mat_in, mat_scale_in, mma_type
        )

    @staticmethod
    def tilert_to_tilert_144sm_mma(
        mat_in: torch.Tensor,
        mat_scale_in: torch.Tensor,
        mma_type: str = "16x32",
    ) -> torch.Tensor:
        """Convert tilert weights and scales to tilert_144sm_mma input format."""
        return ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm_mma(
            mat_in, mat_scale_in, mma_type
        )


class UnProjOAllreduceWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def tilert_to_tilert_112sm_mma(
        mat_in: torch.Tensor,
        mat_scale_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert tilert weights to tilert_112sm_mma input format.

        Args:
            mat_in: tilert weights, [7168, 2048]
            mat_scale_in: tilert scales, [896, 16]
        Returns:
            tilert_112sm weights, scales
        """
        swizzle_for_mma_16x32 = True
        assert mat_in.shape == (7168, 2048)
        assert mat_scale_in.shape == (896, 16)
        weights_trt = mat_in.reshape(112, 64, 2048)
        # to 64*512 blocks
        weights_trt = weights_trt.reshape(112, 64, 4, 512).transpose(1, 2)  # (112, 4, 64, 512)
        if swizzle_for_mma_16x32:
            # to (112, 4, 4(n), 16(k), 16, 32)
            weights_trt = weights_trt.reshape(112, 4, 4, 16, 16, 32).transpose(-2, -3)
            weights_trt = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_trt)
            weights_trt = weights_trt.reshape(112, 4, 4 * 16 * 16 * 32)

        # For scales
        scales_trt = mat_scale_in.reshape(56, 16, 16)[:, 0, :]
        scales_trt = scales_trt.reshape(56, 16)
        return weights_trt.contiguous(), scales_trt.contiguous()


class DownAllreduceWeightsConverter:
    """Weights converter class."""

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        # PTX isa fig.88
        pre_shape = mat_in.shape[:-2]
        return mat_in.reshape(*pre_shape, 8, 2, 4, 4).transpose(-2, -3).contiguous()

    @staticmethod
    def tilert_to_tilert_mma(
        mat_in: torch.Tensor, scale_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tilert weights and scales to tilert_mma input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
        Returns:
            tilert_mma weights and scales
        """
        exp_num = mat_in.shape[0]
        mat_in_s = mat_in.reshape(exp_num, 128, 56, 256)
        # mat_in_s[:, :, 48:56] = 0;
        mat_in_0 = mat_in_s[:, :, :16].reshape(exp_num, 128, 16, 8, 32).transpose(2, 3)
        mat_in_0 = DownAllreduceWeightsConverter._swizzle_mma_16x32(mat_in_0).reshape(
            exp_num, 128, -1
        )
        mat_in_1 = mat_in_s[:, :, 16:32].reshape(exp_num, 128, 16, 8, 32).transpose(2, 3)
        mat_in_1 = DownAllreduceWeightsConverter._swizzle_mma_16x32(mat_in_1).reshape(
            exp_num, 128, -1
        )
        mat_in_2 = mat_in_s[:, :, 32:48].reshape(exp_num, 128, 16, 8, 32).transpose(2, 3)
        mat_in_2 = DownAllreduceWeightsConverter._swizzle_mma_16x32(mat_in_2).reshape(
            exp_num, 128, -1
        )
        mat_in_3 = mat_in_s[:, :, 48:56].reshape(exp_num, 128, 8, 8, 32).transpose(2, 3)
        mat_in_3 = DownAllreduceWeightsConverter._swizzle_mma_8x32(mat_in_3).reshape(
            exp_num, 128, -1
        )

        mat_in_swizzled = torch.cat([mat_in_0, mat_in_1, mat_in_2, mat_in_3], dim=2)
        return mat_in_swizzled.reshape(exp_num, 7168, 256).contiguous(), scale_in
