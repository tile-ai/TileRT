"""MTP preprocess layer for DeepSeek v3."""

from dataclasses import dataclass

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import init_func, linear
from tilert.models.deepseek_v3_2.model_args import ModelArgs

__all__ = [
    "mtp_preprocess_layer",
    "MTPPreprocessLayer",
    "MTPPreprocessRefWeightsAlias",
    "MTPPreprocessTilertWeightsAlias",
    "MTPPreprocessWeightsConverter",
]


def mtp_preprocess_layer(
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
) -> torch.Tensor:
    """MTP preprocess layer op for DeepSeek v3.

    Output is in temp_vars[28] (eh_proj) for DSA temp vars layout.
    """
    return torch.ops.tilert.mtp_preprocess_layer(params, temp_vars, profile_logs)


@dataclass
class MTPPreprocessRefWeightsAlias:
    """Reference (golden/PyTorch) weight keys for MTP preprocess."""

    embedding_rmsnorm = "enorm.weight"
    hidden_rmsnorm = "hnorm.weight"
    eh_proj = "eh_proj.weight"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            self.embedding_rmsnorm,
            self.hidden_rmsnorm,
            self.eh_proj,
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class MTPPreprocessTilertWeightsAlias:
    """TileRT weight keys for MTP preprocess."""

    embedding_rmsnorm_gamma = "embedding_rmsnorm_gamma"
    hidden_rmsnorm_gamma = "hidden_rmsnorm_gamma"
    eh_proj_weights = "eh_proj_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.embedding_rmsnorm_gamma,
            self.hidden_rmsnorm_gamma,
            self.eh_proj_weights,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class MTPPreprocessWeightsConverter(TilertWeightsConverter):
    """Converts ref-format weights to TileRT format for MTP preprocess."""

    def convert_to_tilert(self, weights: list[torch.Tensor], device_id: int) -> list[torch.Tensor]:
        """
        Convert ref weights to TileRT format for a specific device.

        Args:
            weights: [embedding_rmsnorm_gamma, hidden_rmsnorm_gamma, eh_proj.weight]
                     Ref format: enorm.weight [7168], hnorm.weight [7168],
                     eh_proj.weight [7168, 14336].
            device_id: Target device ID for weight placement.

        Returns:
            MTPPreprocessParams with converted weights for device_id.
        """
        device = torch.device(f"cuda:{device_id}")
        embedding_rmsnorm_gamma, hidden_rmsnorm_gamma, eh_proj_weight = weights

        embedding_rmsnorm_gamma = embedding_rmsnorm_gamma.to(device=device, dtype=torch.float32)
        hidden_rmsnorm_gamma = hidden_rmsnorm_gamma.to(device=device, dtype=torch.float32)
        # eh_proj: [out, in] = [7168, 14336]; split on dim=1 -> 8 x [7168, 1792]
        # Reshape: [7168, 1792] -> [128, 56, 7, 256] -> transpose(1,2) -> [128, 7, 56, 256]
        # eh_proj_weight_splited = torch.chunk(eh_proj_weight, self.num_devices, dim=1)
        eh_proj_weights = (
            eh_proj_weight.reshape(
                128, self.model_args.dim // 128, self.model_args.dim * 2 // 256 // 8, 256
            )
            .transpose(1, 2)
            .contiguous()
            .to(device=device, dtype=torch.bfloat16)
        )
        return [embedding_rmsnorm_gamma, hidden_rmsnorm_gamma, eh_proj_weights]


class MTPPreprocessLayer(TileRTModule):
    """MTP preprocess layer: RMSNorm(embedding), RMSNorm(hidden), concat & project."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        device_id: int,
        ref_weights_alias: MTPPreprocessRefWeightsAlias | None = None,
    ) -> None:
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )
        self.tilert_weights_alias = MTPPreprocessTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias if ref_weights_alias is not None else MTPPreprocessRefWeightsAlias()
        )
        self.hidden_size = model_args.dim

        self.tilert_embedding_rmsnorm_gamma: torch.Tensor | None = None
        self.tilert_hidden_rmsnorm_gamma: torch.Tensor | None = None
        self.tilert_eh_proj_weights: torch.Tensor | None = None

        self.ref_embedding_rmsnorm_gamma: torch.Tensor | None = None
        self.ref_hidden_rmsnorm_gamma: torch.Tensor | None = None
        self.ref_eh_proj_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return [
            self.tilert_embedding_rmsnorm_gamma,
            self.tilert_hidden_rmsnorm_gamma,
            self.tilert_eh_proj_weights,
        ]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Repeat ref weights for each device (for init_tilert_weights from ref)."""
        embedding_gamma = weights_map[self.ref_weights_alias.embedding_rmsnorm]
        hidden_gamma = weights_map[self.ref_weights_alias.hidden_rmsnorm]
        eh_proj_weights = weights_map[self.ref_weights_alias.eh_proj]
        return {
            self.tilert_weights_alias.embedding_rmsnorm_gamma: (
                embedding_gamma[None, ...].repeat(self.num_devices, 1)
            ),
            self.tilert_weights_alias.hidden_rmsnorm_gamma: (
                hidden_gamma[None, ...].repeat(self.num_devices, 1)
            ),
            self.tilert_weights_alias.eh_proj_weights: (
                eh_proj_weights[None, ...]
                .reshape(
                    self.model_args.dim,
                    self.num_devices,
                    self.model_args.dim * 2 // self.num_devices,
                )
                .transpose(0, 1)
            ),
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load ref-format weights (enorm.weight, hnorm.weight, eh_proj.weight)."""
        self.ref_embedding_rmsnorm_gamma = state_dict[self.ref_weights_alias.embedding_rmsnorm]
        self.ref_hidden_rmsnorm_gamma = state_dict[self.ref_weights_alias.hidden_rmsnorm]
        self.ref_eh_proj_weight = state_dict[self.ref_weights_alias.eh_proj]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load TileRT weights from state_dict.

        state_dict may use:
        - Full keys: layer_{layer_id}_{alias}_dev_{device_id}
        - Short keys: embedding_rmsnorm_gamma, hidden_rmsnorm_gamma, eh_proj_weights
        - Ref keys: enorm.weight, hnorm.weight, eh_proj.weight (then convert)
        """
        converter = MTPPreprocessWeightsConverter(self.model_args, self.num_devices)
        params = converter.convert_to_tilert(
            [state_dict[k] for k in self.tilert_weights_alias()], self.device_id
        )
        self.tilert_embedding_rmsnorm_gamma = params[0]
        self.tilert_hidden_rmsnorm_gamma = params[1]
        self.tilert_eh_proj_weights = params[2]

    def init_random_weights(self) -> dict[str, torch.Tensor]:
        """Initialize random ref weights and convert to TileRT for this device."""
        embedding_gamma = init_func(torch.randn(self.hidden_size, dtype=torch.float32))
        hidden_gamma = init_func(torch.randn(self.hidden_size, dtype=torch.float32))
        eh_proj_weights = init_func(
            torch.randn(self.hidden_size, self.hidden_size * 2, dtype=torch.bfloat16)
        )
        return {
            self.ref_weights_alias.embedding_rmsnorm: embedding_gamma,
            self.ref_weights_alias.hidden_rmsnorm: hidden_gamma,
            self.ref_weights_alias.eh_proj: eh_proj_weights,
        }

    def golden_forward(
        self,
        x: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reference forward: enorm(x), hnorm(last_hidden), concat & eh_proj.

        Args:
            x: [batch, seq_len, hidden_size] embedded tokens
            last_hidden_states: [batch, seq_len, hidden_size] previous hidden

        Returns:
            [batch, seq_len, hidden_size] projected hidden
        """
        assert self.ref_embedding_rmsnorm_gamma is not None
        assert self.ref_hidden_rmsnorm_gamma is not None
        assert self.ref_eh_proj_weight is not None

        future_norm = torch.nn.functional.rms_norm(
            x.float(),
            [x.size(-1)],
            self.ref_embedding_rmsnorm_gamma,
            1e-6,
        )
        prev_norm = torch.nn.functional.rms_norm(
            last_hidden_states.float(),
            [last_hidden_states.size(-1)],
            self.ref_hidden_rmsnorm_gamma,
            1e-6,
        )
        combined = torch.cat([future_norm, prev_norm], dim=-1)
        return linear(combined, self.ref_eh_proj_weight)

    def tilert_forward(
        self,
        params: list[torch.Tensor],
        temp_vars: list[torch.Tensor],
        profile_logs: torch.Tensor,
    ) -> torch.Tensor:
        """Run TileRT mtp_preprocess_layer op."""
        return mtp_preprocess_layer(params, temp_vars, profile_logs)
