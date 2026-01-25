"""DSA MTP E2E show hands for DeepSeek v3.2."""

from typing import Any

import torch

from tilert.models.deepseek_v3_2.dsa_show_hands import ShowHandsDSALayer

__all__ = [
    "ShowHandsDsaMtpE2eLayer",
    "dsa_mtp_e2e_show_hands_prepare_money",
    "dsa_mtp_e2e_show_hands",
    "dsa_mtp_e2e_show_hands_reset",
    "dsa_mtp_e2e_show_hands_go_home",
]


def dsa_mtp_e2e_show_hands_prepare_money(
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    cache_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
) -> Any:
    """Prepare money for MTP E2E show hands."""
    return torch.ops.tilert.dsa_mtp_e2e_show_hands_prepare_money(
        params, temp_vars, cache_vars, profile_logs
    )


def dsa_mtp_e2e_show_hands(draft_tokens: torch.Tensor) -> Any:
    """Show hands with MTP E2E."""
    return torch.ops.tilert.dsa_mtp_e2e_show_hands(draft_tokens)


def dsa_mtp_e2e_show_hands_reset(placeholder: torch.Tensor) -> Any:
    """Reset MTP E2E show hands."""
    return torch.ops.tilert.dsa_mtp_e2e_show_hands_reset(placeholder)


def dsa_mtp_e2e_show_hands_go_home(placeholder: torch.Tensor) -> Any:
    """Cleanup MTP E2E show hands."""
    return torch.ops.tilert.dsa_mtp_e2e_show_hands_go_home(placeholder)


def dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(
    placeholder: torch.Tensor, num_valid_tokens: int
) -> Any:
    """Set the number of valid (non-padding) tokens for prefill mode.

    This controls how many tokens are copied from draft_tokens to predicted_tokens
    during prefill. Should be called before forward() when the chunk has padding.

    Args:
        placeholder: Placeholder tensor for PyTorch dispatch (not used).
        num_valid_tokens: Number of valid tokens in the chunk (1-4).
    """
    return torch.ops.tilert.dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(
        placeholder, num_valid_tokens
    )


class ShowHandsDsaMtpE2eLayer(ShowHandsDSALayer):
    """Show hands DSA MTP E2E layer for DeepSeek v3.2.

    Inherits from ShowHandsDSALayer and adds MTP-specific functionality.
    """

    # MTP constants
    NUM_MTP = 3
    MTP_SEQ_LEN = NUM_MTP + 1  # 4

    def __init__(
        self,
        max_seq_len: int,
        with_weight_conversion: bool = True,
    ) -> None:
        super().__init__(
            max_seq_len=max_seq_len,
            model_path="",
            with_weight_conversion=with_weight_conversion,
            with_mtp=True,
        )

    def _get_num_cache_layers(self) -> int:
        """Return number of cache layers (+1 for shared MTP cache)."""
        return int(self.NUM_LAYERS) + 1

    def _prepare_money(
        self,
        params: list[torch.Tensor],
        intermediates: list[torch.Tensor],
        caches: list[torch.Tensor],
        profile_logs: torch.Tensor,
    ) -> None:
        """Prepare money for MTP E2E show hands."""
        dsa_mtp_e2e_show_hands_prepare_money(params, intermediates, caches, profile_logs)

    def _show_hands(self, draft_tokens: torch.Tensor) -> Any:
        """Run MTP E2E show hands forward."""
        return dsa_mtp_e2e_show_hands(draft_tokens.cpu())

    def _reset_sequence_impl(self) -> None:
        """Reset MTP E2E sequence."""
        dsa_mtp_e2e_show_hands_reset(self.placeholder)

    def _cleanup_impl(self) -> None:
        """Cleanup MTP E2E resources."""
        dsa_mtp_e2e_show_hands_go_home(self.placeholder)

    def set_prefill_valid_tokens(self, num_valid_tokens: int) -> None:
        """Set the number of valid tokens for prefill mode.

        This controls how many tokens are copied from draft_tokens to predicted_tokens
        during prefill. Should be called before forward() when the chunk has padding.

        Args:
            num_valid_tokens: Number of valid tokens in the chunk (1-4).
        """
        dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(self.placeholder, num_valid_tokens)

    def get_next_draft_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get next_draft_tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            next_draft_tokens tensor of shape [1, MTP_SEQ_LEN].
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        # next_draft_tokens is at index 38 (DsaTempVars::kNextDraftTokensIdx)
        return intermediates[38]

    def get_num_accepted(self, device_id: int = 0) -> int:
        """Get number of accepted tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            Number of accepted tokens.
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        # accepted_tokens (num_accepted) is at index 37 (DsaTempVars::kAcceptedTokensIdx)
        return int(intermediates[37][0].item())

    def get_predicted_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get predicted_tokens from the specified device.

        Args:
            device_id: Device ID to get results from.

        Returns:
            predicted_tokens tensor containing main model predictions.
        """
        intermediates, _, _, _ = self._get_device_result(device_id)
        # predicted_tokens is at index 35 (DsaTempVars::kPredictedTokensIdx)
        return intermediates[35]
