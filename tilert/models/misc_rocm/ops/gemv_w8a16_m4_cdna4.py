"""gemv_w8a16_m4_cdna4 op wrapper: 4x4-MFMA-tile packing + forward."""

import torch

from tilert.models.misc_rocm.ops.gemv_w8a16_mfma import (
    HIDDEN,
    ROWS,
    SCALE_BLK,
    SUPPORTED_SEQS,
    GemvW8A16Mfma,
)

ROWS_PER_BLOCK = 8
WAVES = 8
CHUNK = 128
_N_RG = ROWS // ROWS_PER_BLOCK
_N_KC = HIDDEN // CHUNK


def swizzle_weights_m4(w_fp8: torch.Tensor) -> torch.Tensor:
    assert w_fp8.shape == (ROWS, HIDDEN)
    w8 = w_fp8.view(torch.uint8)
    rg = torch.arange(_N_RG)
    kc = torch.arange(_N_KC)
    lane = torch.arange(64)
    s = torch.arange(4)
    i = torch.arange(4)
    RG, KC, L, S, II = torch.meshgrid(rg, kc, lane, s, i, indexing="ij")
    rows = RG * ROWS_PER_BLOCK + (L >> 5) * 4 + (L & 3)
    ks = KC * CHUNK + S * 32 + (L >> 2 & 7) * 4 + II
    return w8[rows, ks].reshape(-1).contiguous()


class GemvW8A16M4Cdna4(GemvW8A16Mfma):
    """4x4-tile CDNA4 op."""

    def __init__(self, device: str = "cuda:0"):
        super().__init__(device)
        self.packed_m4: torch.Tensor | None = None

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def tilert_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.packed_m4 is not None
        out = torch.empty(hidden.size(0), ROWS, dtype=torch.bfloat16, device=hidden.device)
        torch.ops.tilert.gemv_w8a16_m4_cdna4_op(hidden, self.packed_m4, self.scales, out)
        return out
