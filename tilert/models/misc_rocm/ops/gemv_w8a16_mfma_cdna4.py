"""gemv_w8a16_mfma_cdna4 op wrapper: the CDNA4 kernel's two weight packings."""

import torch

from tilert.models.misc_rocm.ops.gemv_w8a16_mfma import (
    FP8_MAX,
    HIDDEN,
    ROWS,
    SCALE_BLK,
    SUPPORTED_SEQS,
    GemvW8A16Mfma,
    quantize_fp8_block,
    swizzle_weights,
)

_CHUNK = 64
_N_RG = ROWS // 16
_N_KC = HIDDEN // _CHUNK


def swizzle_weights_contig(w_fp8: torch.Tensor) -> torch.Tensor:
    assert w_fp8.shape == (ROWS, HIDDEN)
    w8 = w_fp8.view(torch.uint8)
    rg = torch.arange(_N_RG)
    kc = torch.arange(_N_KC)
    lane = torch.arange(64)
    sp = torch.arange(2)
    i = torch.arange(8)
    RG, KC, L, SP, II = torch.meshgrid(rg, kc, lane, sp, i, indexing="ij")
    rows = RG * 16 + L % 16
    ks = KC * _CHUNK + SP * 32 + L // 16 * 8 + II
    return w8[rows, ks].reshape(-1).contiguous()


class GemvW8A16MfmaCdna4(GemvW8A16Mfma):
    """CDNA4 op. Inherits the CDNA2 quantizer/golden so the two kernels are compared on identical numbers; only the packing and the launch differ."""

    def __init__(self, device: str = "cuda:0", contig: bool = True):
        super().__init__(device)
        self.contig = contig
        self.packed_contig: torch.Tensor | None = None

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def packed_for_kernel(self) -> torch.Tensor:
        packed = self.packed_contig if self.contig else self.packed
        assert packed is not None
        return packed

    def tilert_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = torch.empty(hidden.size(0), ROWS, dtype=torch.bfloat16, device=hidden.device)
        torch.ops.tilert.gemv_w8a16_mfma_cdna4_op(
            hidden, self.packed_for_kernel(), self.scales, out
        )
        return out
