"""gemv_w8a16_mfma op wrapper: quantizer, weight swizzler, golden, forward."""

import torch

ROWS = 2048
HIDDEN = 6144
SCALE_BLK = 128
FP8_MAX = 448.0
SUPPORTED_SEQS = (1, 2, 4)
_ROWS_PER_BLOCK = 32
_CHUNK = 64


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert w.shape == (ROWS, HIDDEN)
    blocks = w.float().view(ROWS // SCALE_BLK, SCALE_BLK, HIDDEN // SCALE_BLK, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (
        q.view(ROWS // SCALE_BLK, SCALE_BLK, HIDDEN // SCALE_BLK, SCALE_BLK)
        .permute(0, 1, 2, 3)
        .reshape(ROWS, HIDDEN)
        .contiguous(),
        scales.view(ROWS // SCALE_BLK, HIDDEN // SCALE_BLK).contiguous(),
    )


def swizzle_weights(w_fp8: torch.Tensor) -> torch.Tensor:
    assert w_fp8.shape == (ROWS, HIDDEN)
    w8 = w_fp8.view(torch.uint8)
    b = torch.arange(ROWS // _ROWS_PER_BLOCK)
    g = torch.arange(2)
    h = torch.arange(4)
    c = torch.arange(HIDDEN // 4 // _CHUNK)
    lane = torch.arange(64)
    s = torch.arange(4)
    i = torch.arange(4)
    B, G, H, C, L, S, II = torch.meshgrid(b, g, h, c, lane, s, i, indexing="ij")
    rows = B * _ROWS_PER_BLOCK + G * 16 + L % 16
    ks = H * (HIDDEN // 4) + C * _CHUNK + S * 16 + L // 16 * 4 + II
    return w8[rows, ks].reshape(-1).contiguous()


class GemvW8A16Mfma:
    """Op class: golden and tilert forwards share weights."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def dequant(self) -> torch.Tensor:
        assert self.w_fp8 is not None and self.scales is not None
        blocks = self.w_fp8.float().view(
            ROWS // SCALE_BLK, SCALE_BLK, HIDDEN // SCALE_BLK, SCALE_BLK
        )
        return (blocks * self.scales[:, None, :, None]).view(ROWS, HIDDEN)

    def golden_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.packed is not None and self.scales is not None
        out = torch.empty(hidden.size(0), ROWS, dtype=torch.bfloat16, device=hidden.device)
        torch.ops.tilert.gemv_w8a16_mfma_op(hidden, self.packed, self.scales, out)
        return out
