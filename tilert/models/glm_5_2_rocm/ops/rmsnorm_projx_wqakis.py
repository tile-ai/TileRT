"""GLM-5.2 RmsnormProjXWqakis op wrapper: quantizer, both weight packers, golden, forward."""

import torch

HIDDEN = 6144
Q_DIM = 2048
KI_DIM = 128
IS_DIM = 32
ROWS = Q_DIM + KI_DIM
SCALE_BLK = 128
SCALE_ROWS = ROWS // SCALE_BLK
SCALE_COLS = HIDDEN // SCALE_BLK
FP8_MAX = 448.0
EPS = 1e-05
SUPPORTED_SAMPLES = (1, 2, 4)
_CHUNK = 64
_WIS_CHUNK = 64
_WIS_ROWS = 8


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % SCALE_BLK == 0 and k == HIDDEN
    blocks = w.float().view(rows // SCALE_BLK, SCALE_BLK, SCALE_COLS, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (
        q.reshape(rows, k).contiguous(),
        scales.view(rows // SCALE_BLK, SCALE_COLS).contiguous(),
    )


def swizzle_weights_contig(w_fp8: torch.Tensor) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows % 16 == 0 and k % _CHUNK == 0
    w8 = w_fp8.view(torch.uint8)
    rg = torch.arange(rows // 16)
    kc = torch.arange(k // _CHUNK)
    lane = torch.arange(64)
    sp = torch.arange(2)
    i = torch.arange(8)
    RG, KC, L, SP, II = torch.meshgrid(rg, kc, lane, sp, i, indexing="ij")
    return w8[RG * 16 + L % 16, KC * _CHUNK + SP * 32 + L // 16 * 8 + II].reshape(-1).contiguous()


def swizzle_wis_bf16(w_bf16: torch.Tensor) -> torch.Tensor:
    rows, k = w_bf16.shape
    assert rows % _WIS_ROWS == 0 and k % _WIS_CHUNK == 0
    w16 = w_bf16.to(torch.bfloat16).view(torch.int16)
    rg = torch.arange(rows // _WIS_ROWS)
    kc = torch.arange(k // _WIS_CHUNK)
    lane = torch.arange(64)
    i = torch.arange(8)
    RG, KC, L, II = torch.meshgrid(rg, kc, lane, i, indexing="ij")
    packed = w16[
        RG * _WIS_ROWS + L % _WIS_ROWS, KC * _WIS_CHUNK + L // 8 % 2 * 32 + L // 16 * 8 + II
    ]
    return packed.reshape(-1).contiguous().view(torch.uint8)


class RmsnormProjXWqakisGlm5:
    """Op class: golden and tilert forwards share the same weights."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None
        self.wis_bf16: torch.Tensor | None = None
        self.wis_packed: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None
        self.gamma_arg: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(
        self, w: torch.Tensor, wis: torch.Tensor, gamma: torch.Tensor
    ) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
