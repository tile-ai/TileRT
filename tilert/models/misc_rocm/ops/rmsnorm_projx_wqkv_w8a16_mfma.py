"""rmsnorm_projx_wqkv_w8a16_mfma op wrapper: quantizer, swizzler, golden."""

import torch

HIDDEN = 2048
Q_DIM = 4096
KV_DIM = 1024
ROWS = Q_DIM + KV_DIM
SCALE_BLK = 128
FP8_MAX = 448.0
EPS = 1e-06
SUPPORTED_SEQS = (1, 2, 4)
_ROWS_PER_BLOCK = 64
_GROUPS = _ROWS_PER_BLOCK // 16
_K_PARTS = 2
_CHUNK = 64
_NUM_BLOCKS = ROWS // _ROWS_PER_BLOCK
_CHUNKS = HIDDEN // _K_PARTS // _CHUNK


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert w.shape == (ROWS, HIDDEN)
    blocks = w.float().view(ROWS // SCALE_BLK, SCALE_BLK, HIDDEN // SCALE_BLK, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (
        q.reshape(ROWS, HIDDEN).contiguous(),
        scales.view(ROWS // SCALE_BLK, HIDDEN // SCALE_BLK).contiguous(),
    )


def swizzle_weights(w_fp8: torch.Tensor) -> torch.Tensor:
    assert w_fp8.shape == (ROWS, HIDDEN)
    w8 = w_fp8.view(torch.uint8)
    v = w8.view(_NUM_BLOCKS, _GROUPS, 16, _K_PARTS, _CHUNKS, 4, 4, 4)
    return v.permute(0, 1, 3, 4, 6, 2, 5, 7).reshape(-1).contiguous()


class RmsNormProjXWqkvW8A16Mfma:
    """Op class: golden and tilert forwards share weights."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.gamma: torch.Tensor | None = None
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def dequant(self) -> torch.Tensor:
        assert self.w_fp8 is not None and self.scales is not None
        blocks = self.w_fp8.float().view(
            ROWS // SCALE_BLK, SCALE_BLK, HIDDEN // SCALE_BLK, SCALE_BLK
        )
        return (blocks * self.scales[:, None, :, None]).view(ROWS, HIDDEN)

    def golden_forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.packed is not None and self.scales is not None
        assert self.gamma is not None
        q_out = torch.empty(hidden.size(0), Q_DIM, dtype=torch.bfloat16, device=hidden.device)
        kv_out = torch.empty(hidden.size(0), KV_DIM, dtype=torch.bfloat16, device=hidden.device)
        torch.ops.tilert.rmsnorm_projx_wqkv_w8a16_mfma_op(
            hidden, self.gamma, self.packed, self.scales, q_out, kv_out
        )
        return (q_out, kv_out)
