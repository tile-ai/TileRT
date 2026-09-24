"""GLM-5.2 projection op wrappers: quantizer, weight packer, golden, forward."""

import torch

KV_LORA_RANK = 512
QK_NOPE_DIM = 192
V_HEAD_DIM = 256
M4_SCALE_BLK_K = 64
M6_SCALE_BLK_K = 128
SCALE_BLK_M = 64
FP8_MAX = 448.0
SUPPORTED_SAMPLES = (1, 2, 4, 8)
SUPPORTED_HEADS = (8, 10)
NUM_HEADS = 10
_CHUNK = 64


def quantize_fp8_block(w: torch.Tensor, scale_blk_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % SCALE_BLK_M == 0 and k % scale_blk_k == 0
    scale_rows, scale_cols = (rows // SCALE_BLK_M, k // scale_blk_k)
    blocks = w.float().view(scale_rows, SCALE_BLK_M, scale_cols, scale_blk_k)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (q.reshape(rows, k).contiguous(), scales.view(scale_rows, scale_cols).contiguous())


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
    rows_ix = RG * 16 + L % 16
    ks = KC * _CHUNK + SP * 32 + L // 16 * 8 + II
    return w8[rows_ix, ks].reshape(-1).contiguous()


class _ProjWkvbGlm5:
    """Shared op class: golden + tilert forwards share one set of weights."""

    K: int
    OUT_DIM: int
    SCALE_BLK_K: int
    OP_NAME: str

    def __init__(self, device: str = "cuda:0", num_heads: int = NUM_HEADS):
        assert num_heads in SUPPORTED_HEADS
        self.device = device
        self.num_heads = num_heads
        self.act_dim = num_heads * self.K
        self.rows = num_heads * self.OUT_DIM
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, act: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")


class ProjQWkvbGlm5(_ProjWkvbGlm5):
    """Project q_nope_down [S, H*192] to q_nope [S, H*512]."""

    K = QK_NOPE_DIM
    OUT_DIM = KV_LORA_RANK
    SCALE_BLK_K = M4_SCALE_BLK_K


class ProjOWkvbGlm5(_ProjWkvbGlm5):
    """Project o [S, H*512] to proj_o [S, H*256] (the flat [S, 2560] the tail reads)."""

    K = KV_LORA_RANK
    OUT_DIM = V_HEAD_DIM
    SCALE_BLK_K = M6_SCALE_BLK_K
