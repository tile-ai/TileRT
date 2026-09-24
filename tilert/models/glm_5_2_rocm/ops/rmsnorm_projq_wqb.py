"""GLM-5.2 RmsnormProjQWqb op wrapper: quantizer, weight packer, golden, forward."""

import torch

Q_LORA_RANK = 2048
NOPE_PER_HEAD = 192
PE_PER_HEAD = 64
SCALE_BLK_M = 64
SCALE_BLK_K = 128
SCALE_COLS = Q_LORA_RANK // SCALE_BLK_K
FP8_MAX = 448.0
EPS = 1e-05
SUPPORTED_SAMPLES = (1, 2, 4)
SUPPORTED_HEADS = (8, 10)
NUM_HEADS = 10
NOPE_DIM = NUM_HEADS * NOPE_PER_HEAD
PE_DIM = NUM_HEADS * PE_PER_HEAD
ROWS = NOPE_DIM + PE_DIM
SCALE_ROWS = ROWS // SCALE_BLK_M
_CHUNK = 64
_N_KC = Q_LORA_RANK // _CHUNK


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = w.shape[0]
    assert rows % SCALE_BLK_M == 0 and w.shape[1] == Q_LORA_RANK
    scale_rows = rows // SCALE_BLK_M
    wf = w.float()
    blocks = wf.view(scale_rows, SCALE_BLK_M, SCALE_COLS, SCALE_BLK_K)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (
        q.reshape(rows, Q_LORA_RANK).contiguous(),
        scales.view(scale_rows, SCALE_COLS).contiguous(),
    )


def swizzle_weights_contig(w_fp8: torch.Tensor) -> torch.Tensor:
    rows = w_fp8.shape[0]
    assert rows % 16 == 0 and w_fp8.shape[1] == Q_LORA_RANK
    w8 = w_fp8.view(torch.uint8)
    rg = torch.arange(rows // 16)
    kc = torch.arange(_N_KC)
    lane = torch.arange(64)
    sp = torch.arange(2)
    i = torch.arange(8)
    RG, KC, L, SP, II = torch.meshgrid(rg, kc, lane, sp, i, indexing="ij")
    rows_ix = RG * 16 + L % 16
    ks = KC * _CHUNK + SP * 32 + L // 16 * 8 + II
    return w8[rows_ix, ks].reshape(-1).contiguous()


class RmsnormProjQWqbGlm5:
    """Op class: golden and tilert forwards share weights."""

    def __init__(self, device: str = "cuda:0", num_heads: int = NUM_HEADS):
        self.device = device
        self.num_heads = num_heads
        self.nope_dim = num_heads * NOPE_PER_HEAD
        self.pe_dim = num_heads * PE_PER_HEAD
        self.rows = self.nope_dim + self.pe_dim
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None
        self.gamma_arg: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, q_down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
