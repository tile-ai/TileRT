"""GLM-5.2 RmsnormProjXWqkva op wrapper: quantizer, weight packer, golden, forward."""

import torch

HIDDEN = 6144
Q_DIM = 2048
KV_DIM = 512
PE_DIM = 64
ROWS = Q_DIM + KV_DIM + PE_DIM
SCALE_BLK = 128
SCALE_ROWS = -(-ROWS // SCALE_BLK)
SCALE_COLS = HIDDEN // SCALE_BLK
FP8_MAX = 448.0
EPS = 1e-05
SUPPORTED_SAMPLES = (1, 2, 4)
_CHUNK = 64
_N_RG = ROWS // 16
_N_KC = HIDDEN // _CHUNK


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert w.shape == (ROWS, HIDDEN)
    wf = w.float()
    padded = torch.zeros(SCALE_ROWS * SCALE_BLK, HIDDEN, dtype=torch.float32)
    padded[:ROWS] = wf
    blocks = padded.view(SCALE_ROWS, SCALE_BLK, SCALE_COLS, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    q_rows = q.reshape(SCALE_ROWS * SCALE_BLK, HIDDEN)[:ROWS].contiguous()
    return (q_rows, scales.view(SCALE_ROWS, SCALE_COLS).contiguous())


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


_N_T8 = ROWS // 8
_N_KC128 = HIDDEN // 128


class RmsnormProjXWqkvaGlm5:
    """Op class: golden and tilert forwards share weights."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None
        self.gamma_arg: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(
        self, hidden: torch.Tensor, cur_pos: torch.Tensor, seq_len: int, pe_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
