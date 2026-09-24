"""GLM-5.2 MoE down-projection bank, golden and symmetric-buffer helpers."""

import torch

HIDDEN = 6144
EXPERT_DIM = 256
TOP_K = 8
SLOTS = TOP_K + 1
SCALE_BLK = 128
NUM_PES = 8
FP8_MAX = 448.0
PROTO_0, PROTO_1 = (0, 1)


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % SCALE_BLK == 0 and k % SCALE_BLK == 0
    sr, sc = (rows // SCALE_BLK, k // SCALE_BLK)
    blocks = w.float().view(sr, SCALE_BLK, sc, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (q.reshape(rows, k).contiguous(), scales.view(sr, sc).contiguous())


def swizzle_m4(w_fp8: torch.Tensor) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows % 8 == 0 and k % 128 == 0
    w8 = w_fp8.view(torch.uint8)
    rg = torch.arange(rows // 8)
    kc = torch.arange(k // 128)
    lane = torch.arange(64)
    s = torch.arange(4)
    i = torch.arange(4)
    RG, KC, L, S, II = torch.meshgrid(rg, kc, lane, s, i, indexing="ij")
    rows_ix = RG * 8 + (L >> 5) * 4 + (L & 3)
    ks = KC * 128 + S * 32 + (L >> 2 & 7) * 4 + II
    return w8[rows_ix, ks].reshape(-1).contiguous()


def sym_bytes(samples: int) -> int:
    return int(torch.ops.tilert.glm5_moe_sym_bytes(samples))


def sym_buffer(samples: int, device) -> torch.Tensor:
    return torch.zeros(sym_bytes(samples), dtype=torch.uint8, device=device)


def sym_table(buffers: list[torch.Tensor], device) -> torch.Tensor:
    assert len(buffers) == NUM_PES
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64, device=device)


class MoeDownBankGlm5:
    """One rank's expert down-projection bank (fp8 + block scales) and the down golden. Packing is the monokernel wrapper's job (pack_down_v4)."""

    def __init__(self, device: str = "cuda:0", num_weights: int = 257) -> None:
        self.device = device
        self.num_weights = num_weights
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def partial_golden(
        self, hidden_mid: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        assert self.w_fp8 is not None and self.scales is not None
        mid = hidden_mid.float().cpu()
        wq = self.w_fp8.float().cpu()
        scales = self.scales.cpu()
        pr = probs.float().cpu()
        idx = indices.cpu()
        s_n = mid.shape[0]
        acc = torch.zeros(s_n, HIDDEN, dtype=torch.float32)
        row_scale = scales.repeat_interleave(SCALE_BLK, dim=1)
        for s in range(s_n):
            for slot in range(SLOTS):
                e = 0 if slot == 0 else 1 + int(idx[s, slot - 1])
                w = 1.0 if slot == 0 else float(pr[s, slot - 1])
                a = mid[s, slot]
                for kb in range(EXPERT_DIM // SCALE_BLK):
                    k0, k1 = (kb * SCALE_BLK, (kb + 1) * SCALE_BLK)
                    part = wq[e, :, k0:k1] @ a[k0:k1]
                    acc[s] += part * (row_scale[e, :, kb] * w)
        return acc.to(torch.bfloat16)

    @staticmethod
    def reduce_golden(partials: list[torch.Tensor], residual: torch.Tensor | None) -> torch.Tensor:
        acc = torch.zeros(partials[0].shape, dtype=torch.float32)
        for p in partials:
            acc += p.float().cpu()
        if residual is not None:
            acc += residual.float().cpu()
        return acc.to(torch.bfloat16)

    def golden_forward(
        self,
        hidden_mid: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")
