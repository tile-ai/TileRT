"""GLM-5.2 UnprojOAllReduce op wrapper: packer, golden, forward."""

import os

import torch

HIDDEN = 6144
V_HEAD_DIM = 256
SCALE_BLK = 128
NUM_PES = 8
FP8_MAX = 448.0
SUPPORTED_HEADS = (8, 10, 16)
PROTO_0, PROTO_1 = (0, 1)
BLOCKS = 256
ROWS_PER_BLOCK = 24
MAX_SEQ = 4


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % SCALE_BLK == 0 and k % SCALE_BLK == 0
    sr, sc = (rows // SCALE_BLK, k // SCALE_BLK)
    blocks = w.float().view(sr, SCALE_BLK, sc, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (q.reshape(rows, k).contiguous(), scales.view(sr, sc).contiguous())


def swizzle_v2(w_fp8: torch.Tensor) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows % 8 == 0 and k % (8 * 64) == 0
    units = k // 8 // 64
    pairs, odd = (units // 2, units % 2)
    aligned_k = 8 * pairs * 128
    w8 = w_fp8.view(torch.uint8)
    lane = torch.arange(64)
    row_in_group = (lane >> 5) * 4 + (lane & 3)
    kslice = (lane >> 2 & 7) * 4
    i = torch.arange(4)
    s4 = torch.arange(4)

    def seg(k_base):
        return k_base[:, None, :, None] + kslice[None, :, None, None] + i[None, None, None, :]

    pieces = []
    if pairs:
        kb = torch.arange(8 * pairs)[:, None] * 128 + s4[None, :] * 32
        pieces.append(seg(kb))
    if odd:
        j = torch.arange(4)
        wave = 2 * j[:, None] + (s4[None, :] >> 1)
        kb = aligned_k + wave * 64 + (s4[None, :] & 1) * 32
        pieces.append(seg(kb))
    kk = torch.cat(pieces, dim=0)
    kidx = kk.reshape(-1)
    ridx = row_in_group[None, :, None, None].expand_as(kk).reshape(-1)
    assert kidx.numel() == 8 * k
    rg = torch.arange(rows // 8)
    rows_ix = rg[:, None] * 8 + ridx[None, :]
    ks = kidx[None, :].expand(rows // 8, -1)
    return w8[rows_ix, ks].reshape(-1).contiguous()


def sym_bytes(samples: int) -> int:
    return int(torch.ops.tilert.glm5_unprojo_allreduce_sym_bytes(samples))


def sym_buffer(samples: int, device) -> torch.Tensor:
    return torch.zeros(sym_bytes(samples), dtype=torch.uint8, device=device)


def sym_table(buffers: list[torch.Tensor], device) -> torch.Tensor:
    assert len(buffers) == NUM_PES
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64, device=device)


def enable_peer_access(ndev: int = NUM_PES) -> int:
    return int(torch.ops.tilert.glm5_enable_peer_access(ndev))


class UnprojOAllReduceGlm5:
    """One rank's output projection: W_o shard + the ops that run on it."""

    def __init__(self, num_heads: int = 10, device: str = "cuda:0") -> None:
        assert num_heads in SUPPORTED_HEADS
        self.num_heads = num_heads
        self.k = num_heads * V_HEAD_DIM
        self.device = device
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: dict[int, torch.Tensor] = {}

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def partial_golden(self, proj_o: torch.Tensor) -> torch.Tensor:
        assert self.w_fp8 is not None and self.scales is not None
        x = proj_o.float().cpu()
        wq = self.w_fp8.float().cpu()
        scales = self.scales.cpu()
        acc = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32)
        for rb in range(HIDDEN // SCALE_BLK):
            r0, r1 = (rb * SCALE_BLK, (rb + 1) * SCALE_BLK)
            part = torch.zeros(x.shape[0], SCALE_BLK, dtype=torch.float32)
            for kb in range(self.k // SCALE_BLK):
                k0, k1 = (kb * SCALE_BLK, (kb + 1) * SCALE_BLK)
                part += x[:, k0:k1] @ wq[r0:r1, k0:k1].T * scales[rb, kb]
            acc[:, r0:r1] = part
        return acc.to(torch.bfloat16)

    @staticmethod
    def reduce_golden(partials: list[torch.Tensor], residual: torch.Tensor | None) -> torch.Tensor:
        acc = torch.zeros(partials[0].shape, dtype=torch.float32)
        for p in partials:
            acc += p.float().cpu()
        if residual is not None:
            acc += residual.float().cpu()
        return acc.to(torch.bfloat16)

    def golden_forward(self, proj_o: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(
        self,
        proj_o: torch.Tensor,
        residual: torch.Tensor | None = None,
        proto: int = PROTO_0,
        sym: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.packed is not None and self.scales is not None
        os.environ["TILERT_GLM5_AR_PROTO"] = str(proto)
        if out is None:
            out = torch.empty(proj_o.shape[0], HIDDEN, dtype=torch.bfloat16, device=proj_o.device)
        torch.ops.tilert.glm5_unprojo_allreduce_op(
            proj_o, self.packed, self.scales, residual, sym, mype, npes, flag, out
        )
        return out
