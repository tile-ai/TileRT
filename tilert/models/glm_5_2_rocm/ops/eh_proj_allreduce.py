"""GLM-5.2 MtpPreProcess op wrapper: packers, golden, forward."""

import os

import torch

HIDDEN = 6144
K_IN = 1536
NUM_PES = 8
EPS = 1e-05
PROTO_0, PROTO_1 = (0, 1)
MAX_SEQ = 8


def swizzle_128_bf16(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 16 == 0 and k % 32 == 0
    w16 = w.to(torch.bfloat16).view(torch.uint16)
    rg = torch.arange(rows // 16)
    kc = torch.arange(k // 32)
    lane = torch.arange(64)
    i = torch.arange(8)
    RG, KC, L, II = torch.meshgrid(rg, kc, lane, i, indexing="ij")
    packed = w16[RG * 16 + L % 16, KC * 32 + L // 16 * 8 + II]
    return packed.reshape(-1).contiguous().view(torch.uint8)


def swizzle_256_bf16(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 8 == 0 and k % 64 == 0
    w16 = w.to(torch.bfloat16).view(torch.uint16)
    rg = torch.arange(rows // 8)
    kc = torch.arange(k // 64)
    lane = torch.arange(64)
    s = torch.arange(2)
    i = torch.arange(4)
    RG, KC, L, S, II = torch.meshgrid(rg, kc, lane, s, i, indexing="ij")
    rows_ix = RG * 8 + (L >> 5) * 4 + (L & 3)
    ks = KC * 64 + S * 32 + (L >> 2 & 7) * 4 + II
    packed = w16[rows_ix, ks]
    return packed.reshape(-1).contiguous().view(torch.uint8)


def rms_inv_ref(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return torch.rsqrt(xf.square().mean(dim=-1) + EPS)


class EhProjAllReduceGlm5:
    """The full W_eh [6144, 12288] plus per-rank packed slices."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.w: torch.Tensor | None = None
        self.e_gamma: torch.Tensor | None = None
        self.h_gamma: torch.Tensor | None = None
        self.packed: dict[int, torch.Tensor] = {}
        self._dev_gammas: tuple[torch.Tensor, torch.Tensor] | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(
        self, w: torch.Tensor, e_gamma: torch.Tensor, h_gamma: torch.Tensor
    ) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def _rank_slice_bf16(self, mype: int) -> torch.Tensor:
        assert self.w is not None
        ws = self.w[:, mype * K_IN : (mype + 1) * K_IN]
        return ws.to(torch.bfloat16)

    def packed_weights(self, mype: int) -> torch.Tensor:
        if mype not in self.packed:
            ws = self._rank_slice_bf16(mype)
            self.packed[mype] = swizzle_128_bf16(ws).to(self.device)
        return self.packed[mype]

    def gammas(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dev_gammas is None:
            self._dev_gammas = (self.e_gamma.to(self.device), self.h_gamma.to(self.device))
        return self._dev_gammas

    def partial_golden(
        self, embed_x: torch.Tensor, last_hidden: torch.Tensor, mype: int
    ) -> torch.Tensor:
        assert self.w is not None
        is_h = mype >= NUM_PES // 2
        src = (last_hidden if is_h else embed_x).float().cpu()
        gamma = self.h_gamma if is_h else self.e_gamma
        g0 = mype % (NUM_PES // 2) * K_IN
        x_slice = src[:, g0 : g0 + K_IN]
        act = (x_slice * gamma[g0 : g0 + K_IN][None, :]).to(torch.bfloat16).float()
        wf = self._rank_slice_bf16(mype).float().cpu()
        rinv = rms_inv_ref(src.to(torch.bfloat16))
        return (act @ wf.T * rinv[:, None]).to(torch.bfloat16)

    @staticmethod
    def reduce_golden(partials: list[torch.Tensor]) -> torch.Tensor:
        acc = torch.zeros(partials[0].shape, dtype=torch.float32)
        for p in partials:
            acc += p.float().cpu()
        return acc.to(torch.bfloat16)

    def golden_forward(
        self, embed_x: torch.Tensor, last_hidden: torch.Tensor, mype: int
    ) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(
        self,
        embed_x: torch.Tensor,
        last_hidden: torch.Tensor,
        proto: int = PROTO_0,
        sym: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        os.environ["TILERT_GLM5_AR_PROTO"] = str(proto)
        e_gamma, h_gamma = self.gammas()
        if out is None:
            out = torch.empty(embed_x.shape[0], HIDDEN, dtype=torch.bfloat16, device=embed_x.device)
        torch.ops.tilert.glm5_eh_proj_allreduce_op(
            embed_x,
            last_hidden,
            e_gamma,
            h_gamma,
            self.packed_weights(mype),
            sym,
            mype,
            npes,
            flag,
            out,
        )
        return out
