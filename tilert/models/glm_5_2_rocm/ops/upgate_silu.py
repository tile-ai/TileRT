"""GLM-5.2 FFN up/gate op wrappers: packers, goldens, forwards."""

import torch

HIDDEN = 6144
DENSE_INTER = 1536
MOE_INTER = 256
TOP_K = 8
MOE_SLOTS = TOP_K + 1
NUM_MOE_WEIGHTS = 257
SCALE_BLK = 128
FP8_MAX = 448.0
EPS = 1e-05
PAIR = 8
_CHUNK = 64
DENSE_SAMPLES = (1, 2, 4)
MOE_SAMPLES = (1, 2, 4, 8)


def pair_interleave_rows(inter: int) -> torch.Tensor:
    p = torch.arange(2 * inter)
    t, u, v = (p // 16, p % 16 // PAIR, p % PAIR)
    return u * inter + t * PAIR + v


def quantize_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = w.shape
    assert rows % SCALE_BLK == 0 and k % SCALE_BLK == 0
    sr, sc = (rows // SCALE_BLK, k // SCALE_BLK)
    blocks = w.float().view(sr, SCALE_BLK, sc, SCALE_BLK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scales = amax / FP8_MAX
    q = (blocks / scales).to(torch.float8_e4m3fn)
    return (q.reshape(rows, k).contiguous(), scales.view(sr, sc).contiguous())


def swizzle_pair_interleaved(w_fp8: torch.Tensor, inter: int) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows == 2 * inter and k % _CHUNK == 0
    w8 = w_fp8.view(torch.uint8)
    perm = pair_interleave_rows(inter)
    t = torch.arange(rows // 16)
    kc = torch.arange(k // _CHUNK)
    lane = torch.arange(64)
    sp = torch.arange(2)
    i = torch.arange(8)
    T, KC, L, SP, II = torch.meshgrid(t, kc, lane, sp, i, indexing="ij")
    rows_ix = perm[T * 16 + L % 16]
    ks = KC * _CHUNK + SP * 32 + L // 16 * 8 + II
    return w8[rows_ix, ks].reshape(-1).contiguous()


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _blocked_gemv(
    act: torch.Tensor, wq: torch.Tensor, scales: torch.Tensor, row0: int
) -> torch.Tensor:
    rows, k = wq.shape
    assert rows % SCALE_BLK == 0 and row0 % SCALE_BLK == 0
    sr0 = row0 // SCALE_BLK
    nrb = rows // SCALE_BLK
    row_scale = scales[sr0 : sr0 + nrb].repeat_interleave(SCALE_BLK, dim=0)
    out = torch.zeros(act.shape[0], rows, dtype=torch.float32)
    for kb in range(k // SCALE_BLK):
        k0, k1 = (kb * SCALE_BLK, (kb + 1) * SCALE_BLK)
        out += act[:, k0:k1] @ wq[:, k0:k1].T * row_scale[None, :, kb]
    return out


class RmsnormUpGateSiluGlm5:
    """The dense MLP's up/gate half (layers 0-2)."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.inter = DENSE_INTER
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None
        self.gamma_arg: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        assert self.packed is not None and self.scales is not None
        if out is None:
            out = torch.empty(
                hidden.shape[0], self.inter, dtype=torch.bfloat16, device=hidden.device
            )
        torch.ops.tilert.glm5_rmsnorm_upgate_silu_op(
            hidden, self.gamma_arg, self.packed, self.scales, out
        )
        return out


class MoeUpGateSiluGlm5:
    """MoE slots 2+3: 257 experts' up/gate halves (index 0 = shared)."""

    def __init__(self, device: str = "cuda:0", num_weights: int = NUM_MOE_WEIGHTS):
        self.device = device
        self.num_weights = num_weights
        self.inter = MOE_INTER
        self.w_fp8: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, norm_hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def golden_routed_forward(
        self, norm_hidden: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        return self.golden_forward(norm_hidden, indices)[:, 1:]


class SharedUpGateSiluGlm5:
    """MoE slot 3 standalone, S x 16-CTA shape: RmsNorm + the SHARED expert's up/gate + SiLU in one launch, reading unproj_o directly."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.moe = MoeUpGateSiluGlm5(device=device, num_weights=1)
        self.gamma: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def golden_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.gamma is not None
        x = hidden.float().cpu()
        gamma = self.gamma.float().cpu()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return (x * gamma[None, :] * rms).to(torch.bfloat16)

    def golden_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")
