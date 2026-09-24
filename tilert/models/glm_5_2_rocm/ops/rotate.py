"""GLM-5.2 rotate op wrappers: the indexer rotate pair (goldens + forwards)."""

import torch

DIM = 128
LANES = 16
PER_LANE = 8
ROPE_DIM = 64
SCALE = 0.08838834764831843
LN_EPS = 1e-06
INDEX_HEADS = 32
SUPPORTED_HEADS = (8, 16, 32)
SUPPORTED_SAMPLES = (1, 2, 4, 8)


def rope_interleaved(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    samples, n, _ = x.shape
    out = x.clone()
    rope = x[..., :ROPE_DIM].reshape(samples, n, ROPE_DIM // 2, 2)
    fr = freqs.reshape(samples, 1, ROPE_DIM // 2, 2)
    a, b = (rope[..., 0], rope[..., 1])
    c, d = (fr[..., 0], fr[..., 1])
    rot = torch.stack([a * c - b * d, a * d + b * c], dim=-1)
    out[..., :ROPE_DIM] = rot.reshape(samples, n, ROPE_DIM)
    return out


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    v = x.reshape(-1, LANES, PER_LANE).clone()
    for st in range(3):
        stride = 1 << st
        for j in range(4):
            lo = j & stride - 1
            idx = (j - lo) * 2 + lo
            a = v[:, :, idx].clone()
            b = v[:, :, idx + stride].clone()
            v[:, :, idx] = a + b
            v[:, :, idx + stride] = a - b
    lane = torch.arange(LANES)
    for st in range(4):
        mask = 1 << st
        partner = v.index_select(1, (lane ^ mask).to(v.device))
        sign = torch.where((lane & mask).bool(), torch.tensor(-1.0), torch.tensor(1.0)).to(v.device)
        v = sign[None, :, None] * v + partner
    return v.reshape(shape)


class RotateGlm5:
    """RoPE + Hadamard on the indexer queries."""

    def __init__(self, device: str = "cuda:0", num_heads: int = INDEX_HEADS):
        assert num_heads in SUPPORTED_HEADS
        self.device = device
        self.num_heads = num_heads

    def golden_forward_f32(self, iq: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        samples = iq.shape[0]
        x = iq.float().cpu().reshape(samples, self.num_heads, DIM)
        x = rope_interleaved(x, freqs.float().cpu())
        return hadamard128(x) * SCALE

    def golden_forward(self, iq: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")


class LayerNormRopeRotateGlm5:
    """LayerNorm + RoPE + Hadamard into the ki cache row at cur_pos."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def rows_golden_f32(self, ki: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        assert self.weight is not None and self.bias is not None
        samples = ki.shape[0]
        x = ki.float().cpu().reshape(samples, 1, DIM)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + LN_EPS)
        x = x * self.weight.float().cpu()[None, None, :] + self.bias.float().cpu()[None, None, :]
        x = rope_interleaved(x, freqs.float().cpu())
        return (hadamard128(x) * SCALE).reshape(samples, DIM)

    def rows_golden(self, ki: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.rows_golden_f32(ki, freqs).to(torch.bfloat16)

    def golden_forward(
        self,
        ki: torch.Tensor,
        ki_cache: torch.Tensor,
        freqs: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")
