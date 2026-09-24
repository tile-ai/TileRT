"""GLM-5.2 QkvRope op wrapper: golden + forward for the MLA pe-path RoPE."""

import torch

ROPE_DIM = 64
NUM_HEADS = 10
SUPPORTED_HEADS = (8, 10)
SUPPORTED_SAMPLES = (1, 2, 4, 8)


def make_rope_freqs(
    samples: int, device: str = "cuda:0", seed: int = 0, positions: list[int] | None = None
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if positions is None:
        ang = torch.rand(samples, ROPE_DIM // 2, generator=gen, dtype=torch.float32)
        ang = ang * (2 * torch.pi)
    else:
        from tilert.models.glm_5_2_rocm.ops.llm_preprocess import make_freqs_cis

        assert len(positions) == samples
        table = make_freqs_cis(max(positions) + 1, device="cpu")
        return table[torch.tensor(positions)].contiguous().to(device)
    freqs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    return freqs.reshape(samples, ROPE_DIM).contiguous().to(device)


def rotate_golden(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    samples = x.shape[0]
    xf = x.float().reshape(samples, -1, ROPE_DIM // 2, 2)
    fr = freqs.float().reshape(samples, 1, ROPE_DIM // 2, 2)
    a, b = (xf[..., 0], xf[..., 1])
    c, d = (fr[..., 0], fr[..., 1])
    out = torch.stack([a * c - b * d, a * d + b * c], dim=-1)
    return out.reshape(samples, -1).to(torch.bfloat16)


class QkvRopeGlm5:
    """Rotate q_pe in place and the pe_cache row at cur_pos in place."""

    def __init__(self, device: str = "cuda:0", num_heads: int = NUM_HEADS):
        assert num_heads in SUPPORTED_HEADS
        self.device = device
        self.num_heads = num_heads
        self.pe_dim = num_heads * ROPE_DIM

    def golden_forward(
        self,
        q_pe: torch.Tensor,
        pe_cache: torch.Tensor,
        freqs: torch.Tensor,
        cur_pos: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
