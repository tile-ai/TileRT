"""GLM-5.2 RmsnormHeadProj op wrapper: golden + forward."""

import torch

HIDDEN = 6144
TOP1_WS_WORDS = 16
VOCAB_SHARD = 19360
TILES = VOCAB_SHARD // 16
EPS = 1e-05
SUPPORTED_SAMPLES = (1, 2, 4, 8)


def swizzle_head_bf16(w: torch.Tensor) -> torch.Tensor:
    assert w.shape == (VOCAB_SHARD, HIDDEN)
    w16 = w.contiguous().to(torch.bfloat16).view(torch.uint16)
    w16 = w16.reshape(TILES, 16, HIDDEN // 32, 4, 8)
    packed = w16.permute(0, 2, 3, 1, 4).contiguous()
    return packed.reshape(-1).contiguous().view(torch.uint8)


class RmsnormHeadProjGlm5:
    """Final rmsnorm + bf16 head projection to f32 logits."""

    OP_NAME = "glm5_rmsnorm_head_proj_op"

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.gamma: torch.Tensor | None = None
        self.head: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, gamma: torch.Tensor, head: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.gamma is not None
        x = hidden.float()
        ssq = (x * x).sum(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(ssq / HIDDEN + EPS)
        return (x * self.gamma[None, :] * rms_inv).to(torch.bfloat16)

    def golden_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(
        self, hidden: torch.Tensor, with_norm_out: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.gamma is not None and self.packed is not None
        samples = hidden.shape[0]
        logits = torch.empty(samples, VOCAB_SHARD, dtype=torch.float32, device=self.device)
        norm_out = (
            torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=self.device)
            if with_norm_out
            else None
        )
        torch.ops.tilert.glm5_rmsnorm_head_proj_op(
            hidden, self.gamma, self.packed, logits, norm_out
        )
        return (logits, norm_out)
