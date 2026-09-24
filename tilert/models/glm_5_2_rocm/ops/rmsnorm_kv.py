"""GLM-5.2 RmsnormKv op wrapper: golden + forward for the kv-cache RMSNorm."""

import torch

KV_DIM = 512
EPS = 1e-05
SUPPORTED_SAMPLES = (1, 2, 4, 8)


class RmsnormKvGlm5:
    """Normalize kv rows into the bf16 kv cache at cur_pos."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.gamma: torch.Tensor | None = None

    def init_random_gamma(self, seed: int = 0) -> None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g = 1.0 + 0.1 * torch.randn(KV_DIM, generator=gen, dtype=torch.float32)
        self.init_reference_gamma(g)

    def init_reference_gamma(self, gamma: torch.Tensor) -> None:
        assert gamma.numel() == KV_DIM
        g = gamma.float().reshape(KV_DIM)
        self.gamma = g.to(self.device)

    def golden_forward(self, kv: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("golden_forward is not available in release builds")
