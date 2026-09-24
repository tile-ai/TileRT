"""GLM-5.2 LlmPreprocess op wrapper: golden + forward."""

import torch

HIDDEN = 6144
ROPE_DIM = 64
SUPPORTED_SAMPLES = (1, 2, 4, 8)


def make_freqs_cis(max_pos: int, theta: float = 8000000.0, device: str = "cuda:0") -> torch.Tensor:
    inv = 1.0 / theta ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    ang = torch.outer(torch.arange(max_pos), inv)
    cis = torch.polar(torch.ones_like(ang), ang)
    return torch.view_as_real(cis).reshape(max_pos, ROPE_DIM).contiguous().to(device)


class LlmPreprocessGlm5:
    """Embedding + rope-freq row gather at cur_pos."""

    OP_NAME = "glm5_llm_preprocess_op"

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.embed: torch.Tensor | None = None
        self.freqs_cis: torch.Tensor | None = None

    def init_random_weights(self, vocab: int, max_pos: int, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def golden_forward(
        self, token_id: torch.Tensor, cur_pos: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(
        self, token_id: torch.Tensor, cur_pos: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.embed is not None and self.freqs_cis is not None
        samples = token_id.numel()
        x = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=self.device)
        freqs = torch.empty(samples, ROPE_DIM, dtype=torch.float32, device=self.device)
        torch.ops.tilert.glm5_llm_preprocess_op(
            token_id, self.embed, self.freqs_cis, cur_pos, x, freqs, seq_len
        )
        return (x, freqs)
