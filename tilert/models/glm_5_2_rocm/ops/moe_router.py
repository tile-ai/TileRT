"""GLM-5.2 router weights container + routing golden."""

import torch

HIDDEN = 6144
NUM_EXPERTS = 256
TOP_K = 8
ROUTE_SCALE = 2.5
EPS = 1e-05
_CHUNK_K = 32


def swizzle_router_bf16(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 16 == 0 and k % _CHUNK_K == 0
    w16 = w.to(torch.bfloat16).view(torch.uint16)
    tile = torch.arange(rows // 16)
    kc = torch.arange(k // _CHUNK_K)
    lane = torch.arange(64)
    i = torch.arange(8)
    T, KC, L, II = torch.meshgrid(tile, kc, lane, i, indexing="ij")
    packed = w16[T * 16 + L % 16, KC * _CHUNK_K + L // 16 * 8 + II]
    return packed.reshape(-1).contiguous().view(torch.uint8)


def select_topk_golden(
    partials: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = partials.float().cpu().sum(dim=0)
    b = bias.float().cpu()
    scores = torch.sigmoid(logits)
    ranked = scores + b[None, :]
    s_n = scores.shape[0]
    probs = torch.zeros(s_n, TOP_K, dtype=torch.float32)
    idx = torch.zeros(s_n, TOP_K, dtype=torch.int32)
    for s in range(s_n):
        r = ranked[s].clone()
        vals = []
        for k in range(TOP_K):
            e = int(torch.argmax(r))
            idx[s, k] = e
            vals.append(float(scores[s, e]))
            r[e] = -float("inf")
        total = 0.0
        for v in vals:
            total += v
        for k, v in enumerate(vals):
            probs[s, k] = v * (ROUTE_SCALE / total)
    return (probs, idx)


class MoeRouterGlm5:
    """One rank's logical router tensor + gamma (weights container/golden)."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.w: torch.Tensor | None = None
        self.packed: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_reference_weights(self, w: torch.Tensor, gamma: torch.Tensor) -> None:
        raise RuntimeError("init_reference_weights is not available in release builds")

    def golden_forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
