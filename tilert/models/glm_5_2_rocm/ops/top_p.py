"""GLM-5.2 top-p op wrapper: golden + forward for distributed sampling."""

import numpy as np
import torch

VOCAB_SHARD = 19360
NUM_PES = 8
TOP_K = 256
SEND_BYTES = 2080
_M64 = (1 << 64) - 1
_SEED_MUL = 19349663
_POS_MUL = 73856093
_STEP_MUL = 8589934591
_COL_MUL = 479001599
_COIN_MUL = 11400714819323198485
_COIN_XOR = 13787848793156543929
NO_DRAFT = -1


def send_bytes() -> int:
    return int(torch.ops.tilert.glm5_top_p_send_bytes())


def sym_bytes(samples: int) -> int:
    return int(torch.ops.tilert.glm5_top_p_sym_bytes(samples))


def sym_buffer(samples: int, device) -> torch.Tensor:
    return torch.zeros(sym_bytes(samples), dtype=torch.uint8, device=device)


def sym_table(buffers: list[torch.Tensor], device) -> torch.Tensor:
    ptrs = [int(b.data_ptr()) for b in buffers]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def gumbel_uniform(seed: int, position: int, col: int) -> np.float32:
    step_seed = seed * _SEED_MUL & _M64 ^ position * _POS_MUL & _M64
    hashed = step_seed * _STEP_MUL & _M64 ^ col * _COL_MUL & _M64
    u = np.float32(hashed % (1 << 24)) * np.float32(1.0 / (1 << 24))
    return np.float32(min(max(u, np.float32(1e-10)), np.float32(1.0 - 1e-10)))


def verify_coin(seed: int, position: int) -> np.float32:
    step_seed = seed * _SEED_MUL & _M64 ^ position * _POS_MUL & _M64
    hashed = step_seed * _COIN_MUL & _M64 ^ _COIN_XOR
    u = np.float32(hashed % (1 << 24)) * np.float32(1.0 / (1 << 24))
    return np.float32(min(max(u, np.float32(1e-10)), np.float32(1.0 - 1e-10)))


def _inv_t(temperature: float) -> np.float32:
    if temperature < 1e-06:
        return np.float32(1e30)
    if temperature != 1.0:
        return np.float32(1.0 / temperature)
    return np.float32(1.0)


def local_scores_golden(
    logits: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    x = logits.float().cpu().numpy().astype(np.float32) * _inv_t(temperature)
    m = np.float32(x.max())
    e = np.exp(x - m, dtype=np.float32)
    order = np.lexsort((np.arange(e.shape[0]), -e))[:TOP_K]
    return (
        torch.from_numpy(e[order].copy()),
        torch.from_numpy(order.astype(np.int32)),
        float(m),
        float(e.sum(dtype=np.float32)),
    )


def sample_golden(
    per_rank: list[tuple[torch.Tensor, torch.Tensor, float, float]],
    top_p: float,
    seed: int,
    position: int,
    draft: int = NO_DRAFT,
) -> tuple[int, float]:
    npes = len(per_rank)
    gm = np.float32(max((np.float32(m) for _, _, m, _ in per_rank)))
    scales = [np.float32(np.exp(np.float32(m) - gm)) for _, _, m, _ in per_rank]
    gl = np.float32(0.0)
    for (_, _, _, l), s in zip(per_rank, scales):
        gl = np.float32(gl + s * np.float32(l))
    vals = np.concatenate(
        [v.numpy().astype(np.float32) * s for (v, _, _, _), s in zip(per_rank, scales)]
    )
    idx = np.concatenate(
        [i.numpy().astype(np.int64) + p * VOCAB_SHARD for p, (_, i, _, _) in enumerate(per_rank)]
    )
    order = np.lexsort((idx, -vals))
    vals, idx = (vals[order], idx[order])
    probs = (vals / gl).astype(np.float32)
    cum = np.cumsum(probs, dtype=np.float32)
    over = np.nonzero(cum > np.float32(top_p))[0]
    cutoff = int(over[0]) if over.size else npes * TOP_K - 1
    accept_col = -1
    if draft != NO_DRAFT:
        hit = np.nonzero(idx[: cutoff + 1] == draft)[0]
        if hit.size:
            dc = int(hit[0])
            p_draft = np.float32(probs[dc] / max(cum[cutoff], np.float32(1e-20)))
            if verify_coin(seed, position) < p_draft:
                accept_col = dc
            else:
                probs = probs.copy()
                probs[dc] = np.float32(0.0)
    if accept_col >= 0:
        return (int(idx[accept_col]), float(probs[accept_col]))
    best, best_col = (-np.float32(np.finfo(np.float32).max), 0)
    for col in range(cutoff + 1):
        u = gumbel_uniform(seed, position, col)
        g = np.float32(-np.log(-np.log(u)))
        pert = np.float32(np.log(probs[col] + np.float32(1e-10)) + g)
        if pert > best:
            best, best_col = (pert, col)
    return (int(idx[best_col]), float(probs[best_col]))


class TopPGlm5:
    """Distributed nucleus sampling (single-GPU npes=1 arm included)."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device

    def tilert_forward(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        sym: torch.Tensor | None = None,
        draft_tokens: torch.Tensor | None = None,
        verify_seq: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = logits.shape[0]
        dev = logits.device
        send_buf = torch.empty(samples * SEND_BYTES, dtype=torch.uint8, device=dev)
        token = torch.empty(samples, dtype=torch.int32, device=dev)
        prob = torch.empty(samples, dtype=torch.float32, device=dev)
        torch.ops.tilert.glm5_top_p_local_scores_op(
            logits, temperature, send_buf, sym, mype, npes, flag
        )
        torch.ops.tilert.glm5_top_p_sample_op(
            send_buf,
            top_p,
            seeds,
            positions,
            sym,
            mype,
            npes,
            flag,
            token,
            prob,
            draft_tokens,
            verify_seq,
        )
        return (token, prob)
