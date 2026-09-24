"""GLM-5.2 FlashSparseMLA: shapes, the index wire helpers and the torch golden shared by the fused-tail wrappers and their tests."""

import math

import torch

KV_LORA_RANK = 512
PE_DIM = 64
QK_DIM = KV_LORA_RANK + PE_DIM
TILE_N = 64
TOPK_DEFAULT = 2048
SUPPORTED_HEADS = (8, 10, 16)
SUPPORTED_SAMPLES = (1, 2, 4, 8)
XCDS = 8
GLM5_SOFTMAX_SCALE = (192 + 64) ** (-0.5)


def split_tile_n() -> int:
    return 32


def softmax_scale(
    qk_nope_head_dim: int = 192, qk_rope_head_dim: int = 64, rope_factor: float | None = None
) -> float:
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** (-0.5)
    mscale = 1.0 if rope_factor is None else 0.1 * math.log(rope_factor) + 1.0
    return scale * mscale * mscale


def pack_xfer(indices: torch.Tensor, flag: int) -> torch.Tensor:
    flat = indices.reshape(-1).to(torch.int32)
    assert flat.numel() % 2 == 0
    out = torch.empty(flat.numel() * 2, dtype=torch.int32, device=flat.device)
    out[0::4] = flat[0::2]
    out[2::4] = flat[1::2]
    fw = torch.tensor(flag, dtype=torch.int64).to(torch.int32)
    out[1::4] = fw
    out[3::4] = fw
    return out


def xfer_send(indices: torch.Tensor, xfer_buf: torch.Tensor, flag: int) -> None:
    torch.ops.tilert.glm5_xfer_send_indices_op(
        indices.reshape(-1).contiguous().to(torch.int32), xfer_buf, flag
    )


def xfer_buffer(samples: int, topk: int, device: str | torch.device) -> torch.Tensor:
    return torch.zeros(samples * topk * 2, dtype=torch.int32, device=device)


def golden_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    indices: torch.Tensor | None,
    cur_pos: int,
    topk: int = TOPK_DEFAULT,
    scale: float = GLM5_SOFTMAX_SCALE,
) -> torch.Tensor:
    seq, heads, _ = q_nope.shape
    seqlen_kv = kv_cache.shape[0]
    kvf = kv_cache.float()
    pef = pe_cache.float()
    dev = q_nope.device
    out = torch.zeros(seq, heads, KV_LORA_RANK, dtype=torch.float32, device=dev)
    for s in range(seq):
        kv_len = cur_pos + 1 + s
        scores = (q_nope[s].float() @ kvf.T + q_pe[s].float() @ pef.T) * scale
        mask = torch.full((seqlen_kv,), float("-inf"), device=dev)
        if kv_len > topk:
            assert indices is not None, "sparse step needs a selection"
            mask[indices[s].long()] = 0.0
        else:
            mask[:kv_len] = 0.0
        probs = (scores + mask).softmax(dim=-1, dtype=torch.float32)
        out[s] = probs.to(torch.bfloat16).float() @ kvf
    return out.reshape(seq, heads * KV_LORA_RANK).to(torch.bfloat16)
