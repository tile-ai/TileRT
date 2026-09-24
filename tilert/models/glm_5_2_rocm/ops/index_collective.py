"""GLM-5.2 indexer collective op wrappers: the GPU0 index broadcast and its all-reduce leg."""

import os

import torch

HIDDEN = 6144
NUM_PES = 8
PROTO_0, PROTO_1 = (0, 1)
BLOCKS = 256
ROWS_PER_BLOCK = 24
MAX_SEQ = 4


def xfer_buf_bytes(samples: int, topk: int) -> int:
    return int(torch.ops.tilert.glm5_broadcast_xfer_buf_bytes(samples, topk))


def xfer_buffer(samples: int, topk: int, device) -> torch.Tensor:
    return torch.zeros(xfer_buf_bytes(samples, topk), dtype=torch.uint8, device=device)


def peer_table(buffers: list[torch.Tensor], device) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64, device=device)


def broadcast(indices: torch.Tensor, sym: torch.Tensor, mype: int, npes: int, flag: int) -> None:
    torch.ops.tilert.glm5_broadcast_selected_token_ids_op(indices, sym, mype, npes, flag)


def broadcast_golden(indices: torch.Tensor, flag: int) -> torch.Tensor:
    pairs = indices.reshape(-1).to(torch.int32).cpu().reshape(-1, 2)
    out = torch.empty(pairs.shape[0], 4, dtype=torch.int32)
    out[:, 0] = pairs[:, 0]
    out[:, 1] = flag
    out[:, 2] = pairs[:, 1]
    out[:, 3] = flag
    return out.reshape(-1)


def padded_allreduce_add(
    out: torch.Tensor,
    residual: torch.Tensor | None = None,
    proto: int = PROTO_0,
    sym: torch.Tensor | None = None,
    mype: int = 0,
    npes: int = 1,
    flag: int = 1,
) -> torch.Tensor:
    os.environ["TILERT_GLM5_AR_PROTO"] = str(proto)
    torch.ops.tilert.glm5_padded_allreduce_add_op(residual, sym, mype, npes, flag, out)
    return out


def reduce_golden(partials: list[torch.Tensor], residual: torch.Tensor | None) -> torch.Tensor:
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for p in partials:
        acc += p.float().cpu()
    if residual is not None:
        acc += residual.float().cpu()
    return acc.to(torch.bfloat16)
