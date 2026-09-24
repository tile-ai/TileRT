"""GLM-5.2 Top1Allreduce op wrapper: golden + forward."""

import torch

VOCAB_SHARD = 19360
NUM_PES = 8


def sym_bytes(samples: int) -> int:
    return int(torch.ops.tilert.glm5_top1_allreduce_sym_bytes(samples))


def sym_buffer(samples: int, device) -> torch.Tensor:
    return torch.zeros(sym_bytes(samples), dtype=torch.uint8, device=device)


def sym_table(buffers: list[torch.Tensor], device) -> torch.Tensor:
    ptrs = [int(b.data_ptr()) for b in buffers]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def argmax_lowest_idx(logits: torch.Tensor) -> torch.Tensor:
    vals = logits.max(dim=-1, keepdim=True).values
    return (logits == vals).int().argmax(dim=-1).to(torch.int32)


class Top1AllreduceGlm5:
    """Greedy token selection over the TP8 vocab shards."""

    OP_NAME = "glm5_top1_allreduce_op"

    def __init__(self, device: str = "cuda:0"):
        self.device = device

    def local_golden(self, logits: torch.Tensor, mype: int) -> torch.Tensor:
        return argmax_lowest_idx(logits) + mype * VOCAB_SHARD

    @staticmethod
    def merged_golden(logits_by_rank: list[torch.Tensor]) -> torch.Tensor:
        return argmax_lowest_idx(torch.cat(logits_by_rank, dim=-1))

    def tilert_forward(
        self,
        logits: torch.Tensor,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        sym: torch.Tensor | None = None,
    ) -> torch.Tensor:
        samples = logits.shape[0]
        token = torch.empty(samples, dtype=torch.int32, device=logits.device)
        torch.ops.tilert.glm5_top1_allreduce_op(logits, sym, mype, npes, flag, token)
        return token
