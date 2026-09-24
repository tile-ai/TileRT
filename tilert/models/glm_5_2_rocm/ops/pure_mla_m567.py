"""GLM-5.2 attention tail (split + combine, ProjOWkvb, UnprojOAllReduce): the tail half's weight shards and scratch, as ``pure_mla_allreduce`` composes them."""

import torch

from tilert.models.glm_5_2_rocm.ops.flash_sparse_mla import GLM5_SOFTMAX_SCALE, KV_LORA_RANK, TILE_N
from tilert.models.glm_5_2_rocm.ops.proj_wkvb import ProjOWkvbGlm5
from tilert.models.glm_5_2_rocm.ops.unprojo_allreduce import UnprojOAllReduceGlm5

XCDS = 8
M56_MAX_SEQ = 8
TILES_PER_HEAD = 16
MAX_SEQ = 4
BLOCKS, STAMPS = (256, 19)
STAMP_NAMES = (
    "start",
    "split",
    "merge",
    "m6poll",
    "m6mfma",
    "m6pub",
    "publish",
    "leadspin",
    "leadflag",
    "xflag",
    "fill",
    "gemv",
    "exchange",
    "end",
    "gather",
    "fix",
    "q",
    "score",
    "idx",
)
LINE_WORDS = 16
XCDS_LEADERS = 8


def proj_words(num_heads: int) -> int:
    nlines = num_heads * TILES_PER_HEAD * MAX_SEQ
    return nlines * LINE_WORDS + XCDS * nlines * 16


class PureMlaM567Glm5:
    """One rank's tail banks: the projection shard, the W_o shard and the tail's scratch (zero-init once, never reset)."""

    def __init__(
        self,
        device: str = "cuda:0",
        num_heads: int = 10,
        topk: int = 2048,
        scale: float = GLM5_SOFTMAX_SCALE,
    ) -> None:
        assert num_heads in (8, 10), "weight layout is H = 8 or 10"
        assert topk > 0 and topk % TILE_N == 0 and (topk <= 2048)
        self.device = device
        self.num_heads = num_heads
        self.topk = topk
        self.scale = scale
        self.num_splits_max = topk // TILE_N
        self.m6 = ProjOWkvbGlm5(device=device, num_heads=num_heads)
        self.m7 = UnprojOAllReduceGlm5(num_heads=num_heads, device=device)
        i32 = {"dtype": torch.int32, "device": device}
        self.sen_a = torch.zeros(2 * XCDS * self.num_splits_max, **i32)
        self.sen_b = torch.zeros(M56_MAX_SEQ * num_heads * KV_LORA_RANK, **i32)
        self.sen_proj = torch.zeros(proj_words(num_heads), **i32)
        self._tag = 0

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def next_tag(self) -> int:
        self._tag += 1
        return self._tag

    def alloc_partials(self, samples: int) -> tuple:
        shape = (samples, self.num_heads, self.num_splits_max)
        return (
            torch.empty(shape + (KV_LORA_RANK,), dtype=torch.float32, device=self.device),
            torch.empty(shape + (1,), dtype=torch.float32, device=self.device),
            torch.empty(shape + (1,), dtype=torch.float32, device=self.device),
        )
