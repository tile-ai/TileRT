"""GLM-5.2 dense MLP DownAllReduce op wrapper (layers 0-2)."""

import os

import torch

from tilert.models.glm_5_2_rocm.ops import unprojo_allreduce as m7

DENSE_INTER = 1536
HIDDEN = m7.HIDDEN
PROTO_0, PROTO_1 = (m7.PROTO_0, m7.PROTO_1)
sym_bytes = m7.sym_bytes
sym_buffer = m7.sym_buffer
sym_table = m7.sym_table
enable_peer_access = m7.enable_peer_access


class DownAllReduceGlm5(m7.UnprojOAllReduceGlm5):
    """One rank's dense-MLP Wdown shard."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.num_heads = 0
        self.k = DENSE_INTER
        self.device = device
        self.w_fp8 = None
        self.scales = None
        self.packed = None

    def tilert_forward(
        self,
        mid: torch.Tensor,
        residual: torch.Tensor | None = None,
        proto: int = PROTO_0,
        sym: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.packed is not None and self.scales is not None
        os.environ["TILERT_GLM5_AR_PROTO"] = str(proto)
        if out is None:
            out = torch.empty(mid.shape[0], HIDDEN, dtype=torch.bfloat16, device=mid.device)
        torch.ops.tilert.glm5_down_allreduce_op(
            mid, self.packed, self.scales, residual, sym, mype, npes, flag, out
        )
        return out
