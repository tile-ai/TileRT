"""GLM-5.2 MoE monokernel (v4) op wrapper: banks, packers, golden, forward."""

import os

import torch

from tilert.models.glm_5_2_rocm.ops.moe_router import HIDDEN, NUM_EXPERTS, MoeRouterGlm5
from tilert.models.glm_5_2_rocm.ops.upgate_silu import (
    MOE_SLOTS,
    NUM_MOE_WEIGHTS,
    TOP_K,
    MoeUpGateSiluGlm5,
)

GRID_BLOCKS = 256
SCORE_LINE_WORDS = 32
FUSED_SAMPLES = (1, 2, 4)


class MoeFrontBanksGlm5:
    """Router bank + ONE MoE up/gate bank (expert 0 = shared) + one gamma: the monokernel's front-half weights and their golden."""

    def __init__(self, device: str = "cuda:0", num_weights: int = NUM_MOE_WEIGHTS):
        self.device = device
        self.router = MoeRouterGlm5(device=device)
        self.moe = MoeUpGateSiluGlm5(device=device, num_weights=num_weights)
        self.packed_m4: torch.Tensor | None = None

    @property
    def gamma(self) -> torch.Tensor | None:
        return self.router.gamma

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def router_packed(self) -> torch.Tensor:
        if self.packed_m4 is None:
            from tilert.models.glm_5_2_rocm.ops.eh_proj_allreduce import swizzle_256_bf16

            assert self.router.w is not None
            self.packed_m4 = swizzle_256_bf16(self.router.w).to(self.device)
        return self.packed_m4

    def golden_forward(
        self, hidden: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")


class FusedMoeAllreduceGlm5:
    """The WHOLE MoE chain in one launch: MoeFrontBanksGlm5's banks plus a down bank (K-split packing) and the all-reduce plumbing."""

    def __init__(self, device: str = "cuda:0", num_weights: int = NUM_MOE_WEIGHTS):
        from tilert.models.glm_5_2_rocm.ops.moe_down_allreduce import MoeDownBankGlm5

        self.device = device
        self.front = MoeFrontBanksGlm5(device=device, num_weights=num_weights)
        self.down = MoeDownBankGlm5(device=device, num_weights=num_weights)
        self.score_lines = torch.zeros(
            max(FUSED_SAMPLES), 32, SCORE_LINE_WORDS, dtype=torch.int32, device=device
        )
        self.flags = torch.zeros(2 * GRID_BLOCKS, dtype=torch.int32, device=device)
        self.mid_pairs = torch.zeros(
            max(FUSED_SAMPLES), MOE_SLOTS, 256, dtype=torch.int32, device=device
        )
        self._sen_tag = 0
        self._down_v4: torch.Tensor | None = None

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def golden_forward(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")

    def pack_down_v4(self) -> torch.Tensor:
        if self._down_v4 is None:
            assert self.down.w_fp8 is not None
            w8 = self.down.w_fp8.view(torch.uint8)
            e = w8.shape[0]
            v = w8.view(e, 256, 6, 4, 8, 2, 16).permute(0, 1, 4, 5, 2, 3, 6)
            self._down_v4 = v.reshape(e, -1).contiguous().view(-1).to(self.device)
        return self._down_v4

    def tilert_forward(
        self,
        hidden: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor | None = None,
        proto: int = 0,
        sym: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        flag: int = 1,
        out: torch.Tensor | None = None,
        sen_tag: int | None = None,
    ):
        assert self.front.moe.packed is not None
        os.environ["TILERT_GLM5_AR_PROTO"] = str(proto)
        s_n = hidden.shape[0]
        assert s_n in FUSED_SAMPLES
        if sen_tag is None:
            self._sen_tag += 1
            sen_tag = self._sen_tag
        dev = hidden.device
        norm = torch.empty(s_n, HIDDEN, dtype=torch.bfloat16, device=dev)
        scores = torch.empty(s_n, NUM_EXPERTS, dtype=torch.float32, device=dev)
        probs = torch.zeros(s_n, TOP_K, dtype=torch.float32, device=dev)
        indices = torch.zeros(s_n, TOP_K, dtype=torch.int32, device=dev)
        mid = torch.zeros(s_n, MOE_SLOTS, 256, dtype=torch.bfloat16, device=dev)
        if out is None:
            out = torch.empty(s_n, HIDDEN, dtype=torch.bfloat16, device=dev)
        torch.ops.tilert.glm5_fused_moe_allreduce_v4_op(
            hidden,
            self.front.router.gamma,
            self.front.router_packed(),
            self.front.moe.packed,
            self.front.moe.scales,
            bias,
            self.pack_down_v4(),
            self.down.scales,
            residual,
            sym,
            mype,
            npes,
            flag,
            norm,
            scores,
            self.score_lines[:s_n],
            self.flags,
            probs,
            indices,
            mid,
            out,
            self.mid_pairs[:s_n],
            sen_tag,
        )
        return (norm, scores, mid, probs, indices, out)
