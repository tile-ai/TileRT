"""GLM-5.2 W8A8 MoE monokernel (v4 skeleton): the v4 op's banks + the fp8 packings + the quantized-math goldens."""

import torch

from tilert.models.glm_5_2_rocm.ops.fused_moe_allreduce import (
    FUSED_SAMPLES,
    GRID_BLOCKS,
    FusedMoeAllreduceGlm5,
)
from tilert.models.glm_5_2_rocm.ops.moe_router import HIDDEN, NUM_EXPERTS
from tilert.models.glm_5_2_rocm.ops.upgate_silu import (
    MOE_SLOTS,
    NUM_MOE_WEIGHTS,
    SCALE_BLK,
    TOP_K,
    _silu,
)
from tilert.models.glm_5_2_rocm.ops.upgate_silu_w8a8 import (
    quant_act_row,
    quant_mid_rows,
    swizzle_pair_interleaved_k128,
)


def swizzle_down_k128(w_fp8: torch.Tensor) -> torch.Tensor:
    rows, k = w_fp8.shape
    assert rows % 24 == 0 and k == 256
    w8 = w_fp8.view(torch.uint8)
    out = torch.zeros(rows // 24 * 6144, dtype=torch.uint8)
    lane = torch.arange(64)
    h = torch.arange(2)
    i = torch.arange(16)
    L, H, II = torch.meshgrid(lane, h, i, indexing="ij")
    for blk in range(rows // 24):
        r0 = blk * 24
        base = blk * 6144
        for c in range(2):
            ks = c * 128 + L // 16 * 32 + H * 16 + II
            main = w8[r0 + L % 16, ks]
            idx = base + c * 2048 + H * 1024 + L * 16 + II
            out[idx.reshape(-1)] = main.reshape(-1)
        m8 = torch.arange(8)
        g = torch.arange(4)
        G, M, H2, I2 = torch.meshgrid(g, m8, h, i, indexing="ij")
        for c in range(2):
            ks = c * 128 + G * 32 + H2 * 16 + I2
            tail = w8[r0 + 16 + M, ks]
            idx = base + 4096 + c * 1024 + H2 * 512 + (G * 8 + M) * 16 + I2
            out[idx.reshape(-1)] = tail.reshape(-1)
    return out.contiguous()


class FusedMoeAllreduceW8A8Glm5(FusedMoeAllreduceGlm5):
    """The v4 monokernel's banks + the fp8 packings + the W8A8 op."""

    def __init__(self, device: str = "cuda:0", num_weights: int = NUM_MOE_WEIGHTS):
        super().__init__(device=device, num_weights=num_weights)
        self.packed_k128: torch.Tensor | None = None
        self.down_k128: torch.Tensor | None = None

    def pack_k128(self) -> torch.Tensor:
        if self.packed_k128 is None:
            moe = self.front.moe
            assert moe.w_fp8 is not None
            ps = [
                swizzle_pair_interleaved_k128(moe.w_fp8[e], moe.inter)
                for e in range(moe.num_weights)
            ]
            self.packed_k128 = torch.cat(ps).contiguous().to(self.device)
        return self.packed_k128

    def pack_down_k128(self) -> torch.Tensor:
        if self.down_k128 is None:
            assert self.down.w_fp8 is not None
            ps = [swizzle_down_k128(self.down.w_fp8[e]) for e in range(self.down.w_fp8.shape[0])]
            self.down_k128 = torch.cat(ps).contiguous().to(self.device)
        return self.down_k128

    def golden_down_w8a8(
        self,
        mid_bf16: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.down.w_fp8 is not None and self.down.scales is not None
        q, ds = quant_mid_rows(mid_bf16.cpu())
        qf = q.float()
        wq = self.down.w_fp8.float().cpu()
        scales = self.down.scales.cpu()
        idx = indices.cpu()
        pr = probs.cpu()
        s_n = mid_bf16.shape[0]
        out = torch.zeros(s_n, HIDDEN, dtype=torch.float32)
        row_blk = torch.arange(HIDDEN) // 128
        for s in range(s_n):
            for slot in range(MOE_SLOTS):
                e = 0 if slot == 0 else 1 + int(idx[s, slot - 1])
                w = 1.0 if slot == 0 else float(pr[s, slot - 1])
                acc = torch.zeros(HIDDEN, dtype=torch.float32)
                for kb in range(2):
                    ks = slice(kb * 128, (kb + 1) * 128)
                    part = wq[e][:, ks] @ qf[s, slot, ks]
                    a_sc = float(ds[s, slot, kb])
                    acc += part * (scales[e][row_blk, kb] * w * a_sc)
                out[s] += acc
        out = out.to(torch.bfloat16).float()
        if residual is not None:
            out += residual.float().cpu()
        return out.to(torch.bfloat16).to(mid_bf16.device)

    def golden_forward_w8a8(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        moe = self.front.moe
        assert moe.w_fp8 is not None and moe.scales is not None
        norm_ref, scores_ref = self.front.router.golden_forward(hidden)
        act8, a_scales = quant_act_row(norm_ref.cpu())
        a = act8.float()
        wq = moe.w_fp8.float().cpu()
        scales = moe.scales.cpu()
        idx = indices.cpu()
        s_n = a.shape[0]
        inter = moe.inter
        mid = torch.zeros(s_n, MOE_SLOTS, inter, dtype=torch.float32)
        n_kb = HIDDEN // SCALE_BLK
        for s in range(s_n):
            for slot in range(MOE_SLOTS):
                e = 0 if slot == 0 else 1 + int(idx[s, slot - 1])
                acc = torch.zeros(2 * inter, dtype=torch.float32)
                for kb in range(n_kb):
                    ks = slice(kb * SCALE_BLK, (kb + 1) * SCALE_BLK)
                    part = wq[e][:, ks] @ a[s, ks]
                    row_blk = torch.arange(2 * inter) // SCALE_BLK
                    acc += part * (scales[e][row_blk, kb] * float(a_scales[s, kb]))
                gate, up = (acc[:inter], acc[inter:])
                mid[s, slot] = _silu(gate) * up
        mid_bf16 = mid.to(torch.bfloat16).to(hidden.device)
        out_ref = self.golden_down_w8a8(mid_bf16, probs, indices, residual)
        return (norm_ref, scores_ref, mid_bf16, out_ref)

    def tilert_forward_w8a8_v4(
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
        import os

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
        torch.ops.tilert.glm5_fused_moe_allreduce_w8a8_v4_op(
            hidden,
            self.front.router.gamma,
            self.front.router_packed(),
            self.pack_k128(),
            self.front.moe.scales,
            bias,
            self.pack_down_k128(),
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


__all__ = [
    "FusedMoeAllreduceW8A8Glm5",
    "GRID_BLOCKS",
    "quant_act_row",
    "quant_mid_rows",
    "swizzle_down_k128",
    "swizzle_pair_interleaved_k128",
]
