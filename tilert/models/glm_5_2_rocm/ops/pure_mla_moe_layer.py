"""GLM-5.2 ONE WHOLE MoE LAYER IN ONE LAUNCH (pure_mla_moe_layer)."""

import torch

from tilert.models.glm_5_2_rocm.ops.flash_sparse_mla import GLM5_SOFTMAX_SCALE
from tilert.models.glm_5_2_rocm.ops.fused_moe_allreduce import FUSED_SAMPLES, FusedMoeAllreduceGlm5
from tilert.models.glm_5_2_rocm.ops.fused_moe_w8a8 import FusedMoeAllreduceW8A8Glm5
from tilert.models.glm_5_2_rocm.ops.moe_router import NUM_EXPERTS
from tilert.models.glm_5_2_rocm.ops.pure_mla_allreduce import PureMlaAllReduceGlm5
from tilert.models.glm_5_2_rocm.ops.unprojo_allreduce import HIDDEN
from tilert.models.glm_5_2_rocm.ops.upgate_silu import MOE_SLOTS, NUM_MOE_WEIGHTS, TOP_K

BLOCKS = 256
MAX_SEQ = 4
HLINE_WORDS = 16
MOE_V4 = 0
MOE_W8A8 = 1


def hline_words() -> int:
    return MAX_SEQ * BLOCKS * HLINE_WORDS


class Legs:
    """The attention -> MoE handoff legs -- one allocation for every layer."""

    def __init__(self, device: str = "cuda:0") -> None:
        i32 = {"dtype": torch.int32, "device": device}
        self.hlines = torch.zeros(hline_words(), **i32)


class PureMlaMoeLayerGlm5:
    """One rank's whole MoE layer: the attention block's weights and scratch (``PureMlaAllReduceGlm5``), the MoE banks and workspaces (``FusedMoeAllreduceGlm5`` / the w8a8 subclass) and the handoff legs."""

    def __init__(
        self,
        device: str = "cuda:0",
        num_heads: int = 10,
        topk: int = 2048,
        scale: float = GLM5_SOFTMAX_SCALE,
        moe_w8a8: bool = False,
        num_weights: int = NUM_MOE_WEIGHTS,
    ) -> None:
        self.device = device
        self.num_heads = num_heads
        self.moe_w8a8 = moe_w8a8
        self.mla = PureMlaAllReduceGlm5(device=device, num_heads=num_heads, topk=topk, scale=scale)
        self.moe = (
            FusedMoeAllreduceW8A8Glm5(device=device, num_weights=num_weights)
            if moe_w8a8
            else FusedMoeAllreduceGlm5(device=device, num_weights=num_weights)
        )
        self.legs = Legs(device)
        self._moe_scratch: dict[int, dict[str, torch.Tensor]] = {}

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def next_tag(self) -> int:
        return self.mla.next_tag()

    def next_sen_tag(self) -> int:
        self.moe._sen_tag += 1
        return self.moe._sen_tag

    def alloc_partials(self, samples: int) -> tuple:
        return self.mla.alloc_partials(samples)

    def moe_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.moe.front.router_packed()
        if self.moe_w8a8:
            return (self.moe.pack_k128(), self.moe.pack_down_k128())
        return (self.moe.front.moe.packed, self.moe.pack_down_v4())

    def moe_scratch(self, samples: int) -> dict[str, torch.Tensor]:
        if samples not in self._moe_scratch:
            dev = self.device
            self._moe_scratch[samples] = {
                "q_nope": torch.empty(samples, self.mla.m4.rows, dtype=torch.bfloat16, device=dev),
                "q_pe": torch.empty(samples, self.mla.m1.pe_dim, dtype=torch.bfloat16, device=dev),
                "norm": torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev),
                "scores": torch.empty(samples, NUM_EXPERTS, dtype=torch.float32, device=dev),
                "probs": torch.zeros(samples, TOP_K, dtype=torch.float32, device=dev),
                "indices": torch.zeros(samples, TOP_K, dtype=torch.int32, device=dev),
                "mid": torch.zeros(samples, MOE_SLOTS, 256, dtype=torch.bfloat16, device=dev),
            }
        return self._moe_scratch[samples]

    def forward(
        self,
        hidden_in: torch.Tensor,
        cur_pos: torch.Tensor,
        rope_freqs: torch.Tensor,
        pe_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor | None,
        partials: tuple,
        bias: torch.Tensor,
        residual: torch.Tensor | None = None,
        sym_attn: torch.Tensor | None = None,
        sym_ffn: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        tag: int | None = None,
        sen_tag: int | None = None,
        ffn_flag: int | None = None,
        unproj_o: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
        xfer_buf: torch.Tensor | None = None,
        flag: int = 0,
        timeline: torch.Tensor | None = None,
        reuse_selection: int = 0,
    ) -> torch.Tensor:
        samples = hidden_in.shape[0]
        assert samples in FUSED_SAMPLES
        dev = hidden_in.device
        mla = self.mla
        acc, pmax, psum = partials
        if unproj_o is None:
            unproj_o = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev)
        if x_out is None:
            x_out = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev)
        ex = mla.exchange(samples)
        idx = None if indices is None else indices.contiguous().to(torch.int32)
        t = self.next_tag() if tag is None else tag
        st = self.next_sen_tag() if sen_tag is None else sen_tag
        ff = st if ffn_flag is None else ffn_flag
        sc = self.moe_scratch(samples)
        q_nope, q_pe = (sc["q_nope"], sc["q_pe"])
        ug_w, down_w = self.moe_banks()
        torch.ops.tilert.glm5_pure_mla_moe_layer_op(
            hidden_in,
            mla.m0.gamma_arg,
            mla.m0.packed,
            mla.m0.scales,
            cur_pos,
            pe_cache,
            ex.q_pairs,
            mla.m1.gamma_arg,
            mla.m1.packed,
            mla.m1.scales,
            q_pe,
            ex.kv_pairs,
            ex.pe_pairs,
            ex.m1_pairs,
            mla.m3.gamma,
            kv_cache,
            mla.m4.packed,
            mla.m4.scales,
            q_nope,
            rope_freqs,
            samples,
            mla.legs.qlines,
            mla.legs.kvnew_pairs,
            mla.legs.penew_pairs,
            idx,
            xfer_buf,
            flag,
            acc,
            pmax,
            psum,
            mla.tail.sen_a,
            mla.tail.sen_b,
            mla.tail.sen_proj,
            t,
            mla.m6.packed,
            mla.m6.scales,
            mla.m7.packed,
            mla.m7.scales,
            residual,
            sym_attn,
            mype,
            npes,
            unproj_o,
            mla.topk,
            mla.scale,
            [
                self.moe.front.router.gamma,
                self.moe.front.router_packed(),
                ug_w,
                self.moe.front.moe.scales,
                bias,
                down_w,
                self.moe.down.scales,
            ],
            sym_ffn,
            ff,
            [
                sc["norm"],
                sc["scores"],
                self.moe.score_lines[:samples],
                self.moe.flags,
                sc["probs"],
                sc["indices"],
                sc["mid"],
                x_out,
                self.moe.mid_pairs[:samples],
                self.legs.hlines,
            ],
            st,
            MOE_W8A8 if self.moe_w8a8 else MOE_V4,
            timeline,
            reuse_selection,
        )
        return x_out

    def chain_forward(
        self,
        hidden_in: torch.Tensor,
        cur_pos: torch.Tensor,
        rope_freqs: torch.Tensor,
        pe_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor | None,
        partials: tuple,
        bias: torch.Tensor,
        residual: torch.Tensor | None = None,
        sym_attn: torch.Tensor | None = None,
        sym_ffn: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        tag: int | None = None,
        sen_tag: int | None = None,
        ffn_flag: int | None = None,
        unproj_o: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
        xfer_buf: torch.Tensor | None = None,
        flag: int = 0,
        timeline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = hidden_in.shape[0]
        dev = hidden_in.device
        if unproj_o is None:
            unproj_o = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev)
        if x_out is None:
            x_out = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev)
        t = self.next_tag() if tag is None else tag
        st = self.next_sen_tag() if sen_tag is None else sen_tag
        ff = st if ffn_flag is None else ffn_flag
        sc = self.moe_scratch(samples)
        ug_w, down_w = self.moe_banks()
        router_w = self.moe.front.router_packed()
        self.mla.forward(
            hidden_in,
            cur_pos,
            rope_freqs,
            pe_cache,
            kv_cache,
            indices,
            partials,
            residual,
            sym=sym_attn,
            mype=mype,
            npes=npes,
            tag=t,
            out=unproj_o,
            xfer_buf=xfer_buf,
            flag=flag,
            q_nope=sc["q_nope"],
            q_pe=sc["q_pe"],
            timeline=timeline,
        )
        moe_op = (
            torch.ops.tilert.glm5_fused_moe_allreduce_w8a8_v4_op
            if self.moe_w8a8
            else torch.ops.tilert.glm5_fused_moe_allreduce_v4_op
        )
        moe_op(
            unproj_o,
            self.moe.front.router.gamma,
            router_w,
            ug_w,
            self.moe.front.moe.scales,
            bias,
            down_w,
            self.moe.down.scales,
            unproj_o,
            sym_ffn,
            mype,
            npes,
            ff,
            sc["norm"],
            sc["scores"],
            self.moe.score_lines[:samples],
            self.moe.flags,
            sc["probs"],
            sc["indices"],
            sc["mid"],
            x_out,
            self.moe.mid_pairs[:samples],
            st,
        )
        return (x_out, unproj_o)


__all__ = ["MOE_V4", "MOE_W8A8", "Legs", "PureMlaMoeLayerGlm5", "hline_words"]
