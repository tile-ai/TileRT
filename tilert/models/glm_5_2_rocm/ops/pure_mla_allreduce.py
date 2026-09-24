"""GLM-5.2 attention block in ONE launch (pure_mla_allreduce)."""

import torch

from tilert.models.glm_5_2_rocm.ops.flash_sparse_mla import GLM5_SOFTMAX_SCALE, KV_LORA_RANK, PE_DIM
from tilert.models.glm_5_2_rocm.ops.proj_wkvb import ProjQWkvbGlm5
from tilert.models.glm_5_2_rocm.ops.pure_mla_m567 import PureMlaM567Glm5
from tilert.models.glm_5_2_rocm.ops.qkv_rope import QkvRopeGlm5
from tilert.models.glm_5_2_rocm.ops.rmsnorm_kv import RmsnormKvGlm5
from tilert.models.glm_5_2_rocm.ops.rmsnorm_projq_wqb import RmsnormProjQWqbGlm5
from tilert.models.glm_5_2_rocm.ops.rmsnorm_projx_wqkva import KV_DIM
from tilert.models.glm_5_2_rocm.ops.rmsnorm_projx_wqkva import PE_DIM as M0_PE_DIM
from tilert.models.glm_5_2_rocm.ops.rmsnorm_projx_wqkva import Q_DIM, RmsnormProjXWqkvaGlm5
from tilert.models.glm_5_2_rocm.ops.unprojo_allreduce import HIDDEN

Q_LINE_WORDS = 16
Q_LINE_MAX_SEQ = 4
MAX_SEQ = 4


def q_tiles(num_heads: int) -> int:
    return num_heads * (KV_LORA_RANK + PE_DIM) // 16


def q_words(num_heads: int) -> int:
    return q_tiles(num_heads) * Q_LINE_MAX_SEQ * Q_LINE_WORDS


class Exchange:
    """The attention block's scratch -- allocate ONCE and share across layers."""

    def __init__(self, samples: int, heads: int = 10, device: torch.device | str = "cuda"):
        self.samples = samples
        self.heads = heads
        self.q_pairs = torch.zeros(samples, Q_DIM // 2, 2, dtype=torch.int32, device=device)
        self.kv_pairs = torch.zeros(samples, KV_DIM // 2, 2, dtype=torch.int32, device=device)
        self.pe_pairs = torch.zeros(samples, M0_PE_DIM // 2, 2, dtype=torch.int32, device=device)
        self.m1_pairs = torch.zeros(samples, heads * 256 // 2, 2, dtype=torch.int32, device=device)


class Legs:
    """The three intra-launch legs -- one allocation shared by every layer."""

    def __init__(self, num_heads: int, device: str = "cuda:0") -> None:
        i32 = {"dtype": torch.int32, "device": device}
        self.qlines = torch.zeros(q_words(num_heads), **i32)
        self.kvnew_pairs = torch.zeros(MAX_SEQ * KV_LORA_RANK * 2, **i32)
        self.penew_pairs = torch.zeros(MAX_SEQ * PE_DIM * 2, **i32)


class PureMlaAllReduceGlm5:
    """One rank's whole attention block: the projection weights, the output shards, the exchange, the tail's scratch and the three legs."""

    def __init__(
        self,
        device: str = "cuda:0",
        num_heads: int = 10,
        topk: int = 2048,
        scale: float = GLM5_SOFTMAX_SCALE,
    ) -> None:
        assert num_heads in (8, 10), "num_heads must be 8 or 10"
        self.device = device
        self.num_heads = num_heads
        self.topk = topk
        self.scale = scale
        self.m0 = RmsnormProjXWqkvaGlm5(device=device)
        self.m1 = RmsnormProjQWqbGlm5(device=device, num_heads=num_heads)
        self.m3 = RmsnormKvGlm5(device=device)
        self.m4 = ProjQWkvbGlm5(device=device, num_heads=num_heads)
        self.tail = PureMlaM567Glm5(device=device, num_heads=num_heads, topk=topk, scale=scale)
        self._ex: dict[int, Exchange] = {}
        self.legs = Legs(num_heads, device)
        self._tag = 0

    @property
    def m6(self):
        return self.tail.m6

    @property
    def m7(self):
        return self.tail.m7

    def init_random_weights(self, seed: int = 0) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def exchange(self, samples: int) -> Exchange:
        if samples not in self._ex:
            self._ex[samples] = Exchange(samples, heads=self.num_heads, device=self.device)
        return self._ex[samples]

    def next_tag(self) -> int:
        self._tag += 1
        return self._tag

    def alloc_partials(self, samples: int) -> tuple:
        return self.tail.alloc_partials(samples)

    def forward(
        self,
        hidden_in: torch.Tensor,
        cur_pos: torch.Tensor,
        rope_freqs: torch.Tensor,
        pe_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor | None,
        partials: tuple,
        residual: torch.Tensor | None = None,
        sym: torch.Tensor | None = None,
        mype: int = 0,
        npes: int = 1,
        tag: int | None = None,
        out: torch.Tensor | None = None,
        xfer_buf: torch.Tensor | None = None,
        flag: int = 0,
        timeline: torch.Tensor | None = None,
        q_nope: torch.Tensor | None = None,
        q_pe: torch.Tensor | None = None,
        reuse_selection: int = 0,
    ) -> torch.Tensor:
        acc, pmax, psum = partials
        samples = hidden_in.shape[0]
        dev = hidden_in.device
        if out is None:
            out = torch.empty(samples, HIDDEN, dtype=torch.bfloat16, device=dev)
        if q_nope is None:
            q_nope = torch.empty(samples, self.m4.rows, dtype=torch.bfloat16, device=dev)
        if q_pe is None:
            q_pe = torch.empty(samples, self.m1.pe_dim, dtype=torch.bfloat16, device=dev)
        ex = self.exchange(samples)
        idx = None if indices is None else indices.contiguous().to(torch.int32)
        torch.ops.tilert.glm5_pure_mla_allreduce_op(
            hidden_in,
            self.m0.gamma_arg,
            self.m0.packed,
            self.m0.scales,
            cur_pos,
            pe_cache,
            ex.q_pairs,
            self.m1.gamma_arg,
            self.m1.packed,
            self.m1.scales,
            q_pe,
            ex.kv_pairs,
            ex.pe_pairs,
            ex.m1_pairs,
            self.m3.gamma,
            kv_cache,
            self.m4.packed,
            self.m4.scales,
            q_nope,
            rope_freqs,
            samples,
            self.legs.qlines,
            self.legs.kvnew_pairs,
            self.legs.penew_pairs,
            idx,
            xfer_buf,
            flag,
            acc,
            pmax,
            psum,
            self.tail.sen_a,
            self.tail.sen_b,
            self.tail.sen_proj,
            self.next_tag() if tag is None else tag,
            self.m6.packed,
            self.m6.scales,
            self.m7.packed,
            self.m7.scales,
            residual,
            sym,
            mype,
            npes,
            out,
            self.topk,
            self.scale,
            timeline,
            reuse_selection,
        )
        return out

    def golden_m01234(
        self,
        hidden_in: torch.Tensor,
        cur_pos: torch.Tensor,
        rope_freqs: torch.Tensor,
        pe_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pe_out = pe_cache.clone()
        q_down, kv = self.m0.golden_forward(hidden_in, cur_pos, seq_len, pe_out)
        q_nope_down, q_pe_raw = self.m1.golden_forward(q_down)
        q_pe, pe_out = QkvRopeGlm5(num_heads=self.num_heads).golden_forward(
            q_pe_raw, pe_out, rope_freqs, cur_pos, seq_len
        )
        kv_rows = self.m3.golden_forward(kv)
        kv_out = kv_cache.clone()
        batch = cur_pos.numel()
        for b in range(batch):
            sp = int(cur_pos[b])
            kv_out[b, sp : sp + seq_len] = kv_rows[b * seq_len : (b + 1) * seq_len]
        q_nope = self.m4.golden_forward(q_nope_down)
        return (q_nope, q_pe, pe_out, kv_out)

    def golden_tail(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        indices: torch.Tensor | None,
        cur_pos: torch.Tensor,
        residual: torch.Tensor | None,
        partials: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        from tilert.models.glm_5_2_rocm.ops.flash_sparse_mla import golden_attention

        samples = q_nope.shape[0]
        attn_o = golden_attention(
            q_nope.view(samples, self.num_heads, KV_LORA_RANK),
            q_pe.view(samples, self.num_heads, PE_DIM),
            kv_cache,
            pe_cache,
            indices,
            int(cur_pos.reshape(-1)[0].item()),
            topk=self.topk,
            scale=self.scale,
        )
        part = self.m7.partial_golden(self.m6.golden_forward(attn_o))
        parts = [part] if partials is None else partials
        return self.m7.reduce_golden(parts, residual).to(q_nope.device)

    def golden_forward(
        self,
        hidden_in: torch.Tensor,
        cur_pos: torch.Tensor,
        rope_freqs: torch.Tensor,
        pe_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor | None,
        residual: torch.Tensor | None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise RuntimeError("golden_forward is not available in release builds")
