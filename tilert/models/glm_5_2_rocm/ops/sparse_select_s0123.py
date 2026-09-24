"""GLM-5.2 fused sparse selector op wrapper (sparse_select_s0123)."""

import torch

from tilert.models.glm_5_2_rocm.ops import rmsnorm_projq_wqi as s1_mod
from tilert.models.glm_5_2_rocm.ops import rmsnorm_projx_wqakis as s0_mod

Q_DIM = s0_mod.Q_DIM
KI_DIM = s0_mod.KI_DIM
IS_DIM = s0_mod.IS_DIM
IQ_DIM = s1_mod.ROWS


class Exchange:
    """The selector's scratch -- allocate ONCE and share across layers."""

    def __init__(self, samples: int, device: torch.device | str = "cuda"):
        self.samples = samples
        self.q_pairs = torch.zeros(samples, Q_DIM // 2, 2, dtype=torch.int32, device=device)
        self.ki_pairs = torch.zeros(samples, KI_DIM // 2, 2, dtype=torch.int32, device=device)
        self.iq_pairs = torch.zeros(samples, IQ_DIM // 2, 2, dtype=torch.int32, device=device)


def fused_forward(
    hidden_in: torch.Tensor,
    s0,
    s1,
    s2,
    s3,
    rope_freqs: torch.Tensor,
    cur_pos: torch.Tensor,
    ki_cache: torch.Tensor,
    ex: Exchange,
    *,
    seq_len: int = 1,
    tag: int = 1,
    want_q: bool = False,
    want_ki: bool = False,
    want_iq: bool = False,
    fp8: dict | None = None,
    want_iq_rt: bool = True,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor
]:
    assert s0.packed is not None and s0.wis_packed is not None
    assert s1.packed is not None
    assert s2.weight is not None and s2.bias is not None
    assert s3.num_heads * KI_DIM == IQ_DIM, "must be the 32-head GPU0 shape"
    assert (
        s0.gamma_arg is not None
        and s0.gamma_arg.numel() > 0
        and (s1.gamma_arg is not None)
        and (s1.gamma_arg.numel() > 0)
    ), "the fused kernel has f32-gamma paths only"
    samples = hidden_in.size(0)
    assert samples == ex.samples
    dev = hidden_in.device
    bf = torch.bfloat16
    q = torch.empty(samples, Q_DIM, dtype=bf, device=dev) if want_q else None
    ki = torch.empty(samples, KI_DIM, dtype=bf, device=dev) if want_ki else None
    is_ = torch.empty(samples, IS_DIM, dtype=bf, device=dev)
    iq = torch.empty(samples, IQ_DIM, dtype=bf, device=dev) if want_iq else None
    if fp8 is not None:
        extra = (fp8["ki_cache8"], fp8["ki_scale"], fp8["iq_rt8"], fp8["iq_scale"])
    else:
        assert want_iq_rt, "only the fp8 arm can drop the bf16 iq_rt"
        extra = (None, None, None, None)
    iq_rt = torch.empty(samples if want_iq_rt else 0, IQ_DIM, dtype=bf, device=dev)
    torch.ops.tilert.glm5_sparse_select_s0123_op(
        hidden_in,
        s0.gamma_arg,
        s0.packed,
        s0.scales,
        s0.wis_packed,
        q,
        ki,
        is_,
        ex.q_pairs,
        ex.ki_pairs,
        ex.iq_pairs,
        s1.gamma_arg,
        s1.packed,
        s1.scales,
        iq,
        s2.weight,
        s2.bias,
        rope_freqs,
        cur_pos,
        ki_cache,
        iq_rt,
        seq_len,
        tag,
        *extra,
    )
    return (q, ki, is_, iq, iq_rt if want_iq_rt else None)


def golden_forward(
    hidden_in: torch.Tensor,
    s0,
    s1,
    s2,
    s3,
    rope_freqs: torch.Tensor,
    cur_pos: torch.Tensor,
    ki_cache: torch.Tensor,
    *,
    seq_len: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError("golden_forward is not available in release builds")


def alloc_fp8(
    samples: int, batch: int, cache_len: int, device: torch.device | str = "cuda"
) -> dict:
    return {
        "ki_cache8": torch.zeros(batch, cache_len, KI_DIM, dtype=torch.uint8, device=device),
        "ki_scale": torch.zeros(batch, cache_len, dtype=torch.float32, device=device),
        "iq_rt8": torch.zeros(samples, IQ_DIM, dtype=torch.uint8, device=device),
        "iq_scale": torch.zeros(samples, IQ_DIM // KI_DIM, dtype=torch.float32, device=device),
    }
