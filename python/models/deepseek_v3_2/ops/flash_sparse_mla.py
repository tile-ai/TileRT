"""Flash Sparse MLA operation module."""

import math

import torch

from tilert.models.base import TileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.profiler.utils import parse_profile_log_tensor
from tilert.utils import get_profile_log_tensor

__all__ = [
    "flash_sparse_mla",
    "FlashSparseMLACombine",
]


def flash_sparse_mla(
    query: torch.Tensor,
    query_pe: torch.Tensor,
    key_value: torch.Tensor,
    key_pe: torch.Tensor,
    indices: torch.Tensor,
    cur_pos: torch.Tensor,
    output: torch.Tensor,
    profile_logs: torch.Tensor,
    split_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash Sparse MLA operation for GLM5.

    Args:
        query: Query tensor. (bs, seqlen, heads, dim)
        query_pe: Query position embedding tensor. (bs, seqlen, heads, pe_dim)
        key_value: Key-value tensor. (bs, seqlen_kv, dim)
        key_pe: Key position embedding tensor. (bs, seqlen_kv, pe_dim)
        indices: Indices tensor. (bs, seqlen, topk)
        cur_pos: cur_pos tensor. (1)
        output: Output tensor.
        profile_logs: Profile logs tensor.
        split_size: Number of splits.
    """
    batch, seqlen, heads, hidden_dim = query.shape
    if split_size != 64:
        raise ValueError(
            "The current implementation of flash_sparse_mla_op only supports split_size=64"
        )
    if batch != 1:
        raise ValueError("The current implementation of flash_sparse_mla_op only supports batch=1")
    if seqlen > 4:
        raise ValueError(
            "The current implementation of flash_sparse_mla_op only supports seqlen<=4"
        )

    seqlen_kv = key_value.shape[1]
    index_len = indices.shape[-1]
    if index_len > seqlen_kv:
        raise ValueError("index_len must be less than or equal to seqlen_kv")

    device = query.device
    acc_type = torch.float32

    dim = key_value.shape[-1]
    max_num_splits = 32  # topk / split_size = 2048/64

    lse = torch.empty((batch, seqlen, heads), device=device, dtype=acc_type)
    lse_acc = torch.empty((batch, seqlen, heads, max_num_splits), device=device, dtype=acc_type)
    output_acc = torch.empty(
        batch, seqlen, heads, max_num_splits, dim, device=device, dtype=acc_type
    )

    if heads == 16:
        torch.ops.tilert.flash_sparse_mla_op(
            query,
            query_pe,
            key_value,
            key_pe,
            indices,
            cur_pos,
            output,
            output_acc,
            lse,
            lse_acc,
            profile_logs,
            split_size,
        )
    elif heads == 8:
        torch.ops.tilert.flash_sparse_mla_glm5_op(
            query,
            query_pe,
            key_value,
            key_pe,
            indices,
            cur_pos,
            output,
            output_acc,
            lse,
            lse_acc,
            profile_logs,
            split_size,
        )
    else:
        raise ValueError(f"Unsupported heads: {heads}")
    return lse, lse_acc, output_acc


class FlashSparseMLACombine(TileRTModule):
    """Flash Sparse MLA combine module; no weights, uses model_args for scale and config."""

    def __init__(
        self,
        model_args: ModelArgs,
        num_devices: int,
        layer_idx: int = 0,
    ):
        super().__init__(
            type(self).__name__,
            model_args=model_args,
            num_devices=num_devices,
            layer_idx=layer_idx,
        )
        self.tilert_tensor_alias: list[str] = []
        self.ref_tensor_alias: list[str] = []

        scale = (model_args.qk_nope_head_dim + model_args.qk_rope_head_dim) ** -0.5
        if model_args.rope_factor is None:
            mscale = 1.0
        else:
            mscale = 0.1 * math.log(model_args.rope_factor) + 1.0
        self.softmax_scale = scale * mscale * mscale

        self.profile_logs = get_profile_log_tensor()

    def init_reference_weights(
        self, state_dict: dict[str, torch.Tensor], device_id: int = 0
    ) -> None:
        del state_dict, device_id
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict
        self.is_tilert_weights_init = True

    def init_random_weights(self) -> None:
        self.is_ref_weights_init = True
        self.is_tilert_weights_init = True

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len
        self.profile_logs = get_profile_log_tensor()
        self.is_var_init = True

    def golden_forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        cur_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Flash Sparse MLA golden version.

        Args:
            q_nope: Query tensor. (bs, seqlen, heads, dim)
            q_pe: Query position embedding tensor. (bs, seqlen, heads, pe_dim)
            kv_cache: Key-value tensor. (bs, seqlen_kv, dim)
            pe_cache: Key position embedding tensor. (bs, seqlen_kv, pe_dim)
            topk_indices: Indices tensor. (bs, seqlen, topk)
            cur_pos: cur_pos tensor. (1)
        """
        batch_size = q_nope.shape[0]
        seqlen = q_nope.shape[1]
        seqlen_kv = kv_cache.shape[1]

        start_pos = int(cur_pos.item())
        mask = (
            torch.full((seqlen, seqlen_kv), float("-inf")).triu_(start_pos + 1)
            if seqlen > 1
            else None
        )

        scores = (
            torch.einsum("bshc,btc->bsht", q_nope.float(), kv_cache.float())
            + torch.einsum("bshr,btr->bsht", q_pe.float(), pe_cache.float())
        ) * self.softmax_scale
        index_mask = torch.full(
            (batch_size, seqlen, seqlen_kv), float("-inf"), device=q_nope.device
        ).scatter_(-1, topk_indices, 0)
        if mask is not None:
            index_mask += mask

        scores += index_mask.unsqueeze(2)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        return torch.einsum("bsht,btc->bshc", scores.to(torch.bfloat16), kv_cache)

    def tilert_forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        cur_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Flash Sparse MLA tilert version.

        Args:
            q_nope: Query tensor. (bs, seqlen, heads, dim)
            q_pe: Query position embedding tensor. (bs, seqlen, heads, pe_dim)
            kv_cache: Key-value tensor. (bs, seqlen_kv, dim)
            pe_cache: Key position embedding tensor. (bs, seqlen_kv, pe_dim)
            topk_indices: Indices tensor. (bs, seqlen, topk)
            cur_pos: cur_pos tensor. (1)
        """
        batch_size, seqlen, heads, dim = q_nope.shape
        v_dim = kv_cache.shape[-1]

        topk_indices = topk_indices.to(torch.int32)
        topk_indices = topk_indices[..., : kv_cache.shape[1]]
        device = q_nope.device
        if any(t.device != device for t in (q_pe, kv_cache, pe_cache, topk_indices, cur_pos)):
            raise RuntimeError(
                "flash_sparse_mla inputs must be on the same device: "
                f"q_nope={device}, q_pe={q_pe.device}, kv_cache={kv_cache.device}, "
                f"pe_cache={pe_cache.device}, topk_indices={topk_indices.device}, "
                f"cur_pos={cur_pos.device}"
            )
        if self.profile_logs is not None and self.profile_logs.device != device:
            self.profile_logs = get_profile_log_tensor(device_index=device.index, device=device)
        output = torch.zeros(
            (batch_size, seqlen, heads, v_dim), dtype=torch.bfloat16, device=device
        )
        flash_sparse_mla(
            q_nope,
            q_pe,
            kv_cache,
            pe_cache,
            topk_indices,
            cur_pos,
            output,
            self.profile_logs,
        )
        if self.flag_enable_profiling_log:
            # TODO: bug fix for this
            torch.cuda.synchronize()
            parse_profile_log_tensor(
                self.profile_logs, self.get_profile_log_path(), [(self.op_name, 0.0)]
            )
        return output

    def to_tilert_weights(self) -> None:
        raise NotImplementedError("to_tilert_weights not implemented")

    def __call__(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        cur_pos: torch.Tensor,
    ) -> torch.Tensor:
        if self.flag_enable_tilert:
            return self.tilert_forward(q_nope, q_pe, kv_cache, pe_cache, topk_indices, cur_pos)
        return self.golden_forward(q_nope, q_pe, kv_cache, pe_cache, topk_indices, cur_pos)
