"""V2 MLA weight generator classes for device-group-specific pipelines.

SparseSelectMlaV2 — Device Group A (GPU 0): sparse selector MLA.
PureMlaV2         — Device Group B (GPU 1-7): pure MLA.
"""

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.layernorm_rope_rotate import LayerNormRoPERotate
from tilert.models.deepseek_v3_2.ops.projo_wkvb import ProjoWKVb
from tilert.models.deepseek_v3_2.ops.projq_wqb import ProjqWqb
from tilert.models.deepseek_v3_2.ops.projx_wis import ProjxWis
from tilert.models.deepseek_v3_2.ops.rmsnorm_kv import KVRMSNorm
from tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqb import (
    RmsnormProjqWqb,
    RmsnormProjqWqbAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqi import (
    RmsnormProjqWqi,
    RmsnormProjqWqiAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqakis import (
    RMSNormProjxWqakis,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqkva import (
    RMSNormProjxWqkva,
    RMSNormProjxWqkvaAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.unproj_o_allreduce import (
    UnProjOAllReduce,
    UnProjOAllReduceAlgorithm,
)


class SparseSelectMlaV2(SerializableTileRTModule):
    """Device Group A (GPU 0): sparse selector MLA.

    10 ops registered (DECOUPLED: rmsnorm_quant + projx_wqaki + projx_wis):
      1a/1b/1c. RmsnormQuant + ProjXWqaki + ProjXWis (DECOUPLED) -> 3 weights (idx 0-2)
      2.        RmsnormProjqWqi               -> 3 weights (idx 3-5)
      3.        LayerNormRoPERotate           -> 2 weights (idx 6-7)
      4.        ProjxWis                      -> 1 weight  (idx 8)
    Plus 2 manually appended runtime tensors (idx 9-10).

    11 param tensors total, matching C++ SparseSelectMlaV2::params_size.
    """

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        peer_bufs: torch.Tensor | None = None,
        partial_buf: torch.Tensor | None = None,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        # Op 1a/1b: RMSNorm + FP8 quantize, then ProjXWqakis FP8 MMA GEMV
        self.rmsnorm_projx_wqakis = RMSNormProjxWqakis(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_projx_wqakis)

        # Op 2: q RMSNorm + IQ-only projection (32 heads)
        self.rmsnorm_projq_wqi = RmsnormProjqWqi(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projq_wqi.algorithm = RmsnormProjqWqiAlgorithm.BF16MMA
        self.register_op(self.rmsnorm_projq_wqi)

        # Op 3: ki LayerNorm + RoPE + Rotate
        self.layernorm_rope_rotate = LayerNormRoPERotate(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.layernorm_rope_rotate)

        # Op 4: importance score projection
        self.projx_wis = ProjxWis(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projx_wis)

        # Runtime tensors (appended after op weights in get_weights_list).
        # Shared across all layers on this device — caller provides pre-allocated
        # CUDA tensors so that P2P pointer exchange can fill them after init.
        self.peer_bufs = peer_bufs
        self.partial_buf = partial_buf

        # Cache tensors
        self.ki_cache: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        """Return weight tensors in C++ params_ptr order.

        idx 0-8:   op weights from registered ops.
        idx 9:     peer_bufs   (peer buffer pointers).
        idx 10:    partial_buf (zero-filled BF16 buffer for PaddedAllReduceAdd).
        """
        weights = super().get_weights_list()

        dev = f"cuda:{self.device_id}"
        # Allocate runtime tensors lazily on the correct CUDA device.
        if self.peer_bufs is None:
            self.peer_bufs = torch.zeros(self.num_devices - 1, dtype=torch.int64, device=dev)
        if self.partial_buf is None:
            self.partial_buf = torch.zeros(
                self.model_args.max_batch_size,
                4,
                self.model_args.dim,
                dtype=torch.bfloat16,
                device=dev,
            )

        weights.append(self.peer_bufs)
        weights.append(self.partial_buf)

        return weights

    def get_cache_vars(self) -> list[torch.Tensor]:
        """Return [ki_cache, kv_cache, pe_cache] matching DsaCacheVars layout."""
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        bs_args = (self.model_args.max_batch_size, cache_seq_len)

        if self.ki_cache is None:
            ki_dim = self.model_args.index_head_dim
            self.ki_cache = torch.zeros(
                *bs_args, ki_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.kv_cache is None:
            kv_dim = self.model_args.kv_lora_rank
            self.kv_cache = torch.zeros(
                *bs_args, kv_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.pe_cache is None:
            pe_dim = self.model_args.qk_rope_head_dim
            self.pe_cache = torch.zeros(
                *bs_args, pe_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        return [*super().get_cache_vars(), self.ki_cache, self.kv_cache, self.pe_cache]


class PureMlaV2(SerializableTileRTModule):
    """Device Group B (GPU 1-7): pure MLA.

    10 ops (DECOUPLED: rmsnorm_quant + projx_wqkva replaces fused Op1).
    Weight tensor ordering:
      1a/1b. RMSNormProjxWqkva (FP8_MMA_GEMV) -> 3 weights (idx 0-2, scales=dummy)
      2.     RmsnormProjqWqb                   -> 3 weights (idx 3-5)
      3.     KVRMSNorm                         -> 1 weight  (idx 6)
      4.     ProjqWqb                          -> 2 weights (idx 7-8)
      5.     ProjoWKVb                         -> 2 weights (idx 9-10)
      6.     UnProjOAllReduce                  -> 2 weights (idx 11-12)
    Plus 1 manually appended runtime tensor (idx 13).

    14 param tensors total, matching C++ PureMlaV2::params_size.
    """

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        ll_buf: torch.Tensor | None = None,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        # Op 1a/1b: RMSNorm + FP8 quantize, then ProjXWqkva FP8 MMA GEMV.
        # Decoupled into standalone rmsnorm_quant + projx_wqkva because the
        # fused RMSNorm+GEMV overflows piped pipeline smem at kSeqLen>=2.
        self.rmsnorm_projx_wqkva = RMSNormProjxWqkva(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projx_wqkva.algorithm = RMSNormProjxWqkvaAlgorithm.DECOUPLED
        self.register_op(self.rmsnorm_projx_wqkva)

        # Op 2: q RMSNorm + W_q_b projection (H=10, no IQ)
        self.rmsnorm_projq_wqb = RmsnormProjqWqb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projq_wqb.algorithm = RmsnormProjqWqbAlgorithm.BF16MMA
        self.register_op(self.rmsnorm_projq_wqb)

        # Op 3: KV cache normalization
        self.rmsnorm_kv = KVRMSNorm(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_kv)

        # Op 4: per-head q_nope_down -> q_nope
        self.projq_wqb = ProjqWqb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projq_wqb)

        # Op 5: per-head output projection o -> proj_o
        self.projo_wkvb = ProjoWKVb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projo_wkvb)

        # Op 6: GEMV + 8-GPU AllReduce + residual (BF16 MMA on DSv32 B200).
        allreduce_algo = UnProjOAllReduceAlgorithm.BF16MMA
        self.unproj_o_allreduce = UnProjOAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=allreduce_algo,
        )
        self.register_op(self.unproj_o_allreduce)

        # Runtime tensor (appended after op weights in get_weights_list).
        # Shared across all layers on this device — caller provides a pre-allocated
        # CUDA tensor. GPU 0's peer_bufs points to this buffer on each peer.
        self.ll_buf = ll_buf

        # Cache tensors
        self.ki_cache: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

    def init_random_weights(self) -> None:
        """Override to re-init ProjQWkvb/ProjOWkvb with HMMA-packed weights.

        Base class calls each op's init_random_weights (GENERAL format).
        PureMlaV2 needs HMMA format for ProjQWkvb and ProjOWkvb. The kBf16MMA
        kernel reuses the same FP8-packed page layout as the FP16 HMMA kernel
        (it does FP8->BF16 cvt internally), so the same Python converter is
        correct for both compute paths.
        """
        super().init_random_weights()

        # Re-init ProjQWkvb and ProjOWkvb: generate fresh random FP8 weights,
        # shard them, and call init_tilert_weights_hmma.
        from tilert.models.common import init_func

        for op in [self.projq_wqb, self.projo_wkvb]:
            padded_total = op.num_local_heads * op.num_devices
            w = init_func(
                torch.empty(
                    padded_total * op.wkvb_head_dim, op.wkvb_lora_rank, dtype=torch.float8_e4m3fn
                )
            )
            s = init_func(
                torch.empty(
                    padded_total * op.wkvb_head_dim // op.model_args.block_size,
                    op.wkvb_lora_rank_qsize,
                    dtype=torch.float32,
                )
            )
            ref_dict = dict(zip(op.ref_weights_alias(), [w, s]))
            op.init_reference_weights(ref_dict)
            sharded = op.device_sharding(ref_dict)
            per_dev = {k: v[op.device_id] for k, v in sharded.items()}
            op.init_tilert_weights_hmma(per_dev)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Override to use HMMA-packed weights for ProjQWkvb and ProjOWkvb."""
        # Mark projq_wqb and projo_wkvb so the base loop skips them.
        self.projq_wqb.is_tilert_weights_init = True
        self.projo_wkvb.is_tilert_weights_init = True

        # Let the base class init all other ops normally.
        super().init_tilert_weights(state_dict)

        # Now init the two ops with HMMA-packed weights (the kBf16MMA kernel
        # reuses the FP8-packed HMMA page layout).
        for op in [self.projq_wqb, self.projo_wkvb]:
            op_state_dict = {}
            for op_key in op.get_tilert_weights_alias():
                for p, s in zip(self.prefix_seq, self.suffix_seq):
                    original_key = f"{p}{op_key}{s}"
                    if original_key in state_dict:
                        op_state_dict[op_key] = state_dict[original_key]
                        break
            op.is_tilert_weights_init = False
            op.init_tilert_weights_hmma(op_state_dict)

    def get_weights_list(self) -> list[torch.Tensor]:
        """Return weight tensors in C++ params_ptr order.

        idx 0-12:  op weights from registered ops.
        idx 13:    ll_buf (receive buffer for ReceiveSelectedTokenIds).
        """
        weights = super().get_weights_list()

        # Allocate on the correct CUDA device lazily.
        if self.ll_buf is None:
            max_seq_len = getattr(self.model_args, "num_mtp", 3) + 1  # kSeqLen for MTP decode
            topk = self.model_args.index_topk  # 2048
            self.ll_buf = torch.zeros(
                max_seq_len * topk * 2, dtype=torch.int32, device=f"cuda:{self.device_id}"
            )

        weights.append(self.ll_buf)

        return weights

    def get_cache_vars(self) -> list[torch.Tensor]:
        """Return [ki_cache, kv_cache, pe_cache] matching DsaCacheVars layout."""
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        bs_args = (self.model_args.max_batch_size, cache_seq_len)

        if self.ki_cache is None:
            ki_dim = self.model_args.index_head_dim
            self.ki_cache = torch.zeros(
                *bs_args, ki_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.kv_cache is None:
            kv_dim = self.model_args.kv_lora_rank
            self.kv_cache = torch.zeros(
                *bs_args, kv_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        if self.pe_cache is None:
            pe_dim = self.model_args.qk_rope_head_dim
            self.pe_cache = torch.zeros(
                *bs_args, pe_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        return [*super().get_cache_vars(), self.ki_cache, self.kv_cache, self.pe_cache]
