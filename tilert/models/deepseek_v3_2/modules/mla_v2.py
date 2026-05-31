"""MLA weight generator classes for device-group-specific pipelines."""

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
    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        peer_bufs: torch.Tensor | None = None,
        partial_buf: torch.Tensor | None = None,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_projx_wqakis = RMSNormProjxWqakis(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_projx_wqakis)

        self.rmsnorm_projq_wqi = RmsnormProjqWqi(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projq_wqi.algorithm = RmsnormProjqWqiAlgorithm.BF16MMA
        self.register_op(self.rmsnorm_projq_wqi)

        self.layernorm_rope_rotate = LayerNormRoPERotate(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.layernorm_rope_rotate)

        self.projx_wis = ProjxWis(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projx_wis)

        self.peer_bufs = peer_bufs
        self.partial_buf = partial_buf

        self.ki_cache: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        """Return weight tensors."""
        weights = super().get_weights_list()

        dev = f"cuda:{self.device_id}"
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

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        ll_buf: torch.Tensor | None = None,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_projx_wqkva = RMSNormProjxWqkva(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projx_wqkva.algorithm = RMSNormProjxWqkvaAlgorithm.DECOUPLED
        self.register_op(self.rmsnorm_projx_wqkva)

        self.rmsnorm_projq_wqb = RmsnormProjqWqb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.rmsnorm_projq_wqb.algorithm = RmsnormProjqWqbAlgorithm.BF16MMA
        self.register_op(self.rmsnorm_projq_wqb)

        self.rmsnorm_kv = KVRMSNorm(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_kv)

        self.projq_wqb = ProjqWqb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projq_wqb)

        self.projo_wkvb = ProjoWKVb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projo_wkvb)

        allreduce_algo = UnProjOAllReduceAlgorithm.BF16MMA
        self.unproj_o_allreduce = UnProjOAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=allreduce_algo,
        )
        self.register_op(self.unproj_o_allreduce)

        self.ll_buf = ll_buf

        self.ki_cache: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None

    def init_random_weights(self) -> None:
        """Initialize random weights for this module."""
        super().init_random_weights()

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
        """Load TileRT weights for this module from state_dict."""
        self.projq_wqb.is_tilert_weights_init = True
        self.projo_wkvb.is_tilert_weights_init = True

        super().init_tilert_weights(state_dict)

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
        """Return weight tensors."""
        weights = super().get_weights_list()

        if self.ll_buf is None:
            max_seq_len = getattr(self.model_args, "num_mtp", 3) + 1
            topk = self.model_args.index_topk
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
