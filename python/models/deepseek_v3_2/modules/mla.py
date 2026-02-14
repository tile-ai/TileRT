import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.ops.layernorm_rope_rotate import LayerNormRoPERotate
from tilert.models.deepseek_v3_2.ops.projo_wkvb import ProjoWKVb
from tilert.models.deepseek_v3_2.ops.projq_wqb import ProjqWqb
from tilert.models.deepseek_v3_2.ops.projx_wis import ProjxWis
from tilert.models.deepseek_v3_2.ops.rmsnorm_kv import KVRMSNorm
from tilert.models.deepseek_v3_2.ops.rmsnorm_projq_wqib import (
    RmsnormProjqWqib,
    RmsnormProjqWqibAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_projx_wqkvia import (
    RMSNormProjxWqkvia,
    RMSNormProjxWqkviaAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.unproj_o_allreduce import (
    UnProjOAllReduce,
    UnProjOAllReduceAlgorithm,
)


class Mla(SerializableTileRTModule):
    """Implement the MLA operations."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_projx_wqkvia = RMSNormProjxWqkvia(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_5":
            self.rmsnorm_projx_wqkvia.algorithm = RMSNormProjxWqkviaAlgorithm.DECOUPLED
        else:
            self.rmsnorm_projx_wqkvia.algorithm = RMSNormProjxWqkviaAlgorithm.GENERAL
        self.register_op(self.rmsnorm_projx_wqkvia)

        self.layernorm_rope_rotate = LayerNormRoPERotate(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.layernorm_rope_rotate)

        self.rmsnorm_projq_wqib = RmsnormProjqWqib(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_5":
            self.rmsnorm_projq_wqib.algorithm = RmsnormProjqWqibAlgorithm.FP16MMA
        else:
            self.rmsnorm_projq_wqib.algorithm = RmsnormProjqWqibAlgorithm.BF16
        self.register_op(self.rmsnorm_projq_wqib)

        self.projx_wis = ProjxWis(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projx_wis)

        self.projq_wqb = ProjqWqb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projq_wqb)

        self.rmsnorm_kv = KVRMSNorm(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_kv)

        self.projo_wkvb = ProjoWKVb(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.projo_wkvb)

        self.unproj_o_allreduce = UnProjOAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=UnProjOAllReduceAlgorithm.FP8MMA,
        )

        if model_args.arch_name == "glm_5":
            self.unproj_o_allreduce.algorithm = UnProjOAllReduceAlgorithm.FP16MMA

        self.register_op(self.unproj_o_allreduce)

        self.kv_cache: torch.Tensor | None = None
        self.pe_cache: torch.Tensor | None = None
        self.ki_cache: torch.Tensor | None = None

    def get_cache_vars(self) -> list[torch.Tensor]:
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        bs_args = (self.model_args.max_batch_size, cache_seq_len)
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
        if self.ki_cache is None:
            ki_dim = self.model_args.index_head_dim
            self.ki_cache = torch.zeros(
                *bs_args, ki_dim, dtype=torch.bfloat16, device=f"cuda:{self.device_id}"
            )
        return [*super().get_cache_vars(), self.ki_cache, self.kv_cache, self.pe_cache]
