import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.ops.expert_down_allreduce import (
    ExpertDownAllReduce,
    ExpertDownAllReduceAlgorithm,
)
from tilert.models.glm_5._dsa_v32.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.glm_5._dsa_v32.ops.rmsnorm_expert_proj import (
    RMSNormExpertProj,
)
from tilert.models.glm_5.modules.mla_v2 import PureMlaV2 as Mla


class Moe(SerializableTileRTModule):
    """Implement the MOE operations."""

    rmsnorm_expert_proj: RMSNormExpertProj

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_expert_proj = RMSNormExpertProj(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_expert_proj)

        self.exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=ExpertSelectUpGateSiLUAlgorithm.BF16MMA,
        )
        self.register_op(self.exp_sel_up_gate_silu)

        _expert_down_algo = ExpertDownAllReduceAlgorithm.BF16MMA
        self.expert_down_allreduce = ExpertDownAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=_expert_down_algo,
        )
        self.register_op(self.expert_down_allreduce)

    def get_weights_list(self) -> list[torch.Tensor]:
        return super().get_weights_list()


class MoeBlock(SerializableTileRTModule):
    """Implement the MOE block operations."""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        remove_selected: bool = False,
        mla_cls: type | None = None,
        mla_num_devices: int | None = None,
        mla_kwargs: dict | None = None,
        moe: "Moe | None" = None,
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        mla_class = mla_cls or Mla
        mla_nd = mla_num_devices if mla_num_devices is not None else num_devices
        self.mla = mla_class(
            model_args=model_args, device_id=device_id, num_devices=mla_nd, **(mla_kwargs or {})
        )
        self.register_op(self.mla)
        self.moe = (
            moe
            if moe is not None
            else Moe(model_args=model_args, device_id=device_id, num_devices=num_devices)
        )
        self.register_op(self.moe)
