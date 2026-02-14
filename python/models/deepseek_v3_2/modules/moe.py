from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.modules.mla import Mla
from tilert.models.deepseek_v3_2.ops.expert_down_allreduce import ExpertDownAllReduce
from tilert.models.deepseek_v3_2.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.deepseek_v3_2.ops.rmsnorm_expert_proj import RMSNormExpertProj


class Moe(SerializableTileRTModule):
    """Implement the MOE operations."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_expert_proj = RMSNormExpertProj(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_expert_proj)

        self.exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        if model_args.arch_name == "glm_5":
            self.exp_sel_up_gate_silu.algorithm = ExpertSelectUpGateSiLUAlgorithm.FP16MMA
        self.register_op(self.exp_sel_up_gate_silu)
        self.expert_down_allreduce = ExpertDownAllReduce(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.expert_down_allreduce)


class MoeBlock(SerializableTileRTModule):
    """Implement the MOE block operations."""

    def __init__(
        self, model_args: ModelArgs, device_id: int, num_devices: int, remove_selected: bool = False
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        self.mla = Mla(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.mla)
        self.moe = Moe(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.moe)
