from tilert.models.base import SerializableTileRTModule
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.models.deepseek_v3_2.modules.mla import Mla
from tilert.models.deepseek_v3_2.ops.down_allreduce import DownAllReduce
from tilert.models.deepseek_v3_2.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)


class Mlp(SerializableTileRTModule):
    """Implement the MLP operations."""

    def __init__(self, model_args: ModelArgs, device_id: int, num_devices: int):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_mlp_up_gate_silu = RMSNormUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        if model_args.arch_name == "glm_5":
            self.rmsnorm_mlp_up_gate_silu.algorithm = RMSNormUpGateSiLUAlgorithm.FP16MMA
        self.register_op(self.rmsnorm_mlp_up_gate_silu)
        self.rmsnorm_mlp_down = DownAllReduce(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.rmsnorm_mlp_down)


class MlpBlock(SerializableTileRTModule):
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
        self.mlp = Mlp(model_args=model_args, device_id=device_id, num_devices=num_devices)
        self.register_op(self.mlp)
