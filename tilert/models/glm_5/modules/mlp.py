from tilert.models.base import SerializableTileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.ops.down_allreduce import (
    DownAllReduce,
    DownAllReduceAlgorithm,
)
from tilert.models.glm_5._dsa_v32.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)
from tilert.models.glm_5.modules.mla_v2 import PureMlaV2 as Mla


class Mlp(SerializableTileRTModule):
    """Implement the MLP operations."""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_mlp_up_gate_silu = RMSNormUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.rmsnorm_mlp_up_gate_silu.algorithm = RMSNormUpGateSiLUAlgorithm.BF16MMA_V2
        self.register_op(self.rmsnorm_mlp_up_gate_silu)

        self.rmsnorm_mlp_down = DownAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=DownAllReduceAlgorithm.BF16MMA_V2,
        )
        self.register_op(self.rmsnorm_mlp_down)


class MlpBlock(SerializableTileRTModule):
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
        mlp: "Mlp | None" = None,
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
        self.mlp = (
            mlp
            if mlp is not None
            else Mlp(
                model_args=model_args,
                device_id=device_id,
                num_devices=num_devices,
            )
        )
        self.register_op(self.mlp)
