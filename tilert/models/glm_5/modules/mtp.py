import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.ops import RMSNormHeadProj
from tilert.models.glm_5.modules.moe import MoeBlock
from tilert.models.glm_5.modules.mtp_preprocess import MTPPreprocessLayer


class MTP(SerializableTileRTModule):
    """MTP module."""

    def __init__(
        self,
        model_args: ModelArgs,
        device_id: int,
        num_devices: int,
        mla_cls: type | None = None,
        mla_num_devices: int | None = None,
        mla_kwargs: dict | None = None,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.embed_tokens_weight = None
        self.freqs_cis = None

        mtp_layer_id = self.model_args.n_layers
        self.register_op(
            MTPPreprocessLayer(self.model_args, self.num_devices, device_id),
            prefix=f"layer_{mtp_layer_id}_",
            suffix=f"_dev_{device_id}",
        )
        self.register_op(
            MoeBlock(
                model_args=model_args,
                device_id=device_id,
                num_devices=num_devices,
                mla_cls=mla_cls,
                mla_num_devices=mla_num_devices,
                mla_kwargs=mla_kwargs,
            ),
            prefix=f"layer_{mtp_layer_id}_",
            suffix=f"_dev_{device_id}",
        )
        self.register_op(
            RMSNormHeadProj(model_args=model_args, device_id=device_id, num_devices=num_devices),
            prefix=f"layer_{mtp_layer_id}_",
            suffix=f"_dev_{device_id}",
            retain_weights=True,
        )

    def init_tilert_weights(self, state_dicts: dict[str, torch.Tensor]) -> None:
        self.embed_tokens_weight = state_dicts["model.embed_tokens.weight"]
        self.freqs_cis = state_dicts["freqs_cis"]
        super().init_tilert_weights(state_dicts)

    def get_weights_list(self) -> list[torch.Tensor]:
        return [
            self.embed_tokens_weight,
            self.freqs_cis,
            *super().get_weights_list(),
        ]
