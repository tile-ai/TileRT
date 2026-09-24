"""Base classes for TileRT model modules."""

from abc import ABC
from enum import Enum
from typing import Any, ClassVar

import torch
import torch.nn as nn

from tilert import logger

__all__ = ["SerializableTileRTModule", "TileRTModule", "TilertWeightsConverter"]


class TilertWeightsConverter:
    """Tilert weights converter: dispatches to ``convert_to_<algorithm.value>``."""

    def __init__(self, model_args: Any, num_devices: int):
        self.model_args = model_args
        self.num_devices = num_devices

    def dispatch(self, algorithm: Enum, weights: list[torch.Tensor]) -> Any:
        dispatch_method = getattr(self, f"convert_to_{algorithm.value}")
        return dispatch_method(weights)


class TileRTModule(nn.Module, ABC):
    """Base class for all TileRT modules."""

    _SUPPORTED_ALGORITHMS: ClassVar[dict[str, list[Enum]]] = {}

    @classmethod
    def get_supported_algorithms(cls, arch_name: str) -> list[Enum]:
        if arch_name not in cls._SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"{cls.__name__} does not support arch '{arch_name}'. Supported: {list(cls._SUPPORTED_ALGORITHMS.keys())}"
            )
        return cls._SUPPORTED_ALGORITHMS[arch_name]

    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
        model_args: Any | None = None,
        num_devices: int = 1,
        device_id: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.num_devices = num_devices
        self.device_id = device_id
        self.algorithm: Enum | None = None
        self.layer_idx = layer_idx
        self.is_tilert_weights_init = False
        self.op_name = type(self).__name__ if op_name == "" else op_name

    def get_cache_vars(self) -> list[torch.Tensor]:
        return []

    def get_tilert_weights_alias(self) -> list[str]:
        return list(self.tilert_weights_alias())

    def set_algorithm(self, algorithm: Enum) -> None:
        if self._SUPPORTED_ALGORITHMS:
            arch = self.model_args.arch_name
            supported = self.get_supported_algorithms(arch)
            if algorithm not in supported:
                raise ValueError(
                    f"{type(self).__name__}: algorithm {algorithm} not supported for arch '{arch}'. Supported: {supported}"
                )
        self.algorithm = algorithm

    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("golden_forward is not available in release builds")

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError("Tilert forward not implemented")

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len


class SerializableTileRTModule(TileRTModule):
    """Composite module: ordered sequence of sub-ops with prefixed weight keys."""

    def __init__(
        self,
        model_args: Any,
        device_id: int = 0,
        num_devices: int = 1,
        remove_selected: bool = False,
    ):
        super().__init__(
            type(self).__name__, model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.remove_selected = remove_selected
        self.exec_seq: list[TileRTModule] = []
        self.prefix_seq: list[str] = []
        self.suffix_seq: list[str] = []
        self.retain_weights_seq: list[bool] = []

    def get_cache_vars(self) -> list[torch.Tensor]:
        cache_vars = []
        for op in self.exec_seq:
            cache_vars.extend(op.get_cache_vars())
        return cache_vars

    def register_op(
        self, op: TileRTModule, prefix: str = "", suffix: str = "", retain_weights: bool = False
    ) -> None:
        self.exec_seq.append(op)
        self.prefix_seq.append(prefix)
        self.suffix_seq.append(suffix)
        self.retain_weights_seq.append(retain_weights)

    def get_tilert_weights_alias(self) -> list[str]:
        weights_alias: list[str] = []
        for op in self.exec_seq:
            weights_alias.extend(op.get_tilert_weights_alias())
        return weights_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        weights = []
        for op in self.exec_seq:
            weights.extend(op.get_weights_list())
        return weights

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        for op, prefix, suffix, retain_weights in zip(
            self.exec_seq, self.prefix_seq, self.suffix_seq, self.retain_weights_seq
        ):
            if op.is_tilert_weights_init:
                logger.debug(f"Skipping init_tilert_weights for {op.op_name} (already initialized)")
                continue
            keys_to_remove = set()
            op_state_dict = {}
            for op_key in op.get_tilert_weights_alias():
                original_key = f"{prefix}{op_key}{suffix}"
                if original_key in state_dict:
                    op_state_dict[op_key] = state_dict[original_key]
                    if self.remove_selected:
                        keys_to_remove.add(original_key)
            op.init_tilert_weights(op_state_dict)
            if self.remove_selected and (not retain_weights):
                for k in keys_to_remove:
                    del state_dict[k]

    def init_random_weights(self) -> None:
        raise RuntimeError("init_random_weights is not available in release builds")

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        for op in self.exec_seq:
            op.init_tilert_vars(batch_size, seq_len)
