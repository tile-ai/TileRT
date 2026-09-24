from __future__ import annotations

from typing import Any, Protocol


class ModelProfile(Protocol):
    name: str
    num_ranks: int
    layout_version: int
    sender_ranks: frozenset

    def buffer_bytes(self, max_seq_len: int) -> int:
        pass

    def hello_layout(self, base_ptr, max_seq_len: int) -> dict:
        pass

    def convert(
        self, buffer: Any, base_ptr: int, max_seq_len: int, received: Any, num_devices: int
    ) -> Any:
        pass

    def classify_layers(self, kv_caches: dict, kv_cache_config: Any) -> Any:
        pass

    def staging_bytes(self, reg: Any, tp_rank: int, max_seq_len: int, nshards: int = 1) -> int:
        pass

    def extract(self, reg: Any, req_meta: Any, tp_rank: int, staging, max_seq_len: int) -> Any:
        pass

    def rdma_plan(
        self, hello: dict, sections: Any, tp_rank: int, seq_len: int, staging_base
    ) -> tuple[list, list, list]:
        pass

    def build_engine(
        self, model_weights_dir: str, max_seq_len: int, with_mtp: bool, ar_steps: int, num_mtp: int
    ) -> Any:
        pass


_REGISTRY: dict[str, ModelProfile] = {}
_ALIASES = {
    "glm5": "glm5",
    "glm_5": "glm5",
    "glm-5": "glm5",
    "glm5_2": "glm5_2",
    "glm_5_2": "glm5_2",
    "glm-5.2": "glm5_2",
    "glm5.2": "glm5_2",
    "glm52": "glm5_2",
    "glm5_3": "glm5_2",
    "glm_5_3": "glm5_2",
    "glm-5.3": "glm5_2",
    "glm5.3": "glm5_2",
    "glm53": "glm5_2",
    "dsv32": "dsv32",
    "deepseek_v3_2": "dsv32",
    "deepseek-v3.2": "dsv32",
    "dsv3.2": "dsv32",
    "v32": "dsv32",
}


def register(profile: ModelProfile) -> None:
    _REGISTRY[profile.name] = profile


def get_profile(name: str) -> ModelProfile:
    canon = _ALIASES.get(name, name)
    if canon not in _REGISTRY:
        # lazy import so a profile's heavy deps load only when selected
        if canon == "glm5":
            from tilert.pd_vllm.profiles import glm5  # noqa: F401
        elif canon == "glm5_2":
            from tilert.pd_vllm.profiles import glm5_2  # noqa: F401
        elif canon == "dsv32":
            from tilert.pd_vllm.profiles import dsv32  # noqa: F401
    if canon not in _REGISTRY:
        raise KeyError(f"unknown model profile {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[canon]


DEFAULT_SUPPORTED_NUM_MTP = (3,)


def resolve_num_mtp(profile: ModelProfile, requested: int, *, with_mtp: bool) -> int:
    supported = tuple(getattr(profile, "supported_num_mtp", DEFAULT_SUPPORTED_NUM_MTP))
    if requested not in supported:
        raise ValueError(
            f"profile {profile.name!r} cannot serve num_mtp={requested}: it supports {list(supported)}."
        )
    if requested != 3 and (not with_mtp):
        raise ValueError(
            f"num_mtp={requested} needs speculative decoding: pass --with-mtp, or drop --num-mtp to run the non-MTP path."
        )
    return requested
