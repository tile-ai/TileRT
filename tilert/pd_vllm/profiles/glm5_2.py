from __future__ import annotations

from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.profiles.glm5_rocm_engine import build_rocm_engine
from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter, MlaNsaProfile

NUM_LAYERS = 79
LAYOUT_VERSION = 12
FULL_LAYERS = [L for L in range(78) if max(L - 2, 0) % 4 == 0] + [78]


def _build_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps):
    return build_rocm_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps)


base.register(
    MlaNsaProfile(
        name="glm5_2",
        num_layers=NUM_LAYERS,
        layout_version=LAYOUT_VERSION,
        engine_factory=_build_engine,
        ki_layer_ids=FULL_LAYERS,
    )
)
