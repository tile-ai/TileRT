from __future__ import annotations

import os

from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter, MlaNsaProfile

_NO_MTP = (os.environ.get("TILERT_PD_NO_MTP") or "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
NUM_LAYERS = 78 if _NO_MTP else 79
LAYOUT_VERSION = 1010 if _NO_MTP else 10


def _build_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps):
    import tilert

    if hasattr(tilert, "load_backend"):
        tilert.load_backend("glm5")
    from tilert.models.glm_5.generator import GLM5Generator
    from tilert.models.glm_5.model_args import ModelArgsGLM5

    gen = GLM5Generator(
        model_args=ModelArgsGLM5(),
        max_new_tokens=max(max_seq_len - 256, 4096 - 256),
        model_weights_dir=model_weights_dir,
        with_mtp=with_mtp,
        use_topp=True,
        enable_thinking=False,
    )
    gen.from_pretrained()
    return MlaNsaEngineAdapter(gen, with_mtp)


base.register(
    MlaNsaProfile(
        name="glm5",
        num_layers=NUM_LAYERS,
        layout_version=LAYOUT_VERSION,
        engine_factory=_build_engine,
    )
)
