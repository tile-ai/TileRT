"""GLM-5 profile — thin config over the shared MLA+NSA data plane.

GLM-5 = 79 cache layers (78 main + 1 MTP draft), MLA latent KV + NSA KI index.
All layout / convert / extract / RDMA logic lives in ``mla_nsa``; this file
only pins the layer count, wire version, and the GLM5Generator engine build.
"""

from __future__ import annotations

from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.profiles.mla_nsa import (
    MlaNsaEngineAdapter,
    MlaNsaProfile,
)

NUM_LAYERS = 79  # 78 main + 1 MTP draft
LAYOUT_VERSION = 10  # glm5 wire family


def _build_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps):
    import tilert

    # multi-backend builds (tilert>=0.1.x) load the per-model .so on demand;
    # single-backend builds auto-register on import and lack load_backend.
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
