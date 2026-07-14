"""DeepSeek-V3.2 profile — thin config over the shared MLA+NSA data plane."""

from __future__ import annotations

from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.profiles.mla_nsa import (
    MlaNsaEngineAdapter,
    MlaNsaProfile,
)

NUM_LAYERS = 62  # 61 main + 1 MTP draft (HF: 61 hidden + 1 nextn)
LAYOUT_VERSION = 11  # dsv32 wire family (distinct from glm5's 10)


def _build_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps):
    import tilert

    if hasattr(tilert, "load_backend"):
        tilert.load_backend("deepseek_v3_2")  # multi-backend builds only
    from tilert.models.deepseek_v3_2.generator import DSAv32Generator
    from tilert.models.deepseek_v3_2.model_args import ModelArgs

    gen = DSAv32Generator(
        model_args=ModelArgs(),
        max_new_tokens=max(max_seq_len - 256, 4096 - 256),
        model_weights_dir=model_weights_dir,
        with_mtp=with_mtp,
        use_topp=True,
    )
    gen.from_pretrained()
    return MlaNsaEngineAdapter(gen, with_mtp)


base.register(
    MlaNsaProfile(
        name="dsv32",
        num_layers=NUM_LAYERS,
        layout_version=LAYOUT_VERSION,
        engine_factory=_build_engine,
    )
)
