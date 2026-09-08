"""GLM-5.2 / GLM-5.3 profile — thin config over the shared MLA+NSA data plane.

GLM-5.3 is a post-training update of GLM-5.2 (same base model, same config,
same 79-layer plane); ``glm5_3`` and friends alias to this profile.

GLM-5.2 has the same PD data plane as GLM-5 (MLA latent KV + NSA KI index +
1 MTP draft layer) with ONE difference: the DSA indexer is sparsified — only
the "full" layers carry an indexer/KI cache; the "shared" layers reuse the
previous full layer's top-k at runtime and have NO KI cache.

vLLM registers a KI (indexer) cache only on the full layers (HF
``config.indexer_types``, mirrored by vLLM's ``_skip_topk`` in the DeepSeek-V3.2
model code and by the engine's own full-layer rule):

  full layer  <=>  max(L - 2, 0) % 4 == 0   ->  {0,1,2,6,10,...,74}  (21)
  MTP tail (layer 78) is always full         ->  + {78}              (= 22)

The KV and PE planes still cover all 79 layers (every layer does MLA
attention). The 22-vs-79 KI difference is absorbed entirely in
``MlaNsaProfile.classify_layers``: the registered full-layer KI list is
expanded to 79 entries (each shared layer references the previous full
layer's KI cache tensor), so extract / rdma_plan / convert / inject /
buffer_bytes stay layer-uniform and unchanged. ``ki_layer_ids`` below turns on
a strict check that vLLM's registered KI set equals exactly this set (the MTP
tail may be absent when prefill runs without --speculative-config).

Engine selection: on a ROCm torch build (``torch.version.hip``), or when
``TILERT_PD_ENGINE_BACKEND=rocm``, the decode engine is the ROCm adapter in
``glm5_rocm_engine``; otherwise the CUDA engine ``tilert.models.glm_5_2`` is
imported lazily and a clear ``ImportError`` names it when the installed tilert
build does not ship it.
"""

from __future__ import annotations

import os

from tilert.pd_vllm.profiles import base
from tilert.pd_vllm.profiles.glm5_rocm_engine import build_rocm_engine, is_rocm_torch
from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter, MlaNsaProfile

NUM_LAYERS = 79  # 78 main + 1 MTP draft layer (same skeleton as GLM-5)
LAYOUT_VERSION = 12  # glm5_2 wire family (distinct from glm5's 10, dsv32's 11)

# Full (indexer-carrying) layers: dense (0,1,2) + every-4th MoE (6,10,...,74)
# + MTP tail (78). Mirrors the engine's full-layer rule (max(L-2,0)%4==0).
FULL_LAYERS = [L for L in range(78) if max(L - 2, 0) % 4 == 0] + [78]


def _use_rocm_engine() -> bool:
    """Select the ROCm adapter on a HIP torch build or when forced by env; CUDA otherwise."""
    backend = os.environ.get("TILERT_PD_ENGINE_BACKEND", "").strip().lower()
    if backend in ("rocm", "hip"):
        return True
    if backend == "cuda":
        return False
    if backend:
        raise ValueError(f"TILERT_PD_ENGINE_BACKEND={backend!r}; want 'rocm' or 'cuda'")
    return is_rocm_torch()


def _build_cuda_engine(model_weights_dir, max_seq_len, with_mtp):
    try:
        import tilert

        # multi-backend builds load the per-model .so on demand; single-backend
        # builds auto-register on import and lack load_backend.
        if hasattr(tilert, "load_backend"):
            tilert.load_backend("glm5_2")
        from tilert.models.glm_5_2.generator import GLM5_2Generator
        from tilert.models.glm_5_2.model_args import ModelArgsGLM5_2
    except (ImportError, ValueError) as e:
        raise ImportError(
            "the installed tilert build ships no CUDA GLM-5.2 engine "
            "(tilert.models.glm_5_2); use a tilert build that includes it, or "
            "a ROCm tilert build (selected automatically on a HIP torch, or with "
            "TILERT_PD_ENGINE_BACKEND=rocm)"
        ) from e

    gen = GLM5_2Generator(
        model_args=ModelArgsGLM5_2(),
        max_new_tokens=max(max_seq_len - 256, 4096 - 256),
        model_weights_dir=model_weights_dir,
        with_mtp=with_mtp,
        use_topp=True,
        enable_thinking=False,
    )
    gen.from_pretrained()
    return MlaNsaEngineAdapter(gen, with_mtp)


def _build_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps):
    # The ROCm tilert build serves GLM-5.2/5.3 through a different engine API
    # (no inject_cache); same PD data plane, so only the engine adapter differs.
    if _use_rocm_engine():
        return build_rocm_engine(model_weights_dir, max_seq_len, with_mtp, ar_steps)
    return _build_cuda_engine(model_weights_dir, max_seq_len, with_mtp)


base.register(
    MlaNsaProfile(
        name="glm5_2",
        num_layers=NUM_LAYERS,
        layout_version=LAYOUT_VERSION,
        engine_factory=_build_engine,
        ki_layer_ids=FULL_LAYERS,
    )
)
