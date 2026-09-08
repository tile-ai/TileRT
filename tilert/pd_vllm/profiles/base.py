"""ModelProfile seam: everything model-specific in the PD data plane.

The framework (prefill connector plumbing, receive server + control plane,
decode server orchestration, router) is model-agnostic and calls into the
active profile for the parts that differ between models:

  GLM-5 / GLM-5.2 / GLM-5.3 : replicated MLA latent KV + NSA KI index + MTP draft
  DeepSeek-V3.2             : replicated MLA latent KV + NSA KI index + MTP draft

A profile owns four concerns:
  1. receive-buffer layout (decode node) + hello fields advertising it
  2. prefill-side extraction (kv_caches -> staged bytes) + the RDMA plan
  3. decode-side convert (received bytes -> native tensors) + inject
  4. engine construction + decode loop (MTP style differs per model)

Registration returned by ``classify_layers`` and ``sections`` returned by
``extract`` are opaque profile-internal objects threaded back by the
framework — they never cross the profile boundary interpreted.
"""

from __future__ import annotations

from typing import Any, Protocol


class ModelProfile(Protocol):
    name: str
    num_ranks: int
    # TP ranks that actually RDMA-send (the framework counts `done` against
    # this set and skips extraction on other ranks). MLA-family profiles: {0}
    # (the MLA latent is replicated across TP).
    sender_ranks: frozenset

    @property
    def layout_version(self) -> int: ...

    # ── receive side (decode node) ───────────────────────────────────────
    def buffer_bytes(self, max_seq_len: int) -> int:
        """Total receive-buffer size for one request slot."""

    def hello_layout(self, base_ptr: int, max_seq_len: int) -> dict[str, int]:
        """Region base addresses, merged into the hello message.

        Tells the sender where to RDMA-write each section.
        """

    def convert(
        self, buffer: Any, base_ptr: int, max_seq_len: int, received: Any, num_devices: int
    ) -> Any:
        """Received buffer -> native per-device tensors (ConvertedRequest)."""

    # ── prefill side (vLLM connector worker) ─────────────────────────────
    def classify_layers(self, kv_caches: dict, kv_cache_config: Any) -> Any:
        """Inspect registered kv_caches and return an opaque registration.

        The framework passes it back to ``staging_bytes``/``extract``. Raise on
        an unexpected layer set (e.g. missing speculative layer).
        """

    def staging_bytes(self, reg: Any, tp_rank: int, max_seq_len: int) -> int:
        """Per-rank staging-buffer size."""

    def extract(self, reg: Any, req_meta: Any, tp_rank: int, staging, max_seq_len: int) -> Any:
        """Copy this rank's KV out of the paged caches into ``staging``.

        Runs inside the forward window; returns opaque ``sections``.
        """

    def rdma_plan(
        self, hello: dict, sections: Any, tp_rank: int, seq_len: int, staging_base: int
    ) -> tuple[list, list, list]:
        """(src_ptrs, dst_ptrs, lengths) for one mooncake batch write."""

    # ── engine (decode node) ─────────────────────────────────────────────
    def build_engine(
        self, model_weights_dir: str, max_seq_len: int, with_mtp: bool, ar_steps: int
    ) -> Any:
        """Construct the decode engine adapter (inject/decode/reset)."""

    # ── optional hooks (checked with hasattr; not every profile has them) ─
    # configure(kv_cache_dtype)      : MLA family — cache dtype sizes the buffer
    # configure_weights(weights_dir) : members whose depth comes from the
    #                                  checkpoint; called on the decode node
    #                                  BEFORE buffer_bytes/hello_layout.
    # declares_penalties: bool       : whether the decode runtime implements
    #                                  repetition/presence penalties. Read by
    #                                  the router at startup (no engine exists
    #                                  yet to ask); absent means "no".


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
    # GLM-5.3 shares GLM-5.2's base model, config and PD data plane.
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
        raise KeyError(
            f"unknown model profile {name!r}; " f"accepted keys (incl. aliases): {sorted(_ALIASES)}"
        )
    return _REGISTRY[canon]
