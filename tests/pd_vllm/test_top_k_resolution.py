"""top_k must reach the engine the way vLLM would resolve it.

The router sends ``top_k`` to *both* backends -- ``build_prefill_body`` copies
the client body to the vLLM prefill instance, and ``_sampling_of`` forwards it
to the decode node. vLLM applies its own rules on the prefill side, so if the
decode side disagreed, a single request would sample its first token under
vLLM's rules and tokens 2..N under ours. That is the same class of bug as the
``max_completion_tokens`` precedence fix.

vLLM 0.25.1's rule (``SamplingParams`` + ``gpu_input_batch.py``)::

    top_k: int = 0          # "Set to 0 (or -1) to consider all tokens."
    if 0 < top_k < vocab_size:  applied
    else:                       top_k = vocab_size   # disabled

``resolve_top_k`` mirrors that shape with the kernel's 256-candidate pool
standing in for vocab_size, so the applied range is [1, 255].

No GPU, no tilert, no vllm: ``sampling`` is pure dict handling.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_top_k_resolution.py -v
"""

from __future__ import annotations

import pytest

from tilert.pd_vllm.sampling import TOP_K_DISABLED, resolve_top_k

# --------------------------------------------------------------------------- #
# the applied range
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 2, 20, 50, 100, 101, 200, 254, 255])
def test_values_the_kernel_can_apply_pass_through(k) -> None:
    """[1, 255] is what the sampler's 256-candidate pool can actually cut to."""
    assert resolve_top_k({"top_k": k}) == k


def test_a_recommended_default_in_the_low_tens_is_applied() -> None:
    """A checkpoint shipping ``top_k: 20`` in generation_config.json must have it
    take effect, not be swallowed as 'disabled'.
    """
    assert resolve_top_k({"top_k": 20}) == 20


# --------------------------------------------------------------------------- #
# vLLM's disable sentinels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [0, -1])
def test_vllm_disable_sentinels(k) -> None:
    """vLLM: "Set to 0 (or -1) to consider all tokens." Both must disable."""
    assert resolve_top_k({"top_k": k}) == TOP_K_DISABLED


def test_absent_is_disabled() -> None:
    assert resolve_top_k({}) == TOP_K_DISABLED


def test_explicit_null_is_disabled() -> None:
    """A client serialising an unset option as null must not crash on int()."""
    assert resolve_top_k({"top_k": None}) == TOP_K_DISABLED


# --------------------------------------------------------------------------- #
# out of range -- must degrade to disabled, never reach the engine unclamped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [256, 257, 1000, 2048, 100000])
def test_at_or_above_the_pool_bound_is_disabled(k) -> None:
    """256 is the kernel's "no rank cut" value; anything above cannot be
    honoured by a 256-candidate pool either. vLLM would apply these, so this is
    a documented divergence -- but it widens the sampled set, never narrows it.
    """
    assert resolve_top_k({"top_k": k}) == TOP_K_DISABLED


@pytest.mark.parametrize("k", [-2, -100])
def test_below_vllms_valid_range_is_disabled_not_forwarded(k) -> None:
    """The vLLM prefill instance 400s these before we are reached; if one ever
    arrives, disable rather than hand a negative to the engine.
    """
    assert resolve_top_k({"top_k": k}) == TOP_K_DISABLED


def test_no_value_ever_escapes_the_engines_accepted_range() -> None:
    """The property that matters: whatever a client sends, the engine sees a value it can accept.

    Guards every call site at once.
    """
    for raw in [None, -100, -2, -1, 0, 1, 20, 100, 255, 256, 999, 10**9]:
        out = resolve_top_k({"top_k": raw})
        assert 1 <= out <= TOP_K_DISABLED, (raw, out)


def test_string_value_is_coerced() -> None:
    assert resolve_top_k({"top_k": "20"}) == 20


def test_other_sampling_keys_are_ignored() -> None:
    """resolve_top_k reads only top_k; top_p et al. are the caller's business."""
    assert resolve_top_k({"top_p": 0.95, "temperature": 0.6}) == TOP_K_DISABLED


# --------------------------------------------------------------------------- #
# every engine adapter goes through it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module", ["tilert.pd_vllm.profiles.mla_nsa"])
def test_adapter_uses_the_shared_resolver(module) -> None:
    """The sampling call sites shared one unclamped expression
    (``int(sampling.get("top_k", 256))``). Pin that none of them reintroduces
    it: the source must reference resolve_top_k and not the old default.

    Source inspection rather than a call, because constructing an adapter needs
    tilert and a GPU.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, module
    src = pathlib.Path(spec.origin).read_text()
    assert "resolve_top_k(sampling)" in src, f"{module} bypasses the resolver"
    assert 'sampling.get("top_k"' not in src, f"{module} reads top_k directly"
