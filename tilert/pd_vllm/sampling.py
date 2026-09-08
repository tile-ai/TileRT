"""Request sampling params -> engine sampling params, for the decode node.

The engine adapters each take the router's forwarded `sampling` dict and call
their generator's ``update_sampling_params``. The dict comes straight from the
client body (``pd_router._sampling_of``), so the translation from "what an
OpenAI/vLLM client may send" to "what the engine accepts" belongs here, once,
rather than three times across the profiles.
"""

from __future__ import annotations

__all__ = [
    "GREEDY_LOGPROBS_TOP_P",
    "TOP_K_DISABLED",
    "VLLM_DEFAULT_TOP_P",
    "resolve_top_k",
    "resolve_top_p",
]

# Two unrelated top_p values live here; keep them apart.
#
# ``VLLM_DEFAULT_TOP_P`` / ``resolve_top_p`` are the CLIENT-FACING nucleus: what
# the request asked for, or the deployment's default when it asked for nothing.
# ``GREEDY_LOGPROBS_TOP_P`` is an INTERNAL mechanism that makes the top-p kernel
# behave as argmax; it is not a nucleus a client can ask for and never passes
# through ``resolve_top_p``.

# top_p for a greedy request that also wants log probabilities.
#
# The engine has no greedy log-probability export: greedy takes a separate top-1
# kernel whose logits buffer the MTP draft head overwrites later in the tape. So
# a greedy logprobs request is decoded on the TOP-P path instead, where the
# export lives inside the sampling op — and this cutoff is what makes that path
# behave exactly like argmax.
#
# The kernel picks its nucleus as the first index where the cumulative
# probability exceeds top_p (top_p.cuh, "Find cutoff point"). cum_probs[0] is the
# largest full-vocab probability and so is at least 1/vocab (~4e-6 at 248320),
# which always exceeds this value: the cutoff is 0, and the multinomial then
# iterates a single candidate. Under MTP verify the accept probability becomes
# topks[0]/cum_probs[0] == 1, so an argmax draft is always accepted and any other
# draft is rejected back to the argmax -- greedy speculative decoding, at full
# MTP speed.
#
# Two orders of margin below 1/vocab is deliberate; do not raise it toward
# realistic top_p values, or the cutoff stops being 0 and greedy silently becomes
# sampling. The paired temperature is 1.0, NOT 0 -- see _greedy_logprobs_params.
GREEDY_LOGPROBS_TOP_P = 1e-9

# vLLM's framework default, from ``_DEFAULT_SAMPLING_PARAMS`` in
# ``vllm/entrypoints/openai/chat_completion/protocol.py``. The protocol field
# itself defaults to None; ``to_sampling_params`` then resolves
#
#     client explicit value  >  default_sampling_params (the deployed model's
#     generation_config.json)  >  this constant
#
# so this is the LAST link in vLLM's own chain, not the whole chain.
VLLM_DEFAULT_TOP_P = 1.0


def resolve_top_p(sampling: dict, default: float = VLLM_DEFAULT_TOP_P) -> float:
    """The ``top_p`` for this request, resolved once for both PD legs.

    Exists because the two legs used to disagree. The decode adapters each read
    ``sampling.get("top_p", 0.95)`` while the vLLM prefill instance resolved its
    own default through the chain above -- so a client that sent ``temperature``
    but no ``top_p`` had token 1 sampled under one nucleus and tokens 2..N under
    another, with nothing in the response to show it.

    The fix is not a better constant: no constant can be right, because the
    middle link is a property of the deployed checkpoint. It is resolving the
    value in ONE place (the router) and sending it explicitly to BOTH legs, so
    they agree by construction whatever the value is. ``default`` is the
    deployment's stand-in for that middle link (``--default-top-p``); wiring
    ``generation_config.json`` in behind it changes only what is passed here.

    Adapters must not re-default: they receive an already-resolved number. A
    greedy logprobs request does not come through here at all -- its top_p is
    ``GREEDY_LOGPROBS_TOP_P``, chosen to defeat the nucleus rather than express
    one.
    """
    raw = sampling.get("top_p")
    if raw is None:
        return float(default)
    if isinstance(raw, bool):
        raise ValueError("top_p must be a number, got bool")
    return float(raw)


# The sampler materialises kTopK=256 candidates per GPU (all-gathered to
# kNumGpus*256), and its cutoff treats any top_k at or above that per-GPU bound
# as "no rank cut":
#
#   effective_top_k = (top_k > 0 && top_k < kTopK) ? top_k : kTotalTopKs
#       -- the engine's top_p sampler kernel
#
# So 256 is the value that disables the rank cut, and [1, 255] is the range the
# kernel can actually apply. These two constants mirror that kernel's kTopK and
# must be kept in step with it: if kTopK ever changes, TOP_K_DISABLED stops
# meaning "disabled" and silently becomes a real rank cut.
_KERNEL_TOP_K_POOL = 256
TOP_K_DISABLED = _KERNEL_TOP_K_POOL
_TOP_K_APPLIED_MAX = _KERNEL_TOP_K_POOL - 1


def resolve_top_k(sampling: dict) -> int:
    """The engine's top_k for this request, following vLLM's convention.

    vLLM is the reference because the router sends ``top_k`` to the vLLM
    prefill instance as well as here (``pd_router.build_prefill_body`` copies the
    client body). If the two disagreed, one request would sample its first token
    under vLLM's rules and the rest under ours.

    vLLM (``SamplingParams``, ``gpu_input_batch.py``)::

        top_k: int = 0          # "Set to 0 (or -1) to consider all tokens."
        if 0 < top_k < vocab_size:  applied
        else:                       top_k = vocab_size   # i.e. disabled

    We follow that shape, with the kernel's candidate pool standing in for
    vocab_size:

    ==================  ==========================================
    request top_k       result
    ==================  ==========================================
    absent / null       disabled
    0, -1               disabled -- vLLM's documented sentinels
    [1, 255]            applied
    >= 256              disabled (pool bound; vLLM would apply it)
    < -1                disabled here; the vLLM prefill instance
                        400s it first, so the client sees an error
    ==================  ==========================================

    The last two rows are the only divergences from vLLM and both are one-way
    (we disable where vLLM would cut), so they can widen the sampled set but
    never narrow it unexpectedly.

    The 255 bound is the kernel's, not a policy choice: measured on 8xB200,
    ``top_k`` of 300 and 2048 sample bit-identically to 256 over 200 seeds, so
    the sampler already treats everything at or above 256 as no cut.

    Known kernel behaviour an applied value inherits: the cutoff is inclusive,
    so ``top_k=N`` keeps N+1 candidates (measured: 1 -> logit ranks {0,1},
    4 -> {0..4}). ``top_k=1`` therefore samples between the top two tokens
    rather than being strict argmax. Greedy requests do not go through here --
    ``temperature ~ 0`` selects a separate captured graph.

    Note ``>= 256`` is *not* what every client expects: some vendor APIs
    document "a value greater than 100 indicates that the top_k strategy is not
    enabled". A generation_config ``top_k`` in the low tens is well inside the
    applied range, so such a default is honoured either way; only explicit
    values in [101, 255] differ, and there we follow vLLM.
    """
    raw = sampling.get("top_k")
    if raw is None:
        return TOP_K_DISABLED
    k = int(raw)
    if k < 1 or k > _TOP_K_APPLIED_MAX:
        return TOP_K_DISABLED
    return k
