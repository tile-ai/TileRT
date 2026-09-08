"""The generation parameters the PD stack can execute, and the gate refusing the rest.

The PD split computes one request in two places: the vLLM prefill instance
samples token 1, the TileRT decode node samples tokens 2..N. The client body is
forwarded to vLLM almost verbatim (``pd_router.build_prefill_body`` only
overrides ``max_tokens`` / ``stream`` / ``logprobs`` / ``kv_transfer_params``),
while the decode node receives only the keys ``pd_router._sampling_of`` selects.

Any field in the gap between those two sets is applied to token 1 and silently
dropped for the rest of the reply. That is the failure mode this module exists to
prevent: a client that asked for ``stop`` or ``seed`` gets a 200 whose content
violates what it asked for, and has no way to detect it. Refusing the request is
strictly better -- the caller can drop the field, lower its expectations, or
route to a native vLLM endpoint.

Two tiers, because two different things are being decided:

``_STATIC_FIELDS``
    Semantics no TileRT decode runtime implements at all. Refused unconditionally
    when non-neutral. No capability probe can change the answer, so this tier is
    always enforced and cannot be wrong.

``_PROFILE_FIELDS``
    Semantics some profiles honour and others do not (penalties need a pre-pass
    the MLA/NSA runtimes do not carry). Refused only when the serving
    node's declared capabilities say it cannot execute them -- see
    :class:`NodeCapabilities` and ``decode_server``'s ``/capabilities``.

"Neutral" means "the value vLLM would have used had the field been absent", read
off ``ChatCompletionRequest`` and ``_DEFAULT_SAMPLING_PARAMS`` in
``vllm/entrypoints/openai/chat_completion/protocol.py``. A neutral value is
accepted, because honouring it and ignoring it are the same computation -- a
client sending ``frequency_penalty: 0`` is not asking for anything. This is what
makes the gate deployable: it refuses requests whose *behaviour* would differ,
not requests that merely mention a field.

Fields whose semantics belong entirely to the prefill stage
(``truncate_prompt_tokens``, ``echo``, the chat-template knobs, ``tools``) are
deliberately absent: vLLM applies them where they are meant to apply, so there
is no gap to close.

So are fields the ROUTER executes itself, over the text it detokenises rather
than by sampling -- ``stop`` and ``include_stop_str_in_output``. The decode
loop's lack of them is real and no longer decides anything: an entry here means
"refuse when non-neutral", so keeping one would refuse a feature the stack
serves. They are validated where the tokenizer that makes them executable
lives. Absence is not silence: ``tests/pd_vllm/test_no_silent_degradation.py``
requires every field the router reads to be declared in ``STATIC_FIELD_NAMES``
or in its own table, so a field can move between the two but cannot fall out of
both.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CapabilityError",
    "CapabilityUnavailable",
    "InvalidParameter",
    "NodeCapabilities",
    "PROFILE_FIELD_NAMES",
    "STATIC_FIELD_NAMES",
    "TYPED_FIELD_NAMES",
    "engine_capabilities",
    "validate_generation_request",
]


# --------------------------------------------------------------------------- #
# Error hierarchy (same envelope shape as GrammarError / LogprobsUnsupported)
# --------------------------------------------------------------------------- #
class CapabilityError(Exception):
    """Base for request-gate failures. Never degrade silently."""

    error_type = "capability_error"
    http_status = 500

    def to_payload(self) -> dict[str, str]:
        return {"error": str(self), "error_type": self.error_type}


class InvalidParameter(CapabilityError):
    """The field's type or value is not usable -> HTTP 400.

    The client's mistake, not a missing feature: no version of this stack would
    accept it. Mirrors ``InvalidGrammarError`` / ``LogprobsUnsupported``.
    """

    error_type = "invalid_parameter"
    http_status = 400


class CapabilityUnavailable(CapabilityError):
    """A valid OpenAI/vLLM field this decode stage cannot execute -> HTTP 501.

    Not a client error: the request is well-formed and a native vLLM endpoint
    would serve it. 501 is the same status the stack already uses for
    "the engine cannot produce what was asked for" (``logprobs_unavailable``).
    """

    error_type = "capability_unavailable"
    http_status = 501


# --------------------------------------------------------------------------- #
# Per-node capability declaration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NodeCapabilities:
    """What one decode node's profile + live engine can execute.

    Defaults are all-False so that every path which cannot obtain a real answer
    -- an unreachable node, a malformed payload, a node predating
    ``/capabilities`` -- fails closed rather than assuming support.
    """

    penalties: bool = False
    ignore_eos: bool = False

    def intersect(self, other: NodeCapabilities) -> NodeCapabilities:
        """The capabilities a request can rely on across a whole pool.

        The router picks a node only after validation, so a field may be
        accepted only if EVERY node could have executed it.
        """
        return NodeCapabilities(
            penalties=self.penalties and other.penalties,
            ignore_eos=self.ignore_eos and other.ignore_eos,
        )

    def to_payload(self) -> dict[str, bool]:
        return {"penalties": self.penalties, "ignore_eos": self.ignore_eos}

    @classmethod
    def from_payload(cls, payload: object) -> NodeCapabilities:
        """Parse a ``/capabilities`` response, treating anything unrecognised as unsupported.

        A node that omits a key does not declare it.
        """
        if not isinstance(payload, dict):
            return cls()
        caps = payload.get("capabilities", payload)
        if not isinstance(caps, dict):
            return cls()
        return cls(
            penalties=caps.get("penalties") is True,
            ignore_eos=caps.get("ignore_eos") is True,
        )


def engine_capabilities(engine: object) -> NodeCapabilities:
    """What the engine actually running on this node can execute.

    Read from the live engine rather than the profile because the adapters
    DEMOTE their own claims after probing the installed ``tilert`` build: a
    from-source serve paired with an older engine wheel exposes the same
    sampling entry points while ignoring penalties, and an adapter that probes
    for the pre-pass turns that into ``supports_penalties() == False``.
    Reporting the profile's static claim here
    would hand the router a promise the engine has already withdrawn.

    Each capability is an optional predicate, matching the ``supports_logprobs``
    convention: absent means unsupported, and a predicate that raises is treated
    the same way rather than failing the endpoint.
    """

    def _ask(name: str) -> bool:
        probe = getattr(engine, name, None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            return False

    return NodeCapabilities(
        penalties=_ask("supports_penalties"),
        ignore_eos=_ask("supports_ignore_eos"),
    )


# --------------------------------------------------------------------------- #
# Field tables
# --------------------------------------------------------------------------- #
# Kinds describe how the neutral value is recognised, not the JSON type:
#   "empty"  -> neutral when falsy (absent, null, "", [], {})
#   "unset"  -> neutral ONLY when absent or null; every other value asks for
#               something. For a field whose zero is a real request -- `seed: 0`
#               is a valid seed, `prompt_logprobs: 0` asks for the prompt
#               tokens' own log probabilities -- "falsy" and "asks for nothing"
#               are different questions, and answering the first would let the
#               request through to be applied on the prefill leg only.
#   "number" -> neutral when numerically equal to `neutral`
#   "flag"   -> neutral when the boolean equals `neutral`
#   "count"  -> neutral when the integer equals `neutral`; below `minimum` is 400
_EMPTY, _UNSET, _NUMBER, _FLAG, _COUNT = ("empty", "unset", "number", "flag", "count")

# (kind, neutral, minimum, why-it-cannot-be-honoured)
_STATIC_FIELDS: dict[str, tuple] = {
    # ── stop conditions ────────────────────────────────────────────────────
    #    `stop` and `include_stop_str_in_output` are absent from this table on
    #    purpose: text-level stop matching happens in the router, over the text
    #    it detokenises, so the decode loop's lack of it does not matter. They
    #    are validated by the router instead, which is where the tokenizer that
    #    makes them executable lives.
    "stop_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode loop uses the model's own stop set and " "accepts no per-request ids",
    ),
    "min_tokens": (
        _COUNT,
        0,
        0,
        "the decode loop cannot suppress its stop set for a " "minimum length",
    ),
    # ── sampling knobs with no decode-side implementation ──────────────────
    "frequency_penalty": (
        _NUMBER,
        0.0,
        None,
        "the decode sampler implements repetition and " "presence penalties only",
    ),
    "min_p": (_NUMBER, 0.0, None, "the decode sampler implements top-p and top-k only"),
    "seed": (
        _UNSET,
        None,
        None,
        "the decode sampler's seed is per-process, so a per-request seed "
        "cannot make the reply reproducible",
    ),
    "logit_bias": (_EMPTY, None, None, "the decode sampler has no per-request logit bias"),
    "bad_words": (_EMPTY, None, None, "the decode loop has no bad-words matcher"),
    "allowed_token_ids": (_EMPTY, None, None, "the decode sampler has no per-request allow list"),
    # ── constraints: response_format / regex / ebnf ARE translated (see
    #    grammar_spec); vLLM 0.24's structured_outputs entry point is not ────
    "structured_outputs": (
        _EMPTY,
        None,
        None,
        "use response_format, which this stack translates " "into a decode-side grammar",
    ),
    # ── multiplicity: one KV state is transferred, so one sequence ──────────
    "n": (
        _COUNT,
        1,
        1,
        "one prefill KV state is transferred per request, so the decode "
        "node produces exactly one sequence",
    ),
    "best_of": (
        _COUNT,
        1,
        1,
        "the decode node produces exactly one sequence, so there is " "nothing to select from",
    ),
    "use_beam_search": (_FLAG, False, None, "the decode node runs single-sequence AR/MTP decode"),
    # ── response shaping the router does not perform ───────────────────────
    "prompt_logprobs": (
        _UNSET,
        None,
        None,
        "the prefill instance is asked for one token, so no " "prompt distribution is collected",
    ),
    "logprob_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode export returns the top candidates, not " "a caller-chosen vocab subset",
    ),
    "skip_special_tokens": (
        _FLAG,
        True,
        None,
        "the reply is detokenised for the output parser, " "which always consumes special tokens",
    ),
}

# (kind, neutral, capability attribute, why-it-may-be-unavailable)
_PROFILE_FIELDS: dict[str, tuple] = {
    "repetition_penalty": (
        _NUMBER,
        1.0,
        "penalties",
        "this model's decode runtime has no penalty " "pre-pass",
    ),
    "presence_penalty": (
        _NUMBER,
        0.0,
        "penalties",
        "this model's decode runtime has no penalty " "pre-pass",
    ),
    "ignore_eos": (
        _FLAG,
        False,
        "ignore_eos",
        "this model's decode loop does not clear its stop set",
    ),
}

# Always executable, so never refused -- but still checked, because the router
# now RESOLVES these and writes the result into the prefill request. That
# overwrites whatever the client sent, which takes away vLLM's own chance to
# reject a bad value: `top_k: 1.9` would be truncated to 1 and served by both
# legs as a materially different request. (kind, minimum)
_TYPED_FIELDS: dict[str, tuple] = {
    "temperature": (_NUMBER, 0.0),
    "top_p": (_NUMBER, 0.0),
    "top_k": (_COUNT, None),
}

STATIC_FIELD_NAMES = tuple(_STATIC_FIELDS)
PROFILE_FIELD_NAMES = tuple(_PROFILE_FIELDS)
TYPED_FIELD_NAMES = tuple(_TYPED_FIELDS)


# --------------------------------------------------------------------------- #
# Neutrality tests
# --------------------------------------------------------------------------- #
def _as_number(field: str, value: object) -> float:
    # bool is an int subclass in Python; `top_p: true` is a type error, not 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameter(f"{field} must be a number, got {type(value).__name__}")
    return float(value)


def _as_flag(field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidParameter(f"{field} must be a boolean, got {type(value).__name__}")
    return value


def _as_count(field: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameter(f"{field} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise InvalidParameter(f"{field} must be >= {minimum}, got {value}")
    return value


def _lenient_float(field: str, raw: object) -> float:
    """A number, accepting the string form vLLM's own validation coerces.

    Deliberately as permissive as vLLM: rejecting ``temperature: "0.7"`` here
    would turn a request vLLM serves into a 400. Only bools and non-numerics are
    refused.
    """
    if isinstance(raw, bool):
        raise InvalidParameter(f"{field} must be a number, got bool")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InvalidParameter(f"{field} must be a number, got {raw!r}") from None


def _lenient_int(field: str, raw: object) -> int:
    """An integer, permissive about representation and strict about integrality.

    ``"20"`` and ``20.0`` coerce losslessly and vLLM accepts both. ``1.9`` does
    not: truncating it to 1 would have the router write a materially different
    request into BOTH legs and report success, where before this resolution the
    value reached vLLM and was rejected there.
    """
    value = _lenient_float(field, raw)
    if value != int(value):
        raise InvalidParameter(f"{field} must be an integer, got {raw!r}")
    return int(value)


def _is_neutral(field: str, value: object, kind: str, neutral, minimum: int | None) -> bool:
    """Whether ``value`` asks for anything beyond vLLM's own default.

    ``None`` is always neutral: an explicit null is how SDKs spell "unset", and
    vLLM resolves it to the same default as an absent key.
    """
    if value is None:
        return True
    if kind == _EMPTY:
        return not value
    if kind == _UNSET:
        return False  # None already returned True above
    if kind == _NUMBER:
        return _as_number(field, value) == neutral
    if kind == _FLAG:
        return _as_flag(field, value) is neutral
    # _COUNT
    return _as_count(field, value, minimum if minimum is not None else 0) == neutral


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def validate_generation_request(
    body: dict,
    capabilities: NodeCapabilities | None = None,
    adopted: dict | None = None,
) -> None:
    """Refuse a request whose generation parameters the decode stage would drop.

    Call BEFORE anything observable happens -- before the vLLM prefill request,
    before a decode node is acquired, before the connector claims anything --
    so a refused request costs nothing and holds nothing.

    ``capabilities`` is what every node in the pool can execute (the
    intersection; see :meth:`NodeCapabilities.intersect`). ``None`` means the
    router could not establish it, which is treated as no support: a field whose
    execution cannot be confirmed must not be accepted.

    ``adopted`` carries the deployment's own defaults for the profile-dependent
    fields (``generation_defaults``). They are checked exactly like a
    client-supplied value, because they end up on the wire the same way: a
    ``repetition_penalty`` taken from the model's ``generation_config.json`` is
    still a penalty the decode node has to apply, and a node whose engine demoted
    its penalty claim must refuse it rather than decode unpenalised. Without this
    the gate would look only at the request and never see it.

    Raises:
        InvalidParameter: unusable type or value (400).
        CapabilityUnavailable: valid field, no decode-side implementation (501).
    """
    for field, (kind, minimum) in _TYPED_FIELDS.items():
        if body.get(field) is None:
            continue
        if kind == _NUMBER:
            value = _lenient_float(field, body[field])
            if minimum is not None and value < minimum:
                raise InvalidParameter(f"{field} must be >= {minimum}, got {value}")
        else:
            _lenient_int(field, body[field])

    for field, (kind, neutral, minimum, why) in _STATIC_FIELDS.items():
        if field not in body:
            continue
        if _is_neutral(field, body[field], kind, neutral, minimum):
            continue
        raise CapabilityUnavailable(
            f"{field} is not supported by the TileRT decode stage: {why}. "
            f"The vLLM prefill instance would apply it to the first token and "
            f"the decode node would ignore it for the rest of the reply, so "
            f"the request is refused instead of served incorrectly."
        )

    caps = capabilities or NodeCapabilities()
    for field, (kind, neutral, attr, why) in _PROFILE_FIELDS.items():
        if field in body:
            value, origin = body[field], "the request"
        elif adopted is not None and field in adopted:
            value, origin = adopted[field], "this deployment's defaults"
        else:
            continue
        if _is_neutral(field, value, kind, neutral, None):
            continue
        if getattr(caps, attr):
            continue
        raise CapabilityUnavailable(
            f"{field} (from {origin}) is not supported by the decode node "
            f"serving this pool: {why}. It would apply to the first token only, "
            f"so the request is refused instead of served incorrectly."
        )
