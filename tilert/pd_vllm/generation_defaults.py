"""Where a request's sampling defaults come from, resolved once for both PD legs.

A PD request is sampled in two places -- the vLLM prefill instance takes token 1,
the decode node takes tokens 2..N -- and each used to work out its own defaults
for a field the client left out. vLLM resolves

    client explicit value  >  the model's generation_config.json  >  1.0 / 1.0 / 0

(``ModelConfig.get_diff_sampling_param``, resolved once at startup and applied per
request in ``to_sampling_params``), while the decode adapters carried literals of
their own. A checkpoint shipping ``temperature: 0.6`` / ``top_k: 20`` then had
token 1 sampled at 0.6 / 20 and tokens 2..N at 1.0 / uncapped.

This module is the single resolution point. It mirrors vLLM's chain, and the
router writes the result into BOTH requests explicitly -- which is what makes the
agreement hold: an explicit value overrides vLLM's own resolution, so the prefill
instance's ``--generation-config`` flag can no longer move one leg without the
other.

Adopted fields
--------------
``temperature``, ``top_p``, ``top_k`` only. Every profile's decode path applies
all three on every request, so a value taken from the model's config is
guaranteed to reach both legs.

``repetition_penalty`` is adopted only where the served model's decode runtime
implements it, which the profile states statically (``declares_penalties``; the
MLA/NSA members -- GLM-5, GLM-5.2, DSV3.2 -- say no). Told which model it serves
(``--model``), the router adopts the config's value on a profile that declares
penalties and refuses to start on the others -- because there the prefill
instance would apply it to the first token and the decode node could not apply
it to the rest. Without ``--model`` the family is unknown and the conservative
answer is taken.

``min_p`` is never adopted: no decode runtime implements it on any member.

``max_new_tokens`` is not adopted either, and needs no guard: the prefill request
is pinned to ``max_tokens=1`` and the decode length comes from the client or the
decode node's own default, so the two legs cannot disagree about it.

Difference from vLLM worth knowing: this reads ``generation_config.json``
directly, while vLLM loads it through HF's ``GenerationConfig.to_diff_dict()``,
which drops any value equal to HF's own default (``top_k=50`` among them). So a
config stating a field at HF's default is honoured here and ignored there. It
cannot split the two legs -- both receive whatever this resolves -- but it can
make a PD deployment sample differently from a stock vLLM one.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from tilert.pd_vllm.sampling import VLLM_DEFAULT_TOP_P

__all__ = [
    "GUARDED_FIELDS",
    "NEUTRAL_MIN_P",
    "NEUTRAL_REPETITION_PENALTY",
    "GenerationDefaults",
    "UnsupportedGenerationDefault",
    "load",
    "penalties_supported_by",
]

logger = logging.getLogger("pd_vllm.generation_defaults")

# vLLM's neutral defaults, the last link of its chain
# (``ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS``). ``top_k`` is 0, vLLM's
# documented "no rank cut" sentinel -- NOT the engine-side ``TOP_K_DISABLED``,
# which is the kernel's candidate-pool bound. What travels on the wire is the
# request-domain value; ``sampling.resolve_top_k`` maps it for the engine.
VLLM_NEUTRAL_TEMPERATURE = 1.0
VLLM_NEUTRAL_TOP_K = 0

# Fields whose adoption depends on the model, with the value that means "asks for
# nothing". A config carrying one at a non-neutral value that this deployment
# cannot execute is a STARTUP error: the prefill instance would apply it to token
# 1 while the decode node could not apply it to the rest, and no per-request check
# can see it -- the client never sent it, so the capability gate never looks at
# it. Better to refuse to start than to serve every request half-penalised.
GUARDED_FIELDS = {"repetition_penalty": 1.0, "min_p": 0.0}

# The value each guarded field takes when it is not adopted. Sent explicitly to
# the prefill leg regardless, so vLLM cannot resolve one of its own from the
# model config behind the router's back.
NEUTRAL_REPETITION_PENALTY = 1.0
NEUTRAL_MIN_P = 0.0


class UnsupportedGenerationDefault(Exception):
    """The model's generation_config asks for something PD cannot guarantee."""


@dataclass(frozen=True)
class GenerationDefaults:
    """The sampling defaults this deployment applies to both legs."""

    temperature: float = VLLM_NEUTRAL_TEMPERATURE
    top_p: float = VLLM_DEFAULT_TOP_P
    top_k: int = VLLM_NEUTRAL_TOP_K
    # Adopted only on a family whose decode runtime implements it; otherwise the
    # runtime no-op, which is what the prefill leg is then pinned to as well.
    repetition_penalty: float = NEUTRAL_REPETITION_PENALTY
    # Where the values came from, for the startup log line.
    source: str = "vllm"

    def resolve(self, body: dict) -> dict:
        """The three fields for this request: client value, else the default.

        ``None`` counts as absent, which is how SDKs spell "unset" and how vLLM
        treats it. Values are returned in the REQUEST domain, so they can be
        written into the prefill body and the decode payload unchanged.
        """
        return {
            "temperature": _as_float("temperature", body.get("temperature"), self.temperature),
            "top_p": _as_float("top_p", body.get("top_p"), self.top_p),
            "top_k": _as_int("top_k", body.get("top_k"), self.top_k),
            # Pinned on both legs even at its no-op value: left unset, the
            # prefill instance would take one from generation_config while the
            # decode node took the router's, which is the split this module
            # exists to close.
            "repetition_penalty": _as_float(
                "repetition_penalty", body.get("repetition_penalty"), self.repetition_penalty
            ),
            # Always the no-op, and always sent. The router's own
            # --generation-config only governs what the ROUTER reads; the vLLM
            # server is launched separately and still defaults to loading the
            # checkpoint's generation_config.json. So a checkpoint carrying
            # min_p would have it applied to token 1 and ignored for the rest --
            # including on the very path documented as the way out of the startup
            # refusal. Pinning it closes vLLM's six-key allowlist: temperature,
            # top_p, top_k and repetition_penalty are resolved above,
            # max_new_tokens is overridden by max_tokens=1, and this is the last.
            "min_p": NEUTRAL_MIN_P,
        }

    def describe(self) -> str:
        return (
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"repetition_penalty={self.repetition_penalty} "
            f"(from {self.source})"
        )


def _as_float(field: str, raw, default: float) -> float:
    if raw is None:
        return float(default)
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be a number, got bool")
    return float(raw)


def _as_int(field: str, raw, default: int) -> int:
    if raw is None:
        return int(default)
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be an integer, got bool")
    return int(raw)


def _read_config(model_path: str) -> dict:
    path = os.path.join(model_path, "generation_config.json")
    if not os.path.isfile(path):
        logger.info(
            "no generation_config.json under %s; using vLLM's neutral " "sampling defaults",
            model_path,
        )
        return {}
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise UnsupportedGenerationDefault(f"{path} does not contain a JSON object")
    return config


def penalties_supported_by(model: str) -> bool:
    """Whether the named model's FAMILY implements repetition/presence penalties.

    Read from the profile's static ``declares_penalties``, which is the only
    answer available at startup: the per-node answer comes from
    ``/capabilities`` and needs a running decode node. An unknown or unnamed
    model is treated as unsupported -- the conservative direction, since the
    cost is refusing to adopt a default rather than serving half-penalised.
    """
    if not model:
        return False
    try:
        from tilert.pd_vllm.profiles import base as profiles

        profile = profiles.get_profile(model)
    except Exception as e:  # unknown name, or a profile that cannot import here
        logger.warning(
            "cannot resolve model %r to a profile (%s); treating "
            "penalties as unsupported for default resolution",
            model,
            e,
        )
        return False
    return bool(getattr(profile, "declares_penalties", False))


def _check_guarded(config: dict, source: str, *, model: str, penalties_ok: bool) -> None:
    """Refuse a config asking for a field this deployment cannot execute.

    ``min_p`` is refused on every member. ``repetition_penalty`` is refused only
    where the family lacks the pre-pass -- a profile declaring penalties adopts it.
    """
    for field, neutral in sorted(GUARDED_FIELDS.items()):
        raw = config.get(field)
        if raw is None or float(raw) == neutral:
            continue
        if field == "repetition_penalty" and penalties_ok:
            continue
        if field == "min_p":
            reason = "no decode runtime implements it on any model"
        elif model:
            reason = (
                f"the decode runtime for {model!r} has no penalty "
                f"pre-pass (the GLM-5 / GLM-5.2 / DSV3.2 members do not "
                f"declare one)"
            )
        else:
            reason = (
                "the router was not told which model it serves, so it "
                "cannot confirm the decode runtime applies it -- pass "
                "--model"
            )
        raise UnsupportedGenerationDefault(
            f"{source} sets {field}={raw}, but {reason}. The vLLM prefill "
            f"instance would apply it to the first token while the decode node "
            f"would not apply it to the rest, and no per-request check can "
            f"catch it because the client never sent it.\n"
            f"Resolve it explicitly, whichever is true:\n"
            f"  - the value is not wanted: launch with --generation-config "
            f"vllm, or remove {field} from generation_config.json;\n"
            f"  - the value is wanted: have clients send {field} per request, "
            f"so the capability gate accepts it on a node that supports it and "
            f"refuses it on one that does not."
        )


def load(
    model_path: str = "",
    source: str = "auto",
    *,
    model: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> GenerationDefaults:
    """Resolve this deployment's sampling defaults, once, at startup.

    ``source`` mirrors vLLM's ``--generation-config``: ``"auto"`` reads
    ``generation_config.json`` under ``model_path``, ``"vllm"`` ignores it and
    uses the neutral defaults. The keyword overrides are the equivalent of
    ``--override-generation-config`` and win over both.

    ``model`` is the profile name the decode nodes serve (``--model``), used only
    to decide whether a ``repetition_penalty`` in the config can be adopted.

    Raises:
        UnsupportedGenerationDefault: the config asks for a guarded field
            (see :data:`GUARDED_FIELDS`) this deployment cannot apply on both
            legs.
    """
    penalties_ok = penalties_supported_by(model)
    config: dict = {}
    if source == "auto" and model_path:
        config = _read_config(model_path)
        origin = os.path.join(model_path, "generation_config.json")
    elif source == "auto":
        logger.info("no --model-path given; using vLLM's neutral sampling " "defaults")
        origin = "vllm neutral defaults"
    else:
        origin = "vllm neutral defaults"

    if config:
        _check_guarded(config, origin, model=model, penalties_ok=penalties_ok)

    overrides = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    # The overrides win over the file, so they need the same guard -- otherwise
    # --default-repetition-penalty on a family without the pre-pass starts the
    # router with an adopted default the gate then refuses on EVERY request,
    # which is a worse outcome than refusing to start.
    _check_guarded(
        {k: v for k, v in overrides.items() if v is not None},
        "command-line overrides",
        model=model,
        penalties_ok=penalties_ok,
    )
    if not penalties_ok:
        # Not executable here, so it is pinned to the no-op on BOTH legs rather
        # than left for vLLM to resolve from the model config on one of them.
        config = {k: v for k, v in config.items() if k != "repetition_penalty"}
    resolved: dict[str, Any] = {}
    for field, neutral in (
        ("temperature", VLLM_NEUTRAL_TEMPERATURE),
        ("top_p", VLLM_DEFAULT_TOP_P),
        ("top_k", VLLM_NEUTRAL_TOP_K),
        ("repetition_penalty", NEUTRAL_REPETITION_PENALTY),
    ):
        if overrides[field] is not None:
            resolved[field] = overrides[field]
        elif config.get(field) is not None:
            resolved[field] = config[field]
        else:
            resolved[field] = neutral

    if any(v is not None for v in overrides.values()):
        origin = f"{origin} + command-line overrides"

    defaults = GenerationDefaults(
        temperature=float(resolved["temperature"]),
        top_p=float(resolved["top_p"]),
        top_k=int(resolved["top_k"]),
        repetition_penalty=float(resolved["repetition_penalty"]),
        source=origin,
    )
    # vLLM logs when the model's config displaces its neutral defaults, and an
    # operator comparing the two stacks needs the same line from this side.
    logger.info("sampling defaults for requests that omit a field: %s", defaults.describe())
    return defaults
