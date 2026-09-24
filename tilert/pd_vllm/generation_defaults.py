from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

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
VLLM_NEUTRAL_TEMPERATURE = 1.0
VLLM_NEUTRAL_TOP_K = 0
GUARDED_FIELDS = {"repetition_penalty": 1.0, "min_p": 0.0}
NEUTRAL_REPETITION_PENALTY = 1.0
NEUTRAL_MIN_P = 0.0


class UnsupportedGenerationDefault(Exception):
    pass


@dataclass(frozen=True)
class GenerationDefaults:
    temperature: float = VLLM_NEUTRAL_TEMPERATURE
    top_p: float = VLLM_DEFAULT_TOP_P
    top_k: int = VLLM_NEUTRAL_TOP_K
    repetition_penalty: float = NEUTRAL_REPETITION_PENALTY
    source: str = "vllm"

    def resolve(self, body: dict) -> dict:
        return {
            "temperature": _as_float("temperature", body.get("temperature"), self.temperature),
            "top_p": _as_float("top_p", body.get("top_p"), self.top_p),
            "top_k": _as_int("top_k", body.get("top_k"), self.top_k),
            "repetition_penalty": _as_float(
                "repetition_penalty", body.get("repetition_penalty"), self.repetition_penalty
            ),
            "min_p": NEUTRAL_MIN_P,
        }

    def describe(self) -> str:
        return f"temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, repetition_penalty={self.repetition_penalty} (from {self.source})"


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
            "no generation_config.json under %s; using vLLM's neutral sampling defaults", model_path
        )
        return {}
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise UnsupportedGenerationDefault(f"{path} does not contain a JSON object")
    return config


def penalties_supported_by(model: str) -> bool:
    if not model:
        return False
    try:
        from tilert.pd_vllm.profiles import base as profiles

        profile = profiles.get_profile(model)
    except Exception as e:
        logger.warning(
            "cannot resolve model %r to a profile (%s); treating penalties as unsupported for default resolution",
            model,
            e,
        )
        return False
    return bool(getattr(profile, "declares_penalties", False))


def _check_guarded(config: dict, source: str, *, model: str, penalties_ok: bool) -> None:
    for field, neutral in sorted(GUARDED_FIELDS.items()):
        raw = config.get(field)
        if raw is None or float(raw) == neutral:
            continue
        if field == "repetition_penalty" and penalties_ok:
            continue
        if field == "min_p":
            reason = "no decode runtime implements it on any model"
        elif model:
            reason = f"the decode runtime for {model!r} has no penalty pre-pass"
        else:
            reason = "the router was not told which model it serves, so it cannot confirm the decode runtime applies it -- pass --model"
        raise UnsupportedGenerationDefault(
            f"{source} sets {field}={raw}, but {reason}. The vLLM prefill instance would apply it to the first token while the decode node would not apply it to the rest, and no per-request check can catch it because the client never sent it.\nResolve it explicitly, whichever is true:\n  - the value is not wanted: launch with --generation-config vllm, or remove {field} from generation_config.json;\n  - the value is wanted: have clients send {field} per request, so the capability gate accepts it on a node that supports it and refuses it on one that does not."
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
    penalties_ok = penalties_supported_by(model)
    config: dict = {}
    if source == "auto" and model_path:
        config = _read_config(model_path)
        origin = os.path.join(model_path, "generation_config.json")
    elif source == "auto":
        logger.info("no --model-path given; using vLLM's neutral sampling defaults")
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
    _check_guarded(
        {k: v for k, v in overrides.items() if v is not None},
        "command-line overrides",
        model=model,
        penalties_ok=penalties_ok,
    )
    if not penalties_ok:
        config = {k: v for k, v in config.items() if k != "repetition_penalty"}
    resolved = {}
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
    logger.info("sampling defaults for requests that omit a field: %s", defaults.describe())
    return defaults
