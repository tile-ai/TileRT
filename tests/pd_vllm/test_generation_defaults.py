"""Sampling defaults resolve once, the way vLLM resolves them, for both PD legs.

vLLM works out its ``default_sampling_params`` at startup
(``ModelConfig.get_diff_sampling_param``): read the model's
``generation_config.json`` unless ``--generation-config vllm``, apply
``--override-generation-config``, keep a six-key allowlist, then per request
resolve ``client value > that > the neutral defaults``.

The decode adapters used to carry literals of their own instead, which for a
checkpoint recommending 0.6 / 20 put ``temperature`` at 0.6 for the first token
and 1.0 for the rest, and ``top_k`` at 20 for the first token and uncapped for
the rest. This module resolves once and
the router writes the result into both requests, so the legs cannot drift.

Guarded fields are the other half: ``min_p`` has no decode implementation on any
member, and ``repetition_penalty`` only where the profile declares a penalty
pre-pass (no public profile does today, so a stub stands in). A config asking for one
this deployment cannot execute fails at STARTUP -- the client never sent it, so no
per-request check would ever see it.

CPU only -- no GPU, no tilert, no vLLM.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_generation_defaults.py -v
"""

import json
import types

import pytest

from tilert.pd_vllm import generation_defaults as gd
from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    NodeCapabilities,
    validate_generation_request,
)

RECOMMENDED_CONFIG = {
    # A shipped generation_config.json with recommended sampling, sampling keys only.
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    # Keys vLLM's allowlist ignores; they must not reach the sampler.
    "do_sample": True,
    "bos_token_id": 248044,
    "eos_token_id": [248046, 248044],
}


@pytest.fixture(autouse=True)
def penalty_profile(monkeypatch):
    """Register a profile whose decode runtime declares penalties.

    ``penalties_supported_by`` reads the profile's static ``declares_penalties``;
    none of the public profiles (GLM-5 / GLM-5.2 / DSV3.2) set it, so the
    adopt-a-penalty path is exercised through a stub registered for the test.
    """
    from tilert.pd_vllm.profiles import base

    stub = types.SimpleNamespace(name="stub_penalties", declares_penalties=True)
    monkeypatch.setitem(base._REGISTRY, "stub_penalties", stub)


@pytest.fixture
def model_dir(tmp_path):
    def _write(config):
        (tmp_path / "generation_config.json").write_text(json.dumps(config))
        return str(tmp_path)

    return _write


# --------------------------------------------------------------------------- #
# The chain, and what it takes from the file
# --------------------------------------------------------------------------- #
def test_the_model_config_supplies_the_defaults(model_dir):
    d = gd.load(model_dir(RECOMMENDED_CONFIG), "auto", model="stub_penalties")
    assert (d.temperature, d.top_p, d.top_k) == (0.6, 0.95, 20)


def test_keys_outside_the_sampling_allowlist_are_ignored(model_dir):
    """``generation_config.json`` is a general HF file: it carries token ids and
    ``do_sample`` too, and vLLM's allowlist is what keeps those out of the
    sampler. Adopting the file wholesale would put ``eos_token_id`` on the wire
    as a sampling parameter.
    """
    d = gd.load(model_dir(RECOMMENDED_CONFIG), "auto", model="stub_penalties")
    resolved = d.resolve({})
    # Exactly vLLM's six-key allowlist minus max_new_tokens, which max_tokens=1
    # already overrides on the prefill leg.
    assert set(resolved) == {"temperature", "top_p", "top_k", "repetition_penalty", "min_p"}


def test_generation_config_vllm_ignores_the_file(model_dir):
    """vLLM's own escape hatch, spelled the same way."""
    d = gd.load(model_dir(RECOMMENDED_CONFIG), "vllm", model="stub_penalties")
    assert (d.temperature, d.top_p, d.top_k) == (1.0, 1.0, 0)


def test_a_missing_file_falls_back_to_the_neutral_defaults(tmp_path):
    d = gd.load(str(tmp_path), "auto", model="stub_penalties")
    assert (d.temperature, d.top_p, d.top_k) == (1.0, 1.0, 0)


def test_no_model_path_falls_back_to_the_neutral_defaults():
    d = gd.load("", "auto", model="stub_penalties")
    assert (d.temperature, d.top_p, d.top_k) == (1.0, 1.0, 0)


def test_a_non_object_file_is_refused(tmp_path):
    (tmp_path / "generation_config.json").write_text("[1, 2, 3]")
    with pytest.raises(gd.UnsupportedGenerationDefault):
        gd.load(str(tmp_path), "auto", model="stub_penalties")


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.9),
        ("top_p", 0.5),
        ("top_k", 40),
    ],
)
def test_a_command_line_override_wins_over_the_file(model_dir, field, value):
    d = gd.load(model_dir(RECOMMENDED_CONFIG), "auto", model="stub_penalties", **{field: value})
    assert getattr(d, field) == value


def test_a_partial_config_only_displaces_what_it_states(model_dir):
    d = gd.load(model_dir({"top_p": 0.8}), "auto", model="stub_penalties")
    assert d.top_p == 0.8
    assert (d.temperature, d.top_k) == (1.0, 0)  # still neutral


def test_the_source_is_named_for_the_startup_log(model_dir):
    path = model_dir(RECOMMENDED_CONFIG)
    assert "generation_config.json" in gd.load(path, "auto", model="stub_penalties").source
    assert "neutral" in gd.load(path, "vllm", model="stub_penalties").source
    assert "override" in gd.load(path, "auto", model="stub_penalties", top_p=0.5).source


# --------------------------------------------------------------------------- #
# Per-request resolution: the client always wins
# --------------------------------------------------------------------------- #
DEFAULTS = gd.GenerationDefaults(temperature=0.6, top_p=0.95, top_k=20)


@pytest.mark.parametrize(
    "field,sent,want",
    [
        ("temperature", 0.2, 0.2),
        ("top_p", 0.5, 0.5),
        ("top_k", 5, 5),
    ],
)
def test_an_explicit_client_value_wins(field, sent, want):
    assert DEFAULTS.resolve({field: sent})[field] == want


@pytest.mark.parametrize(
    "field,want",
    [
        ("temperature", 0.6),
        ("top_p", 0.95),
        ("top_k", 20),
    ],
)
def test_an_absent_field_takes_the_deployment_default(field, want):
    assert DEFAULTS.resolve({})[field] == want


@pytest.mark.parametrize(
    "field,want",
    [
        ("temperature", 0.6),
        ("top_p", 0.95),
        ("top_k", 20),
    ],
)
def test_an_explicit_null_is_treated_as_absent(field, want):
    """How SDKs spell "unset", and how vLLM resolves it."""
    assert DEFAULTS.resolve({field: None})[field] == want


def test_zero_is_a_value_not_an_absence():
    """``temperature: 0`` is a greedy request, not a missing field.

    Falling back to the default here would silently make it sample.
    """
    assert DEFAULTS.resolve({"temperature": 0})["temperature"] == 0.0
    assert DEFAULTS.resolve({"top_k": 0})["top_k"] == 0


@pytest.mark.parametrize("field", ["temperature", "top_p", "top_k"])
def test_a_bool_is_not_a_number(field):
    with pytest.raises(ValueError):
        DEFAULTS.resolve({field: True})


def test_top_k_travels_in_the_request_domain():
    """0 is vLLM's "no rank cut" sentinel, which is what belongs on the wire.

    The kernel's own disabled value (``TOP_K_DISABLED`` = the candidate-pool
    bound) is an engine-side mapping and would be a real rank cut to vLLM.
    """
    from tilert.pd_vllm.sampling import TOP_K_DISABLED, resolve_top_k

    wire_value = gd.GenerationDefaults().resolve({})["top_k"]
    assert wire_value == 0
    # ... and the engine maps that sentinel to "no cut" on its side.
    assert resolve_top_k({"top_k": wire_value}) == TOP_K_DISABLED


# --------------------------------------------------------------------------- #
# Guarded fields: refused at startup, per model family
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model,supported",
    [
        ("stub_penalties", True),
        ("glm5", False),
        ("glm5_2", False),
        ("glm5_3", False),
        ("dsv32", False),
        ("", False),
        ("not-a-model", False),
    ],
)
def test_which_families_declare_penalties(model, supported):
    assert gd.penalties_supported_by(model) is supported


@pytest.mark.parametrize("model", ["stub_penalties"])
def test_a_penalty_is_adopted_on_a_family_that_declares_it(model_dir, model):
    d = gd.load(model_dir({**RECOMMENDED_CONFIG, "repetition_penalty": 1.05}), "auto", model=model)
    assert d.repetition_penalty == 1.05


@pytest.mark.parametrize("model", ["glm5", "glm5_2", "dsv32"])
def test_a_penalty_refuses_startup_on_a_family_without_the_pre_pass(model_dir, model):
    """Serving would mean penalising the first token and not the rest, on every
    request, with no per-request check able to see it.
    """
    with pytest.raises(gd.UnsupportedGenerationDefault) as e:
        gd.load(model_dir({**RECOMMENDED_CONFIG, "repetition_penalty": 1.05}), "auto", model=model)
    assert "repetition_penalty" in str(e.value)
    # The message must say how to proceed, not just that it stopped.
    assert "--generation-config vllm" in str(e.value)


def test_an_unnamed_model_refuses_a_penalty(model_dir):
    """Unknown family is the conservative answer: the cost is refusing to adopt
    a default, not serving half-penalised.
    """
    with pytest.raises(gd.UnsupportedGenerationDefault) as e:
        gd.load(model_dir({**RECOMMENDED_CONFIG, "repetition_penalty": 1.05}), "auto")
    assert "--model" in str(e.value)


@pytest.mark.parametrize("model", ["stub_penalties", "glm5_2"])
def test_min_p_refuses_startup_on_every_family(model_dir, model):
    with pytest.raises(gd.UnsupportedGenerationDefault) as e:
        gd.load(model_dir({**RECOMMENDED_CONFIG, "min_p": 0.05}), "auto", model=model)
    assert "min_p" in str(e.value)


@pytest.mark.parametrize("model", ["stub_penalties", "glm5_2"])
def test_a_guarded_field_at_its_neutral_value_is_accepted(model_dir, model):
    """Stating the no-op asks for nothing, so there is nothing to refuse."""
    d = gd.load(
        model_dir({**RECOMMENDED_CONFIG, "repetition_penalty": 1.0, "min_p": 0.0}),
        "auto",
        model=model,
    )
    assert d.repetition_penalty == 1.0


def test_generation_config_vllm_clears_a_guarded_field(model_dir):
    """The documented way out of the startup refusal."""
    d = gd.load(model_dir({**RECOMMENDED_CONFIG, "min_p": 0.05}), "vllm", model="glm5_2")
    assert d.repetition_penalty == 1.0


def test_an_unadoptable_penalty_is_pinned_to_the_no_op(model_dir):
    """On a family without the pre-pass, the penalty must not be left for vLLM
    to resolve from the model config on the prefill leg alone -- both legs are
    pinned to the runtime no-op instead.
    """
    d = gd.load(
        model_dir({**RECOMMENDED_CONFIG, "repetition_penalty": 1.0}), "auto", model="glm5_2"
    )
    assert d.resolve({})["repetition_penalty"] == 1.0


# --------------------------------------------------------------------------- #
# An adopted default is gated like a client-sent one
# --------------------------------------------------------------------------- #
def test_an_adopted_penalty_is_refused_on_a_node_that_cannot_apply_it():
    """The family declares support, but THIS node's engine demoted its claim (an
    older tilert wheel without the pre-pass). The value never appears in the
    request body, so the gate has to be told about it or it decodes unpenalised.
    """
    adopted = gd.GenerationDefaults(repetition_penalty=1.05).resolve({})
    body = {"messages": [{"role": "user", "content": "hi"}]}

    # A node that can apply it: served.
    validate_generation_request(body, NodeCapabilities(penalties=True), adopted)

    # A node that cannot: refused, and the message says where the value is from.
    with pytest.raises(CapabilityUnavailable) as e:
        validate_generation_request(body, NodeCapabilities(penalties=False), adopted)
    assert "deployment" in str(e.value)


def test_a_neutral_adopted_penalty_needs_no_capability():
    adopted = gd.GenerationDefaults().resolve({})
    validate_generation_request({}, NodeCapabilities(penalties=False), adopted)


def test_a_client_value_is_reported_as_the_client_s():
    """The two origins must be distinguishable in the error, or an operator
    cannot tell "drop the field" from "fix the deployment".
    """
    with pytest.raises(CapabilityUnavailable) as e:
        validate_generation_request(
            {"repetition_penalty": 1.2},
            NodeCapabilities(penalties=False),
            gd.GenerationDefaults().resolve({}),
        )
    assert "the request" in str(e.value)


# --------------------------------------------------------------------------- #
# Review findings on PR #40 (codex)
# --------------------------------------------------------------------------- #
def test_min_p_is_pinned_even_when_the_router_ignores_the_model_config():
    """`--generation-config vllm` governs the ROUTER's reading, not the vLLM
    server's.

    The prefill instance is launched separately and still defaults to loading the
    checkpoint's generation_config.json, so a checkpoint carrying min_p would
    have it applied to token 1 and ignored for the rest -- on the very path the
    startup refusal documents as the way out. Pinning the no-op explicitly is
    what closes it.
    """
    for source in ("auto", "vllm"):
        assert gd.load("", source, model="stub_penalties").resolve({})["min_p"] == 0.0


def test_min_p_is_pinned_regardless_of_what_the_client_sent():
    """A non-neutral client min_p is refused by the gate, so the only value that
    may reach the sampler is the no-op.
    """
    assert DEFAULTS.resolve({"min_p": 0.4})["min_p"] == 0.0


def test_the_resolution_covers_vllms_whole_allowlist():
    """vLLM takes six keys from generation_config.

    Any one left unpinned is a field its own resolution can still move on the prefill leg alone.
    """
    vllm_allowlist = {
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "max_new_tokens",
    }
    resolved = set(gd.GenerationDefaults().resolve({}))
    # max_new_tokens is covered by max_tokens=1 on the prefill request instead.
    assert vllm_allowlist - resolved == {"max_new_tokens"}


@pytest.mark.parametrize("model", ["glm5", "glm5_2", "dsv32", ""])
def test_a_command_line_penalty_override_is_guarded_too(model):
    """The overrides win over the file, so they need the same guard.

    Without it the router starts with an adopted default the gate refuses on EVERY request --
    worse than refusing to start.
    """
    with pytest.raises(gd.UnsupportedGenerationDefault) as e:
        gd.load("", "vllm", model=model, repetition_penalty=1.2)
    assert "command-line overrides" in str(e.value)


@pytest.mark.parametrize("model", ["stub_penalties"])
def test_a_command_line_penalty_override_is_honoured_where_executable(model):
    assert gd.load("", "vllm", model=model, repetition_penalty=1.2).repetition_penalty == 1.2


def test_a_neutral_penalty_override_needs_no_capability():
    assert gd.load("", "vllm", model="glm5_2", repetition_penalty=1.0).repetition_penalty == 1.0


def test_a_min_p_override_is_refused_on_every_family():
    """There is no --default-min-p, but the guard covers the override path
    generically, so adding one later cannot bypass it.
    """
    with pytest.raises(gd.UnsupportedGenerationDefault):
        gd._check_guarded(
            {"min_p": 0.05}, "command-line overrides", model="stub_penalties", penalties_ok=True
        )
