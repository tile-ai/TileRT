"""Adapter tests for MlaNsaEngineAdapter.prepare_grammar (Stage 2/3).

The full masked decode loops (_decode_standard / _decode_mtp) need the real GPU
decode layer + a GLM tokenizer, so they are validated on-cluster (ladder rung
2/3). Here we cover what IS reachable off-GPU:
  * no spec -> None (unconstrained; no backend import attempted)
  * prepare_grammar builds a GrammarSession with the right num_positions
    (mtp_seq_len for MTP, 1 for AR) and the per-request think_end_id.

  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_grammar_adapter.py -v
"""

import types

from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter


def _fake_generator():
    return types.SimpleNamespace(
        mtp_seq_len=4,
        decode_layer=types.SimpleNamespace(max_seq_len=4096),
        stop_token_ids={2},
        config=types.SimpleNamespace(vocab_size=288),
        tokenizer=types.SimpleNamespace(convert_tokens_to_ids=lambda s: 257),  # </think> id
        _grammar_engine=None,
    )


def test_prepare_none_returns_none():
    for with_mtp in (True, False):
        adapter = MlaNsaEngineAdapter(_fake_generator(), with_mtp=with_mtp)
        assert adapter.prepare_grammar(None) is None


def _install_fake_grammar(monkeypatch):
    """Stub the backend lookup with recording doubles, so session construction
    can be checked without a real tokenizer/GPU.

    Patch ``load_grammar_backend`` — the indirection — rather than a module
    path in ``sys.modules``. The stubs used to be installed at
    ``tilert.models.glm_5_2.grammar``; the engine moved the wrapper to
    ``tilert.grammar`` and left that path as a fallback the loader only reaches
    when the new one is absent. With a real engine installed the new path
    resolves, the stub was never consulted, and these tests ran the REAL
    GrammarEngine against a SimpleNamespace tokenizer — xgrammar rejected it
    and the tests failed. Patching the lookup keeps them independent of where
    the engine happens to keep the wrapper, which is the whole point of
    ``grammar_backend``.

    The name is bound at import time in the profile module, so the patch has to
    land on that module's attribute, not on ``grammar_backend``'s.
    """
    calls: dict = {}

    class FakeEngine:
        def __init__(self, tok, padded_vocab_size, stop_token_ids):
            calls["engine"] = (padded_vocab_size, tuple(stop_token_ids))

    class FakeSession:
        def __init__(self, engine, spec, num_positions, think_end_id):
            calls["session"] = {
                "num_positions": num_positions,
                "think_end_id": think_end_id,
                "spec": spec,
            }

    monkeypatch.setattr(
        "tilert.pd_vllm.profiles.mla_nsa.load_grammar_backend", lambda: (FakeEngine, FakeSession)
    )
    return calls


def test_prepare_mtp_uses_mtp_seq_len_positions(monkeypatch):
    calls = _install_fake_grammar(monkeypatch)
    adapter = MlaNsaEngineAdapter(_fake_generator(), with_mtp=True)
    adapter.prepare_grammar({"type": "regex", "value": "[0-9]"}, enable_thinking=True)
    assert calls["session"]["num_positions"] == 4  # == mtp_seq_len
    assert calls["session"]["think_end_id"] == 257  # </think> resolved
    assert calls["engine"][0] == 288  # padded vocab size


def test_prepare_ar_uses_single_position(monkeypatch):
    calls = _install_fake_grammar(monkeypatch)
    adapter = MlaNsaEngineAdapter(_fake_generator(), with_mtp=False)
    adapter.prepare_grammar({"type": "regex", "value": "[0-9]"}, enable_thinking=True)
    assert calls["session"]["num_positions"] == 1  # non-MTP AR


def test_prepare_no_thinking_gate_when_disabled(monkeypatch):
    calls = _install_fake_grammar(monkeypatch)
    adapter = MlaNsaEngineAdapter(_fake_generator(), with_mtp=True)
    adapter.prepare_grammar({"type": "regex", "value": "[0-9]"}, enable_thinking=False)
    assert calls["session"]["think_end_id"] is None
