"""`ignore_eos` must empty the MLA/NSA adapter's stop set for that request.

The router half is pinned in tests/pd_vllm/test_stream_e2e.py (the flag reaches
the decode node's `sampling` dict). This file covers the far end: the adapter
reading it, and both decode loops binding the stop set through it.

Wiring only one of the two loops is the failure this guards. It is silent --
nothing errors, the request just stops at the first EOS, which is precisely
what a fixed-output-length benchmark (`vllm bench serve --ignore-eos`) asked it
not to do, so measured decode throughput becomes whatever the model happened to
emit.

Scope note: the MLA/NSA adapter honours the flag (tile-ai/TileRT#55); the
ROCm GLM adapter does too, and is covered in test_glm5_rocm_engine.py.

No GPU, no tilert: the adapter is built with ``object.__new__`` and handed a
recording stand-in for ``self.gen``, with ``max_tokens=0`` so ``decode()``
returns right after ``update_sampling_params``.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_ignore_eos.py -v
"""

from __future__ import annotations

import pathlib

import pytest

from tilert.pd_vllm.profiles.mla_nsa import MlaNsaEngineAdapter


class _StubGen:
    """Enough generator for decode() to reach its budget short-circuit."""

    def update_sampling_params(self, **kw):
        pass


def _adapter() -> MlaNsaEngineAdapter:
    a = object.__new__(MlaNsaEngineAdapter)
    a.gen = _StubGen()
    a.with_mtp = False
    a.mtp_seq_len = 4
    a.max_seq_len = 4096
    a._seq_len = 8
    a.stop_ids = {7, 8}
    a._ignore_eos = False
    a.last_stats = {}
    return a


def _decode_once(sampling: dict) -> MlaNsaEngineAdapter:
    """max_tokens=0 => budget <= 0 => returns before touching the model."""
    a = _adapter()
    assert a.decode(
        first_token_id=5, max_tokens=0, sampling=sampling, cancel_event=None, grammar_session=None
    ) == [5]
    return a


def test_the_adapter_records_the_flag() -> None:
    assert _decode_once({"ignore_eos": True})._ignore_eos is True


def test_absent_means_stop_at_eos() -> None:
    """The default every other request has must not shift."""
    assert _decode_once({"temperature": 0.0})._ignore_eos is False


def test_explicit_false_means_stop_at_eos() -> None:
    assert _decode_once({"ignore_eos": False})._ignore_eos is False


def test_the_flag_does_not_persist_into_the_next_request() -> None:
    """The adapter is reused across requests on a decode node, so a sticky flag
    would leak EOS-less decoding into every request that followed.
    """
    a = _adapter()
    a._ignore_eos = True
    a.decode(first_token_id=5, max_tokens=0, sampling={}, cancel_event=None, grammar_session=None)
    assert a._ignore_eos is False


def test_both_decode_loops_read_the_guarded_stop_set() -> None:
    """Neither loop may bind the raw stop set.

    The adapter has two (`_decode_mtp` and `_decode_standard`) and the flag only
    bites where the loop actually breaks. Source-level, in the shape
    test_top_k_resolution.py uses, because reaching those loops needs a GPU and
    weights.
    """
    src = pathlib.Path(pytest.importorskip("tilert.pd_vllm.profiles.mla_nsa").__file__).read_text()
    binds = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("stop_ids = ")]
    assert len(binds) == 2, f"expected one binding per decode loop: {binds}"
    assert all(b == "stop_ids = set() if self._ignore_eos else self.stop_ids" for b in binds), binds
