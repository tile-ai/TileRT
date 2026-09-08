"""RocmGlm52EngineAdapter: cache slot mapping and decode loop, no GPU.

A FakeShowHands stands in for the ROCm e2e: CPU cache tensors in the real
per-rank layout, a scripted accepted-token stream, and a call log. The tests
pin the contract the GPU run relies on: which slot each (ki, kv, pe) lands
in, cur_pos == seq_len after inject, and the emitted stream honouring stop /
budget / cancel in both MTP and plain modes.
"""

from __future__ import annotations

import threading
import types

import pytest
import torch

from tilert.pd_vllm.grammar_spec import GrammarBackendUnavailable
from tilert.pd_vllm.profiles.glm5_rocm_engine import RocmGlm52EngineAdapter


def _layer_kind(i: int) -> int:
    """Mirror of the engine's rule: 0 = dense, 1 = full (indexer), 2 = shared."""
    if i < 3:
        return 0
    return 1 if (i - 2) % 4 == 0 else 2


def _full_layer_ordinals(n: int) -> list[int]:
    return [i for i in range(n) if _layer_kind(i) != 2]


@pytest.fixture(autouse=True, scope="module")
def _rocm_engine_helpers():
    """Stand in for the two helpers the adapter imports from the ROCm tilert build.

    ``tilert.models.glm_5.model_args.full_layer_ordinals`` and
    ``tilert.models.glm_5.weight_converter.{fp8_ki_enabled,pure_tp8_enabled}``
    ship with the ROCm engine only. When the installed ``tilert`` lacks them
    (the CUDA tree, or no engine at all) stub modules are put in ``sys.modules``
    for the duration of this module so the mapping logic is testable anywhere.
    """
    import sys

    try:
        from tilert.models.glm_5.model_args import full_layer_ordinals  # noqa: F401
        from tilert.models.glm_5.weight_converter import fp8_ki_enabled  # noqa: F401
    except ImportError:
        pass
    else:
        yield
        return
    ma = types.ModuleType("tilert.models.glm_5.model_args")
    ma.layer_kind, ma.full_layer_ordinals = _layer_kind, _full_layer_ordinals
    wc = types.ModuleType("tilert.models.glm_5.weight_converter")
    wc.fp8_ki_enabled = lambda: False
    wc.pure_tp8_enabled = lambda: True
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "tilert.models.glm_5.model_args", ma)
        mp.setitem(sys.modules, "tilert.models.glm_5.weight_converter", wc)
        yield


N_LAYERS = 8  # 3 dense + 5 MoE -> full layers {0,1,2,6}
L = 32  # max_seq_len
NPES = 8
KV, PE, KI = 512, 64, 128


class FakeShowHands:
    def __init__(self, num_mtp: int, pure_tp8: bool, fp8_ki: bool, stream: list[int]):
        self.args = types.SimpleNamespace(max_seq_len=L, num_devices=NPES)
        self.n_layers = N_LAYERS
        self.num_mtp = num_mtp
        self.npes = NPES
        self.calls: list[tuple] = []
        self._stream = list(stream)  # tokens the "device" will emit
        self._ar: list[int] = []
        n_extra = 1 if num_mtp > 0 else 0
        n_full = len(_full_layer_ordinals(N_LAYERS))
        self._caches = []
        for rank in range(NPES):
            c = []
            if rank == 0:
                if pure_tp8:
                    for _ in range(N_LAYERS):
                        c.append(torch.zeros(1, L, KV, dtype=torch.bfloat16))
                        c.append(torch.zeros(1, L, PE, dtype=torch.bfloat16))
                for _ in range(n_full + n_extra):
                    if fp8_ki:
                        c.append(torch.zeros(L * (KI + 4), dtype=torch.uint8))
                    else:
                        c.append(torch.zeros(1, L, KI, dtype=torch.bfloat16))
            else:
                for _ in range(N_LAYERS + n_extra):
                    c.append(torch.zeros(1, L, KV, dtype=torch.bfloat16))
                    c.append(torch.zeros(1, L, PE, dtype=torch.bfloat16))
            self._caches.append(c)

    # -- e2e API --
    def reset_sequence(self):
        self.calls.append(("reset",))
        self._ar = []

    def set_cur_pos(self, p):
        self.calls.append(("set_cur_pos", p))

    def update_sampling(self, use_topp, temperature, top_p):
        self.calls.append(("sampling", use_topp, temperature, top_p))

    def seed_draft(self, tok, draft):
        self.calls.append(("seed_draft", tok, draft))

    def _take(self, n):
        out, self._stream = self._stream[:n], self._stream[n:]
        self._ar.extend(out)
        return out

    def mtp_n(self, k):
        self.calls.append(("mtp_n", k))
        # every verify step accepts 2 tokens (or what is left)
        return len(self._take(2 * k))

    def step(self, tok):
        self.calls.append(("step", tok))
        return self._take(1)[0]

    def decode_n(self, n):
        self.calls.append(("decode_n", n))
        self._take(n)

    @property
    def accepted_count(self):
        return len(self._ar)

    def accepted_tokens(self, start=0, end=None):
        end = len(self._ar) if end is None else end
        return list(self._ar[start:end])


def _gen(dl, stop=(999,)):
    return types.SimpleNamespace(decode_layer=dl, stop_token_ids=set(stop))


def _req(seq, n_layers_sent, first=7):
    layers = []
    for lid in range(n_layers_sent):
        ki = torch.full((seq, KI), float(lid) + 0.5, dtype=torch.bfloat16)
        kv = torch.full((seq, KV), float(lid), dtype=torch.bfloat16)
        pe = torch.full((seq, PE), -float(lid), dtype=torch.bfloat16)
        layers.append((ki, kv, pe))
    return types.SimpleNamespace(
        seq_len=seq, layers=layers, first_token_id=first, last_prompt_token=3
    )


@pytest.fixture(autouse=True)
def _no_gpu_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


@pytest.mark.parametrize("pure_tp8", [True, False])
def test_inject_slot_mapping_bf16_ki(pure_tp8):
    dl = FakeShowHands(num_mtp=3, pure_tp8=pure_tp8, fp8_ki=False, stream=[])
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=True, pure_tp8=pure_tp8, fp8_ki=False)
    seq = 5
    ad.inject(_req(seq, N_LAYERS + 1))
    assert ("reset",) in dl.calls and ("set_cur_pos", seq) == dl.calls[-1]
    # ranks 1..7: pair 2*lid / 2*lid+1 holds layer lid, incl. the MTP block (lid=N_LAYERS)
    for rank in range(1, NPES):
        for lid in range(N_LAYERS + 1):
            assert dl._caches[rank][2 * lid][0, :seq].float().unique().tolist() == [float(lid)]
            assert dl._caches[rank][2 * lid + 1][0, :seq].float().unique().tolist() == [-float(lid)]
            assert dl._caches[rank][2 * lid][0, seq:].abs().sum() == 0
    # rank 0: ki slots follow the full-layer ordinals, MTP block last
    ki_base = 2 * N_LAYERS if pure_tp8 else 0
    full = _full_layer_ordinals(N_LAYERS)
    for slot, lid in enumerate(full + [N_LAYERS]):
        got = dl._caches[0][ki_base + slot][0, :seq].float().unique().tolist()
        assert got == [float(lid) + 0.5], (slot, lid, got)
    if pure_tp8:
        # rank 0 also receives kv/pe for the main layers (its shared-layer MLA chain)
        assert dl._caches[0][2 * 6][0, :seq].float().unique().tolist() == [6.0]
        assert len(dl._caches[0]) == 2 * N_LAYERS + len(full) + 1


def test_inject_fp8_ki_plane_roundtrips():
    dl = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=True, stream=[])
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=False, pure_tp8=True, fp8_ki=True)
    seq = 4
    req = _req(seq, N_LAYERS + 1)
    # a non-constant row so the per-token scale is exercised
    ki0 = torch.arange(seq * KI, dtype=torch.float32).reshape(seq, KI) / 37.0 - 3.0
    req.layers[0] = (ki0.to(torch.bfloat16), req.layers[0][1], req.layers[0][2])
    ad.inject(req)
    plane = dl._caches[0][2 * N_LAYERS]  # ki slot of layer 0
    q = plane[: L * KI].view(torch.float8_e4m3fn).view(L, KI)[:seq].float()
    s = plane[L * KI : L * KI + L * 4].view(torch.float32)[:seq]
    deq = q * s.unsqueeze(-1)
    assert torch.allclose(deq, ki0, rtol=0.13, atol=0.05)  # e4m3 has 3 mantissa bits
    assert plane[L * KI + seq * 4 : L * KI + L * 4].view(torch.float32).abs().sum() == 0


def test_inject_without_mtp_drops_the_tail_layer():
    dl = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=False, stream=[])
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=False, pure_tp8=True, fp8_ki=False)
    ad.inject(_req(3, N_LAYERS + 1))  # profile always ships the tail
    assert len(dl._caches[1]) == 2 * N_LAYERS  # no slot for it, silently dropped


def test_inject_rejects_missing_mtp_layer_and_bad_seq():
    dl = FakeShowHands(num_mtp=3, pure_tp8=True, fp8_ki=False, stream=[])
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=True, pure_tp8=True, fp8_ki=False)
    with pytest.raises(RuntimeError, match="speculative-config"):
        ad.inject(_req(3, N_LAYERS))
    with pytest.raises(RuntimeError, match="seq_len"):
        ad.inject(_req(L + 1, N_LAYERS + 1))


def test_decode_mtp_stream_stop_and_budget():
    stream = [11, 12, 13, 14, 999, 15]
    dl = FakeShowHands(num_mtp=3, pure_tp8=True, fp8_ki=False, stream=stream)
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=True, ar_steps=1, pure_tp8=True, fp8_ki=False)
    ad.inject(_req(4, N_LAYERS + 1))
    seen = []
    out = ad.decode(7, 100, {"temperature": 0.0}, on_token=seen.append)
    assert out == [7, 11, 12, 13, 14] and seen == out
    assert ad.last_stats["finish_reason"] == "stop"
    assert ("seed_draft", 7, 7) in dl.calls
    assert ("sampling", False, 1.0, 1.0) in dl.calls  # greedy arm
    # budget cut
    dl2 = FakeShowHands(num_mtp=3, pure_tp8=True, fp8_ki=False, stream=list(range(100, 140)))
    ad2 = RocmGlm52EngineAdapter(_gen(dl2), with_mtp=True, ar_steps=2, pure_tp8=True, fp8_ki=False)
    ad2.inject(_req(4, N_LAYERS + 1))
    out2 = ad2.decode(7, 5, {"temperature": 0.7, "top_p": 0.9})
    assert out2 == [7, 100, 101, 102, 103] and ad2.last_stats["finish_reason"] == "length"
    assert ("sampling", True, 0.7, 0.9) in dl2.calls


def test_decode_mtp_respects_cache_edge():
    # seq_len close to max_seq_len: no room for a verify chunk -> length, no mtp_n call
    dl = FakeShowHands(num_mtp=3, pure_tp8=True, fp8_ki=False, stream=list(range(100, 120)))
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=True, pure_tp8=True, fp8_ki=False)
    ad.inject(_req(L - 4, N_LAYERS + 1))  # room = 4, needs mtp_seq(4)+slack(2)
    out = ad.decode(7, 50, {})
    assert out == [7] and ad.last_stats["finish_reason"] == "length"
    assert not any(c[0] == "mtp_n" for c in dl.calls)


def test_decode_plain_uses_step_then_decode_n_and_cancel():
    stream = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    dl = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=False, stream=stream)
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=False, pure_tp8=True, fp8_ki=False)
    ad.inject(_req(2, N_LAYERS + 1))
    out = ad.decode(7, 6, {"temperature": 1.0})
    assert out == [7, 21, 22, 23, 24, 25]
    kinds = [c[0] for c in dl.calls if c[0] in ("step", "decode_n")]
    assert kinds[0] == "step" and "decode_n" in kinds
    ev = threading.Event()
    ev.set()
    dl3 = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=False, stream=list(range(50, 90)))
    ad3 = RocmGlm52EngineAdapter(_gen(dl3), with_mtp=False, pure_tp8=True, fp8_ki=False)
    ad3.inject(_req(2, N_LAYERS + 1))
    out3 = ad3.decode(7, 30, {}, cancel_event=ev)
    assert out3 == [7, 50] and ad3.last_stats["finish_reason"] == "cancelled"


def test_first_token_stop_ignore_eos_and_unsupported_features():
    dl = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=False, stream=[999, 5])
    ad = RocmGlm52EngineAdapter(_gen(dl), with_mtp=False, pure_tp8=True, fp8_ki=False)
    ad.inject(_req(2, N_LAYERS + 1))
    assert ad.decode(999, 10, {}) == [] and ad.last_stats["finish_reason"] == "stop"
    dl2 = FakeShowHands(num_mtp=0, pure_tp8=True, fp8_ki=False, stream=[999, 5])
    ad2 = RocmGlm52EngineAdapter(_gen(dl2), with_mtp=False, pure_tp8=True, fp8_ki=False)
    ad2.inject(_req(2, N_LAYERS + 1))
    assert ad2.decode(999, 3, {"ignore_eos": True}) == [999, 999, 5]
    assert ad.prepare_grammar(None) is None
    with pytest.raises(GrammarBackendUnavailable):
        ad.prepare_grammar({"type": "regex", "value": "a+"})
    with pytest.raises(NotImplementedError):
        ad.decode(7, 3, {"repetition_penalty": 1.2})
    assert not ad.supports_logprobs() and not ad.supports_penalties() and ad.supports_ignore_eos()
