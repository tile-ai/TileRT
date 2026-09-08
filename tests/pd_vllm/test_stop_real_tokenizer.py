"""``stop`` against a real byte-level BPE, rather than a stub vocabulary.

test_stop_strings.py and test_reply.py drive a stub whose every id
decodes to one fixed string. A real tokenizer does not behave that way, and the
differences are exactly what the hold-back and the entry-to-text alignment exist
for:

* a stop string is almost never token-aligned, so the cut usually lands inside a
  token -- and that token's prefix is still part of the reply;
* one character can span two tokens, so the first of them decodes to nothing
  emittable and the second carries the whole character;
* a special decodes to a tag or to nothing depending on the policy, and the
  matcher and the emitted text have to agree about which.

Skipped automatically when no tokenizer is on disk, so a dev box stays green:

  TOKENIZER_PATH=/path/to/tokenizer \
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_stop_real_tokenizer.py -v

No GPU and no weights -- the tokenizer alone.
"""

from __future__ import annotations

import os

import pytest

transformers = pytest.importorskip("transformers")

# Any byte-level BPE tokenizer will do (GLM-5 / DeepSeek-V3.2 checkpoints
# both qualify); the rules under test must not depend on which one.
_PATHS = [p for p in (os.environ.get("TOKENIZER_PATH", ""),) if p and os.path.isdir(p)]
if not _PATHS:
    pytest.skip("no tokenizer on disk", allow_module_level=True)

from tilert.pd_vllm.logprobs import LogprobsRequest  # noqa: E402
from tilert.pd_vllm.reply import (  # noqa: E402
    CONTENT,
    ReplyStream,
)

_ASCII = "The answer is 42.\nObservation: done\nMore text follows here."
# Chosen so the stops below cut inside a token and land on a multi-byte
# character -- neither is possible with a one-id-one-string stub.
_CJK = "今天天气很好。所以我们出去散步了。"


@pytest.fixture(scope="module", params=_PATHS, ids=lambda p: p.split("/")[-1])
def tok(request):
    return transformers.AutoTokenizer.from_pretrained(request.param, trust_remote_code=True)


def _drive(tok, text, stop, *, include=False, chunk=1, logprobs=False, ids=None):
    """Push `text`'s real token ids through an assembler; return the reply."""
    ids = tok.encode(text, add_special_tokens=False) if ids is None else ids
    asm = ReplyStream(
        tok,
        stop=stop,
        include_stop_in_output=include,
        logprobs_req=LogprobsRequest(top_n=1) if logprobs else None,
    )
    ems = []
    for i in range(0, len(ids), chunk):
        g = ids[i : i + chunk]
        ems += asm.push(
            g, [-0.5] * len(g) if logprobs else None, [[(t, -0.5)] for t in g] if logprobs else None
        )
    ems += asm.finish()
    return {
        "text": "".join(e.text for e in ems if e.channel == CONTENT),
        "entries": [x for e in ems if e.channel == CONTENT for x in e.logprobs],
        "asm": asm,
        "ids": ids,
    }


def test_a_multi_token_stop_cuts_at_the_right_character(tok):
    got = _drive(tok, _ASCII, ["Observation:"])
    assert got["text"] == "The answer is 42.\n"
    assert got["asm"].stop_reason == "Observation:"


def test_include_keeps_exactly_the_stop_string(tok):
    got = _drive(tok, _ASCII, ["Observation:"], include=True)
    assert got["text"] == "The answer is 42.\nObservation:"


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8, 64])
def test_chunking_real_tokens_changes_nothing(tok, chunk):
    """MTP delivers several tokens per message, and the batch size varies with acceptance.

    A reply that depended on it would be non-deterministic.
    """
    one = _drive(tok, _ASCII, ["Observation:"], logprobs=True, chunk=1)
    got = _drive(tok, _ASCII, ["Observation:"], logprobs=True, chunk=chunk)
    assert got["text"] == one["text"]
    assert len(got["entries"]) == len(one["entries"])
    assert got["asm"].completion_tokens == one["asm"].completion_tokens


def test_one_entry_per_token_counted(tok):
    got = _drive(tok, _ASCII, ["Observation:"], logprobs=True)
    assert len(got["entries"]) == got["asm"].completion_tokens
    assert got["asm"].completion_tokens < len(
        got["ids"]
    ), "the stream did not run to the end -- the reply stopped earlier"


def test_the_count_covers_the_tokens_the_stop_consumed(tok):
    """A stop string is several tokens long on a real vocabulary, and those
    tokens ran even though their text is gone.

    "Observation:" is three tokens here. Counting only the visible ones would
    bill for less than the model did -- and the visible text is not on a token
    boundary in general, so no count can describe it. vLLM reports the
    untruncated id list for exactly this reason.
    """
    got = _drive(tok, _ASCII, ["Observation:"], logprobs=True)
    n = got["asm"].completion_tokens
    assert tok.decode(got["ids"][:n], skip_special_tokens=True).startswith(
        got["text"]
    ), "the visible text is a prefix of what was counted"
    assert len(tok.decode(got["ids"][:n], skip_special_tokens=True)) > len(
        got["text"]
    ), "and the stop's own tokens are inside the count"


def test_the_reply_is_a_prefix_of_the_full_decode(tok):
    got = _drive(tok, _ASCII, ["Observation:"])
    full = tok.decode(got["ids"], skip_special_tokens=True)
    assert full.startswith(got["text"])


def test_a_stop_that_cuts_inside_a_token(tok):
    """ "。所以" starts mid-token, so the token that begins the stop also holds text the reply keeps.

    Dropping the whole token would lose "好".
    """
    got = _drive(tok, _CJK, ["。所以"])
    assert got["text"] == "今天天气很好"
    assert got["asm"].stop_reason == "。所以"


def test_a_stop_landing_on_a_multi_byte_character(tok):
    got = _drive(tok, _CJK, ["散步"])
    assert got["text"] == "今天天气很好。所以我们出去"


def test_without_a_stop_nothing_is_held_or_cut(tok):
    got = _drive(tok, _ASCII, [])
    assert got["text"] == tok.decode(got["ids"], skip_special_tokens=True)
    assert got["asm"].completion_tokens == len(got["ids"])


def test_an_unmatched_stop_releases_the_tail(tok):
    got = _drive(tok, _ASCII, ["ZZZZZZ"])
    assert got["text"] == tok.decode(got["ids"], skip_special_tokens=True)
    assert got["asm"].stop_reason is None


def test_a_special_does_not_surface_as_content(tok):
    """With no parser nothing downstream would consume it, so the assembler
    strips it -- and the stop matcher has to see the same text, or a stop
    spelled like a tag would end one channel's reply and not the other's.
    """
    ids = tok.encode("hi", add_special_tokens=False) + [tok.eos_token_id]
    got = _drive(tok, "", ["ZZZZZZ"], ids=ids)
    assert got["text"].strip() == "hi"
    assert "<" not in got["text"]
