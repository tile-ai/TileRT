"""``ReplyStream``: the five transformations, driven directly.

No HTTP, no asyncio, no vLLM — a list of token ids in, a list of emissions out.
That is the point of the module: the invariants that used to have to hold at
every point either response path emitted something are properties of one object
here, so they can be stated once and checked once.

The properties that matter, and why:

* **A token's logprob entry travels with the emission carrying its text.** Every
  earlier attempt attached entries at the emit site, and every emit site added
  was another chance to attach the wrong ones or none.
* **``completion_tokens`` counts what the reply contains.** A stop removes text,
  and the tokens whose text it removed are not in the reply.
* **Chunking does not matter.** The same tokens fed one at a time, in pairs, or
  all at once produce the same emissions — which is what makes the streaming and
  non-streaming replies to one request agree.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest tests/pd_vllm/test_reply.py -v
"""

import pytest

from tilert.pd_vllm.logprobs import (
    LOGPROB_UNAVAILABLE,
    LogprobsRequest,
)
from tilert.pd_vllm.reply import (
    CONTENT,
    REASONING,
    TOOL_CALL,
    ReplyStream,
    as_logprobs,
)

# "alpha " / "STOP" / " beta", plus the pieces the interesting cases need:
# 13+14 spell the stop across two tokens, 17 carries text AND the stop, and 15
# is a special that spells out only when specials are kept.
_VOCAB = {
    10: "alpha ",
    11: "STOP",
    12: " beta",
    13: "ST",
    14: "OP",
    16: "abc",
    17: " abcSTOP",
    20: "|",
    21: "R",
    22: "|C",
}
_SPECIAL = {15: "<|s|>"}
# 18 alone decodes to the replacement character; 18+19 decode to "。STOP".
_PAIRS = {(18, 19): "。STOP"}


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        out, k, ids = [], 0, list(ids)
        while k < len(ids):
            pair = tuple(ids[k : k + 2])
            if pair in _PAIRS:
                out.append(_PAIRS[pair])
                k += 2
                continue
            i = ids[k]
            if i in _SPECIAL:
                if not skip_special_tokens:
                    out.append(_SPECIAL[i])
            elif i in (18, 19):
                out.append("�")
            else:
                out.append(_VOCAB.get(i, ""))
            k += 1
        return "".join(out)


def _drive(ids, *, stop=(), include=False, logprobs=False, session=None, batch=None):
    """Run the assembler and return (emissions, assembler)."""
    req = LogprobsRequest(top_n=1) if logprobs else None
    asm = ReplyStream(
        _Tok(), stop=stop, include_stop_in_output=include, parser_session=session, logprobs_req=req
    )
    ems = []
    groups = batch or [[i] for i in ids]
    for g in groups:
        ems += asm.push(
            g, [-0.5] * len(g) if logprobs else None, [[(t, -0.5)] for t in g] if logprobs else None
        )
    return ems + asm.finish(), asm


def _content(ems):
    return "".join(e.text for e in ems if e.channel == CONTENT)


def _entries(ems):
    return [x for e in ems if e.channel == CONTENT for x in e.logprobs]


# --------------------------------------------------------------------------- #
# Text: where the cut lands
# --------------------------------------------------------------------------- #
def test_no_stop_passes_everything_through():
    ems, asm = _drive([10, 12])
    assert _content(ems) == "alpha  beta"
    assert asm.completion_tokens == 2
    assert asm.stop_reason is None


@pytest.mark.parametrize(
    "ids,want_text,why",
    [
        ([10, 11, 12], "alpha ", "the stop is its own token"),
        ([16, 13, 14], "abc", "the stop spans two tokens"),
        ([17], " abc", "the stop begins inside a token, whose prefix survives"),
        ([16, 15, 11], "abc", "a stripped special sits before the stop"),
        ([16, 18, 19], "abc", "a byte fragment is absorbed by the stop"),
    ],
)
def test_where_a_stop_cuts(ids, want_text, why):
    stop = ["。STOP"] if ids == [16, 18, 19] else ["STOP"]
    ems, asm = _drive(ids, stop=stop)
    assert _content(ems) == want_text, why
    assert asm.stop_reason == stop[0]


@pytest.mark.parametrize(
    "ids,stop,want,why",
    [
        (
            [16, 13, 14],
            ["STOP"],
            3,
            "13 and 14 spell the stop; their text is gone " "and they still ran",
        ),
        ([17], ["STOP"], 1, "the stop begins inside the only token"),
        ([16, 18, 19], ["。STOP"], 3, "a byte fragment absorbed by the stop"),
        ([16, 15, 11], ["STOP"], 3, "a stripped special contributes no text"),
        ([10, 12], ["ZZZZ"], 2, "no match at all"),
    ],
)
def test_the_cut_does_not_reduce_the_token_count(ids, stop, want, why):
    """`completion_tokens` counts what ran, not what came back.

    vLLM's is `len()` of its detokeniser's UNtruncated id list -- measured on
    0.25.1, not assumed: a stop of `" abcSTOP"` leaves `output_text` ending at
    `"xy"` while still reporting all twelve ids. Counting only the tokens whose
    text survived would under-report a cost the client is billed for.
    """
    ems, asm = _drive(ids, stop=stop)
    assert asm.completion_tokens == want, why


def test_tokens_arriving_after_the_stop_are_not_counted():
    """The reply ended at the stop.

    The node kept generating only because it cannot see text, and the router cancels it -- those
    tokens are not part of what was asked for, and in vLLM they would never have been generated.
    """
    ems, asm = _drive([10, 11, 12, 12, 12], stop=["STOP"])
    assert _content(ems) == "alpha "
    assert asm.completion_tokens == 2, "10 and the token that completed the stop"


def test_include_stop_str_in_output_keeps_the_stop_and_its_token():
    ems, asm = _drive([10, 11, 12], stop=["STOP"], include=True)
    assert _content(ems) == "alpha STOP"


def test_nothing_is_held_back_when_the_stop_stays_in_the_output():
    """There is nothing to remove from released text, so nothing to hold it for.

    vLLM computes its hold-back under the same condition. Without this, every
    delta of an `include_stop_str_in_output` request is delayed by up to
    len(stop)-1 characters for no reason.
    """
    from tilert.pd_vllm.stop_strings import StopWindow

    kept = StopWindow(["STOP"], True)
    kept.push("hi ST")
    assert kept.take() == "hi ST"
    cut = StopWindow(["STOP"], False)
    cut.push("hi ST")
    assert cut.take() == "hi"


def test_an_unmatched_stop_releases_the_held_tail():
    """The tracker holds back what could still become a stop; it must come out."""
    ems, asm = _drive([10, 12], stop=["ZZZZ"])
    assert _content(ems) == "alpha  beta"
    assert asm.stop_reason is None


def test_a_byte_fragment_whose_character_survives_is_kept():
    """Byte-level BPE splits "。" across both ids, so both contributed it."""
    ems, asm = _drive([16, 18, 19], stop=["STOP"])
    assert _content(ems) == "abc。", "the stop is cut; the character is not"


def test_the_finish_reason_is_overridden_only_by_a_stop():
    _, asm = _drive([10, 12], stop=["ZZZZ"])
    assert asm.finish_reason("length") == "length"
    _, asm = _drive([10, 11], stop=["STOP"])
    assert asm.finish_reason("length") == "stop"


# --------------------------------------------------------------------------- #
# Logprobs: one entry per token the reply contains, no more and no fewer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ids,stop,why",
    [
        ([10, 12], (), "no stop"),
        ([10, 11, 12], ("STOP",), "the stop is its own token"),
        ([16, 13, 14], ("STOP",), "the stop spans two tokens"),
        ([17], ("STOP",), "the stop begins inside a token"),
        ([16, 15, 11], ("STOP",), "a stripped special is kept"),
        ([15, 11], ("STOP",), "a kept token produced no visible text at all"),
        ([10, 12], ("ZZZZ",), "the tail is released at the end"),
    ],
)
def test_entries_match_the_token_count(ids, stop, why):
    ems, asm = _drive(ids, stop=stop, logprobs=True)
    assert len(_entries(ems)) == asm.completion_tokens, (
        f"{why}: {len(_entries(ems))} entries for " f"{asm.completion_tokens} tokens"
    )


def test_no_entries_when_logprobs_were_not_requested():
    ems, _ = _drive([10, 12], logprobs=False)
    assert _entries(ems) == []


def test_the_first_token_takes_its_logprob_from_the_prefill():
    """The decode node echoed it rather than sampling, so it sends null."""
    asm = ReplyStream(
        _Tok(), logprobs_req=LogprobsRequest(top_n=1), first_token_logprob=(-0.125, [(10, -0.125)])
    )
    ems = asm.push([10, 12], [None, -0.5], [[], [(12, -0.5)]]) + asm.finish()
    entries = _entries(ems)
    assert len(entries) == 2
    assert entries[0]["logprob"] == pytest.approx(-0.125)


def test_the_first_tokens_candidates_come_from_the_prefill_too():
    """The row describes the PROMPT's last distribution, not the decode token.

    The decode node sends an empty row for that position along with the null
    logprob. Filling the logprob from the prefill entry but leaving the row
    empty would report a token with no alternatives; taking the row from the
    decode position would report the alternatives of a distribution that never
    produced this token.
    """
    asm = ReplyStream(
        _Tok(), logprobs_req=LogprobsRequest(top_n=1), first_token_logprob=(-0.125, [(16, -0.125)])
    )
    ems = asm.push([10, 12], [None, -0.5], [[], [(12, -0.5)]]) + asm.finish()
    row = _entries(ems)[0]["top_logprobs"]
    assert [c["token"] for c in row] == ["abc"], "prefill's alternative id (16)"
    assert row[0]["logprob"] == pytest.approx(-0.125)


def test_the_first_token_without_a_prefill_entry_takes_the_sentinel():
    """No prefill value available -> the documented -9999.0, never a fake one.

    Every other position is still correct, so one sentinel entry beats failing
    the whole completion.
    """
    ems, _ = _drive([10, 12], logprobs=True)
    asm = ReplyStream(_Tok(), logprobs_req=LogprobsRequest(top_n=1))
    ems = asm.push([10, 12], [None, -0.5], [[], [(12, -0.5)]]) + asm.finish()
    assert _entries(ems)[0]["logprob"] == LOGPROB_UNAVAILABLE


def test_the_prefill_value_does_not_leak_onto_a_later_content_token():
    """It belongs to position 0 and to no other token.

    When position 0 goes to `reasoning` the client never sees its entry, and it
    is dropped. What must not happen is the value landing on whichever token
    reached `content` first -- that would report the prompt's last distribution
    as if it described the reply's first content token.
    """
    asm = ReplyStream(
        _Tok(),
        parser_session=_ReasoningThenContent(),
        logprobs_req=LogprobsRequest(top_n=1),
        first_token_logprob=(-0.125, [(16, -0.125)]),
    )
    # 10 -> "alpha " (reasoning), 20 -> "|", 12 -> " beta" (content).
    ems = (
        asm.push([10, 20, 12], [None, -0.25, -0.5], [[], [(20, -0.25)], [(12, -0.5)]])
        + asm.finish()
    )
    entries = _entries(ems)
    assert [e["logprob"] for e in entries] == [
        pytest.approx(-0.5)
    ], "only the content token has an entry, with its own value"


def test_an_entry_can_ride_an_emission_with_no_text():
    """A token that produced no visible text still owes an entry.

    Here 15 is a special the caller strips and 11 is the stop itself; neither
    contributes to `content`, and both were generated.
    """
    ems, asm = _drive([15, 11], stop=["STOP"], logprobs=True)
    assert _content(ems) == ""
    assert len(_entries(ems)) == 2
    assert any(e.channel == CONTENT and not e.text and e.logprobs for e in ems)


# --------------------------------------------------------------------------- #
# The property both response paths rest on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("stop", [(), ("STOP",), ("ZZZZ",)])
def test_chunking_changes_nothing(size, stop):
    ids = [10, 13, 14, 12]
    whole, a1 = _drive(ids, stop=stop, logprobs=True, batch=[ids])
    groups = [ids[i : i + size] for i in range(0, len(ids), size)]
    part, a2 = _drive(ids, stop=stop, logprobs=True, batch=groups)
    assert _content(whole) == _content(part)
    assert len(_entries(whole)) == len(_entries(part))
    assert a1.completion_tokens == a2.completion_tokens
    assert a1.stop_reason == a2.stop_reason


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
class _AllReasoning:
    def feed(self, text):
        return [{"kind": "reasoning", "text": text}]

    def finish(self):
        return []


class _ReasoningThenContent:
    """Everything before "|" is reasoning, everything after is content."""

    def __init__(self):
        self._switched = False

    def feed(self, text):
        out = []
        for part in text.split("|"):
            out.append({"kind": "content" if self._switched else "reasoning", "text": part})
            self._switched = self._switched or "|" in text
        return [e for e in out if e["text"]]

    def finish(self):
        return []


class _AllContent:
    def feed(self, text):
        return [{"kind": "content", "text": text}]

    def finish(self):
        return []


class _OneToolCall:
    def feed(self, text):
        return [{"kind": "tool", "index": 0, "id": "call_x", "name": "f", "arguments": '{"a":1}'}]

    def finish(self):
        return []


def test_stop_with_a_parser_and_logprobs_is_refused_not_guessed():
    """The combination this design refuses, asserted at the object that cannot
    serve it.

    With a stop configured the matcher releases text in chunks of its own, so one
    chunk can span the end of a reasoning segment and the parser answers with a
    reasoning event followed by a content event. There is then no signal saying
    which of the chunk's tokens produced which: the parser reports no difference
    between buffering an ambiguous marker prefix and consuming a complete marker.
    Dropping the entry under-reports; keeping it puts a consumed marker's token
    text into `logprobs.content`.

    Refusing is the design decision. `refuse_unattributable_logprobs` answers 501
    before any backend work; this guard catches a routing bug that got past it,
    where 502 is the honest answer.
    """
    with pytest.raises(ValueError, match="attributable"):
        ReplyStream(
            _Tok(),
            stop=["ZZZZ"],
            parser_session=_ReasoningThenContent(),
            logprobs_req=LogprobsRequest(top_n=1),
        )


@pytest.mark.parametrize(
    "stop,session,lp,why",
    [
        ((), None, True, "no parser, no stop"),
        (("ZZZZ",), None, True, "no parser: the reply IS content"),
        ((), "session", True, "parser without stop: nothing is held back"),
        (("ZZZZ",), "session", False, "parser with stop but nothing to attribute"),
    ],
)
def test_every_other_combination_is_served(stop, session, lp, why):
    """Only the one row is refused. The other three need no offset arithmetic --

    see the table in the module docstring.
    """
    ReplyStream(
        _Tok(),
        stop=stop,
        parser_session=_AllContent() if session else None,
        logprobs_req=LogprobsRequest(top_n=1) if lp else None,
    )


def test_reasoning_logprobs_never_reach_the_content_channel():
    """logprobs cover message.content alone."""
    ems, asm = _drive([10, 12], logprobs=True, session=_AllReasoning())
    assert _content(ems) == ""
    assert "".join(e.text for e in ems if e.channel == REASONING) == "alpha  beta"
    assert _entries(ems) == []
    assert asm.completion_tokens == 2, "the tokens were still generated"


def test_a_tool_call_arrives_whole():
    """The parser emits each index once, so a non-streaming caller can collect
    them without reassembling arguments.
    """
    ems, _ = _drive([10], session=_OneToolCall())
    calls = [e.tool_call for e in ems if e.channel == TOOL_CALL]
    assert calls == [{"index": 0, "id": "call_x", "name": "f", "arguments": '{"a":1}'}]


def test_emissions_are_immutable():
    """A caller cannot rewrite an emission's logprobs after the fact."""
    ems, _ = _drive([10], logprobs=True)
    with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError
        ems[0].logprobs = []


def test_as_logprobs_shapes_the_envelope():
    assert as_logprobs([{"token": "x"}]) == {"content": [{"token": "x"}], "refusal": None}


def test_finish_is_idempotent():
    asm = ReplyStream(_Tok(), stop=["ZZZZ"])
    asm.push([10, 12])
    first = asm.finish()
    assert first and asm.finish() == []


def test_a_split_multibyte_tokens_entry_is_held_not_dropped():
    """ "No text yet" is not "text the parser sent elsewhere".

    18 decodes to nothing on its own; 18+19 decode to "。STOP". So token 18
    produces no visible text and its character arrives with 19. Dropping its
    entry there loses an entry for a token whose text does reach `content`;
    holding it attributes both to whichever channel 19's text goes to.
    """
    asm = ReplyStream(_Tok(), parser_session=_AllContent(), logprobs_req=LogprobsRequest(top_n=1))
    ems = (
        asm.push([16, 18, 19], [-0.1, -0.2, -0.3], [[(16, -0.1)], [(18, -0.2)], [(19, -0.3)]])
        + asm.finish()
    )
    assert _content(ems) == "abc。STOP"
    assert [pytest.approx(e["logprob"]) for e in _entries(ems)] == [
        pytest.approx(-0.1),
        pytest.approx(-0.2),
        pytest.approx(-0.3),
    ], "all three tokens contributed text that reached content"


def test_a_held_entry_still_follows_its_text_to_reasoning():
    """Held is not "kept": the token that completes the text decides.

    Fixing the held case must not undo the reason entries are dropped at all --
    a reasoning token's entry must never reach `content`.
    """
    asm = ReplyStream(_Tok(), parser_session=_AllReasoning(), logprobs_req=LogprobsRequest(top_n=1))
    ems = asm.push([18, 19], [-0.2, -0.3], [[(18, -0.2)], [(19, -0.3)]]) + asm.finish()
    assert _content(ems) == ""
    assert _entries(ems) == [], "the text went to reasoning, so neither is owed"


def test_a_terminal_byte_fragment_still_reaches_the_reply():
    """18 alone: generation stopped mid-character, and that is not nothing.

    The tokenizer's own decode of `[18]` is the replacement character, and the
    blocking path used to produce it by decoding the whole id list. Holding it
    inside the detokeniser and never flushing dropped a character the reply had,
    while `token_ids`, `completion_tokens` and the entry all still counted the
    token.
    """
    asm = ReplyStream(_Tok(), parser_session=_AllContent(), logprobs_req=LogprobsRequest(top_n=1))
    ems = asm.push([18], [-0.2], [[(18, -0.2)]]) + asm.finish()
    assert _content(ems) == "\ufffd", "what the tokenizer itself decodes"
    assert len(_entries(ems)) == 1, "and the token that produced it is described"
    assert asm.completion_tokens == 1


def test_a_split_multibyte_tokens_entry_waits_for_its_character():
    """`ends_at` is where a token's text is COMPLETE, not where it was queued.

    18 decodes to nothing on its own; 18+19 decode to "。". Marking 18 complete
    at the current offset makes its entry due before the character is out, so it
    rides an empty chunk while the chunk carrying "。" describes one token too
    few. Both entries belong with the character.
    """
    # No stop, so nothing is held back and every token's text goes out at once.
    # That is what exposes it: with a hold-back the entry is delayed anyway.
    ems, _ = _drive([16, 18, 19], logprobs=True)
    for e in ems:
        if e.channel == CONTENT and e.logprobs and not e.text:
            raise AssertionError("an entry rode an emission with no text")
    assert len(_entries(ems)) == 3, "all three tokens are still described"
    assert _content(ems) == "abc。STOP", "18+19 spell 。STOP in the stub vocab"
    # Not asserted: that the entries' `token` strings rebuild the emission's
    # text. A byte fragment decodes to U+FFFD on its own, which is why the
    # OpenAI schema carries `bytes` at all -- so this cannot hold for a split
    # character, and requiring it would pin a property the format disclaims.


def test_the_terminal_flush_agrees_with_decoding_the_whole_id_list():
    """The property the flush restores, stated directly.

    Whatever the reply shows must be what the tokenizer makes of the ids the
    reply reports -- for a generation that ends mid-character too. Without the
    flush this was `"abc"` against `"abc\ufffd"`.
    """
    for ids in ([16, 18], [18], [16, 18, 19], [16]):
        asm = ReplyStream(_Tok())
        ems = asm.push(ids) + asm.finish()
        assert _content(ems) == _Tok().decode(asm.token_ids), (
            f"{ids}: reply {_content(ems)!r} vs " f"decode {_Tok().decode(asm.token_ids)!r}"
        )


def test_the_detokenisers_flush_is_idempotent_on_its_own():
    """`ReplyStream.finish()` already guards against a second call, but the
    detokeniser should not depend on that guard to avoid emitting text twice --
    a second caller would otherwise duplicate the terminal character.
    """
    from tilert.pd_vllm.oai_parser import IncrementalDetok

    d = IncrementalDetok(_Tok(), skip_special_tokens=True)
    assert d.push([18]) == "", "held: the window ends mid-character"
    assert d.finish() == "\ufffd"
    assert d.finish() == "", "and not a second time"


def test_a_stop_cut_tokens_entry_rides_its_surviving_prefix():
    """17 decodes to " abcSTOP"; the stop begins inside it.

    `ends_at` points past the untruncated token, so the offset comparison alone
    deferred the entry to a later empty chunk while the visible `" abc"` went out
    describing nothing. Once the window has stopped no more text can arrive, so
    the token is as visible as it will ever be.
    """
    ems, asm = _drive([17], stop=["STOP"], logprobs=True)
    assert _content(ems) == " abc"
    carrying = [e for e in ems if e.channel == CONTENT and e.logprobs]
    assert len(carrying) == 1, "exactly one emission owns the entry"
    assert carrying[0].text == " abc", "and it is the one with the text"
    assert asm.completion_tokens == 1


@pytest.mark.parametrize(
    "ids,stop,want_text,why",
    [
        ([17], ["STOP"], " abc", "the stop begins inside a token"),
        ([16, 13, 14], ["STOP"], "abc", "the stop spans two tokens"),
        ([10, 11, 12], ["STOP"], "alpha ", "the stop is its own token"),
    ],
)
def test_no_entry_rides_an_empty_chunk_when_text_survived(ids, stop, want_text, why):
    """An entry on an empty chunk is only right when NO token produced visible
    text -- a stripped special, or a stop that consumed everything. Whenever some
    text survived, every entry belongs with text.
    """
    ems, asm = _drive(ids, stop=stop, logprobs=True)
    assert _content(ems) == want_text, why
    orphans = [e for e in ems if e.channel == CONTENT and e.logprobs and not e.text]
    assert not orphans, f"{why}: {len(orphans)} entries rode an empty chunk"
    assert len(_entries(ems)) == asm.completion_tokens


def test_an_emission_never_ends_inside_a_token_whose_entry_is_still_due():
    """Measured on a live pair before it was a test.

    The window's holdback is a CHARACTER count, so it lands mid-token: "abc" +
    "alpha " with a 4-character stop makes 6 characters visible, three of them
    "alpha "'s. Those went out describing nothing, and when the stop then matched
    at exactly that cursor there was no text left for the entry to ride:

        delta=' *'  entries=[]                        <- hardware, chunk 9
        delta=''    entries=[' **', 'An', 'alyze']    <- hardware, chunk 10

    Rounding the ceiling down to a token boundary makes the surviving prefix and
    its entry the same emission. The two fully-cut tokens keep riding it: their
    own text is gone, which is the one case an entry has nowhere else to go.
    """
    ems, asm = _drive([16, 10, 11], stop=["ha S"], logprobs=True)
    assert _content(ems) == "abcalp"
    orphans = [e for e in ems if e.channel == CONTENT and e.logprobs and not e.text]
    assert not orphans, f"{len(orphans)} entries rode an empty chunk"
    texted = [e for e in ems if e.channel == CONTENT and e.text]
    assert texted[-1].logprobs, "the last chunk with text carries its entries"
    assert asm.stop_reason == "ha S"
    assert asm.completion_tokens == 3
