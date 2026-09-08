"""OpenAI-semantics parser adapter over vLLM's parser engine (decision B1).

Wraps ``vllm.parser`` (the NEW engine architecture in vllm >= 0.24; the old
``ReasoningParser``/``ToolParserManager`` API is superseded) into the
small surface the router needs:

  parser = make_parser("glm47", tokenizer, thinking=True)
  parsed  = parser.parse_complete(text)          # non-streaming
  sess    = parser.stream()                      # per-request streaming
  events  = sess.feed(delta_text); sess.finish() # normalized event dicts

Runs in the ROUTER environment only — that env must have vllm installed
(CPU-only import is fine; verified with CUDA_VISIBLE_DEVICES=""). The decode
node never imports vllm.

Verified against real AIME transcripts (prefilled-<think> convention, incl. a
135K-char truncated-thinking sample) and template-format tool calls with
random-delta streaming fuzz. Key engine semantics (cost a bug to learn):
``TOOL_NAME`` is an incremental chunk event — fragments must be concatenated.
"""

import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("pd_vllm.oai_parser")


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON string (OpenAI convention)

    @property
    def id(self) -> str:  # noqa: A003
        """Alias of ``call_id`` under the OpenAI field name."""
        return self.call_id

    def to_openai(self, index: int) -> dict:
        return {
            "index": index,
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Parsed:
    reasoning_content: str | None
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# family -> (config-builder import path, arg-converter import path). The
# glm47_moe parser engine uses the vllm.parser API shape (a `*_config(thinking)`
# builder + a `_*_arg_converter(raw, partial)`); the adapter picks the engine
# by family name, so another family with the same shape is one table entry.
_FAMILIES = {
    "glm47": ("vllm.parser.glm47_moe", "glm47_moe_config", "_glm47_arg_converter"),
}


def make_parser(family: str, tokenizer, thinking: bool = True) -> "OaiParser":
    if family not in _FAMILIES:
        raise KeyError(f"unknown parser family {family!r}; " f"known: {sorted(_FAMILIES)}")
    return OaiParser(family, tokenizer, thinking)


class OaiParser:
    """Family-parameterized parser; one instance per model, ``stream()`` per request.

    Family is a vllm.parser engine (glm47).
    """

    def __init__(self, family: str, tokenizer, thinking: bool = True):
        import importlib

        from vllm.parser.engine.events import EventType
        from vllm.parser.engine.streaming_parser_engine import (
            StreamingParserEngine,
        )

        mod_name, cfg_name, conv_name = _FAMILIES[family]
        mod = importlib.import_module(mod_name)
        self._family = family
        self._cfg_fn = getattr(mod, cfg_name)
        self._Engine = StreamingParserEngine
        self._ET = EventType
        self._config = self._cfg_fn(thinking=thinking)
        self._convert = getattr(mod, conv_name)
        self._tok = tokenizer

    def with_thinking(self, thinking: bool) -> "OaiParser":
        if thinking == (self._config.initial_state.name == "REASONING"):
            return self
        clone = object.__new__(OaiParser)
        clone.__dict__.update(self.__dict__)
        clone._config = self._cfg_fn(thinking=thinking)
        return clone

    # ── non-streaming ────────────────────────────────────────────────────
    def parse_complete(self, text: str) -> Parsed:
        engine = self._Engine(self._config, self._tok)
        return self._reduce(engine.parse_complete(text))

    def _reduce(self, events) -> Parsed:
        ET = self._ET
        reasoning, content = [], []
        slots: dict[int, dict] = {}
        for e in events:
            if e.type == ET.REASONING_CHUNK:
                reasoning.append(e.value)
            elif e.type == ET.TEXT_CHUNK:
                content.append(e.value)
            elif e.type in (ET.TOOL_NAME, ET.ARG_VALUE_CHUNK):
                s = slots.setdefault(e.tool_index, {"name": [], "args": []})
                s["name" if e.type == ET.TOOL_NAME else "args"].append(e.value)
        calls = []
        for i in sorted(slots):
            name = "".join(slots[i]["name"]).strip()
            if not name:
                continue  # unnamed fragment (heavy truncation) — drop
            raw = "".join(slots[i]["args"])
            calls.append(ToolCall(_new_call_id(), name, self._convert(raw, True)))
        r = "".join(reasoning)
        c = "".join(content)
        return Parsed(r if r else None, c if c else None, calls)

    # ── streaming ────────────────────────────────────────────────────────
    def stream(self) -> "OaiStream":
        return OaiStream(self)


class OaiStream:
    """Per-request streaming session.

    ``feed``/``finish`` return normalized event dicts:
      {"kind": "reasoning", "text": ...}
      {"kind": "content",   "text": ...}
      {"kind": "tool", "index": i, "id": ..., "name": ..., "arguments": ...}

    Reasoning/content stream through per delta. Tool calls are buffered and
    emitted whole at TOOL_CALL_END (OpenAI clients accept arguments in any
    fragmentation; whole-call emission sidesteps XML→JSON incremental
    conversion). ``finish`` flushes a truncated trailing tool call with
    partial-args conversion.
    """

    def __init__(self, parent: "OaiParser"):
        self._p = parent
        self._engine = parent._Engine(parent._config, parent._tok)
        self._slots: dict[int, dict] = {}
        self._emitted: set[int] = set()

    def feed(self, delta_text: str) -> list[dict]:
        if not delta_text:
            return []
        return self._consume(self._engine.feed(delta_text, []))

    def finish(self) -> list[dict]:
        out = self._consume(self._engine.finish())
        # flush truncated trailing tool call (never saw TOOL_CALL_END)
        for i in sorted(self._slots):
            if i in self._emitted:
                continue
            ev = self._flush_tool(i, partial=True)
            if ev:
                out.append(ev)
        return out

    def _consume(self, events) -> list[dict]:
        ET = self._p._ET
        out: list[dict] = []
        for e in events:
            if e.type == ET.REASONING_CHUNK:
                out.append({"kind": "reasoning", "text": e.value})
            elif e.type == ET.TEXT_CHUNK:
                out.append({"kind": "content", "text": e.value})
            elif e.type in (ET.TOOL_NAME, ET.ARG_VALUE_CHUNK):
                s = self._slots.setdefault(e.tool_index, {"name": [], "args": []})
                s["name" if e.type == ET.TOOL_NAME else "args"].append(e.value)
            elif e.type == ET.TOOL_CALL_END:
                ev = self._flush_tool(e.tool_index, partial=False)
                if ev:
                    out.append(ev)
        return out

    def _flush_tool(self, index: int, partial: bool) -> dict | None:
        s = self._slots.get(index)
        if s is None or index in self._emitted:
            return None
        name = "".join(s["name"]).strip()
        if not name:
            return None
        self._emitted.add(index)
        args = self._p._convert("".join(s["args"]), partial)
        return {
            "kind": "tool",
            "index": index,
            "id": _new_call_id(),
            "name": name,
            "arguments": args,
        }


class IncrementalDetok:
    r"""Incremental token→text for byte-level BPE tokenizers.

    Decodes a bounded trailing window; holds output while the window ends in
    a partial multi-byte sequence (\ufffd). Window folding is safe for
    byte-level BPE: separate windows decode to concatenable byte streams.

    ``finish`` releases what is held. A generation can end mid-character -- EOS
    or ``max_tokens`` after the first byte of two -- and the ids are reported
    either way, so without the release the reply would omit a character the
    tokenizer makes of the ids it reports.

    Specials are kept by default — a parser consumes </think> and friends, and
    the stop token never reaches the stream because the engine adapter
    suppresses it. A caller with no parser passes True instead: nothing
    downstream would consume a special, so one would surface as content. The
    policy is a constructor argument rather than a per-call one so a request
    cannot be matched against one spelling of its own output and shown another.
    """

    _FOLD = 256

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self._tok = tokenizer
        self._skip = skip_special_tokens
        self._ids: list[int] = []
        self._emitted = 0
        self._holding = False

    @property
    def holding(self) -> bool:
        """Whether the last push produced nothing because it was incomplete.

        Distinguishes a byte fragment — whose text arrives with a later token —
        from a token that genuinely decodes to nothing, such as a special the
        caller strips. Both return an empty delta, and a caller pairing
        per-token metadata against the text has to tell them apart.
        """
        return self._holding

    def finish(self) -> str:
        """Whatever ``push`` held back because the window ended mid-character.

        Generation can stop between the byte-level tokens of one character -- EOS
        or ``max_tokens`` arriving after its first byte -- and the tokenizer's own
        decode of the complete id list then ends in the replacement character.
        Dropping it loses a character the reply had, while ``token_ids``, usage
        and the logprob entry all still count the token.
        """
        # Idempotent through `_emitted`, which the first call advances to the
        # end; the `_holding` early-out is a shortcut, not the guarantee.
        if not self._holding:
            return ""
        text = self._tok.decode(self._ids, skip_special_tokens=self._skip)
        delta = text[self._emitted :]
        self._emitted = len(text)
        self._holding = False
        return delta  # noqa: R504 (self._emitted mutated after delta is computed)

    def push(self, ids: list[int]) -> str:
        self._ids.extend(ids)
        text = self._tok.decode(self._ids, skip_special_tokens=self._skip)
        if text.endswith("\ufffd"):
            self._holding = True
            return ""
        self._holding = False
        delta = text[self._emitted :]
        self._emitted = len(text)
        if len(self._ids) > self._FOLD:
            self._ids = []
            self._emitted = 0
        return delta  # noqa: R504 (self._emitted mutated after delta is computed)
