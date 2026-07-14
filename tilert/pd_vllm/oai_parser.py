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
# by family name.
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
    a partial multi-byte sequence (\\ufffd). Window folding is safe for
    byte-level BPE: separate windows decode to concatenable byte streams.
    Specials are KEPT (skip_special_tokens=False) — the parser consumes
    </think> etc.; the stop token never reaches the stream (engine adapter
    suppresses it).
    """

    _FOLD = 256

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._emitted = 0

    def push(self, ids: list[int]) -> str:
        self._ids.extend(ids)
        text = self._tok.decode(self._ids, skip_special_tokens=False)
        if text.endswith("�"):
            return ""
        delta = text[self._emitted :]
        self._emitted = len(text)
        if len(self._ids) > self._FOLD:
            self._ids = []
            self._emitted = 0
        return delta  # noqa: R504 (self._emitted mutated after delta is computed)
