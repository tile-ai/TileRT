import json
import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("pd_vllm.oai_parser")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self, index: int) -> dict:
        return {
            "index": index,
            "id": self.id,
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


_FAMILIES = {"glm47": ("vllm.parser.glm47_moe", "glm47_moe_config", "_glm47_arg_converter")}


def make_parser(family: str, tokenizer, thinking: bool = True) -> "OaiParser":
    if family not in _FAMILIES:
        raise KeyError(f"unknown parser family {family!r}; known: {sorted(_FAMILIES)}")
    return OaiParser(family, tokenizer, thinking)


class OaiParser:

    def __init__(self, family: str, tokenizer, thinking: bool = True):
        import importlib

        from vllm.parser.engine.events import EventType
        from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

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

    def parse_complete(self, text: str) -> Parsed:
        engine = self._Engine(self._config, self._tok)
        return self._reduce(engine.parse_complete(text))

    def _reduce(self, events) -> Parsed:
        ET = self._ET
        reasoning, content = ([], [])
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
                continue
            raw = "".join(slots[i]["args"])
            calls.append(ToolCall(_new_call_id(), name, self._convert(raw, True)))
        r = "".join(reasoning)
        c = "".join(content)
        return Parsed(r if r else None, c if c else None, calls)

    def stream(self) -> "OaiStream":
        return OaiStream(self)


class OaiStream:

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
    _FOLD = 256

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self._tok = tokenizer
        self._skip = skip_special_tokens
        self._ids: list[int] = []
        self._emitted = 0
        self._holding = False

    @property
    def holding(self) -> bool:
        return self._holding

    def finish(self) -> str:
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
        if text.endswith("�"):
            self._holding = True
            return ""
        self._holding = False
        delta = text[self._emitted :]
        self._emitted = len(text)
        if len(self._ids) > self._FOLD:
            self._ids = []
            self._emitted = 0
        return delta  # noqa: R504 (self._emitted mutated after delta is computed)
