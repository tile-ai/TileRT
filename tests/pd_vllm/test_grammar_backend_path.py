"""Which module the xgrammar host wrapper is loaded from.

The wrapper moved from ``tilert.models.glm_5_2.grammar`` to ``tilert.grammar``
in the engine. One serve version must work against engine wheels from
either side of that move, so both paths are exercised here with stubbed
modules — no engine install required.
"""

from __future__ import annotations

import sys
import types

import pytest

from tilert.pd_vllm.grammar_backend import load_grammar_backend

_NEW = "tilert.grammar"
_OLD = "tilert.models.glm_5_2.grammar"
_PARENTS = ("tilert", "tilert.models", "tilert.models.glm_5_2")


def _stub(monkeypatch, path: str, tag: str) -> None:
    """Install a fake wrapper module at ``path`` whose classes carry ``tag``."""
    for parent in _PARENTS:
        monkeypatch.setitem(
            sys.modules, parent, sys.modules.get(parent) or types.ModuleType(parent)
        )
    mod = types.ModuleType(path)
    for cls_name in ("GrammarEngine", "GrammarSession"):
        setattr(mod, cls_name, type(cls_name, (), {"came_from": tag}))
    monkeypatch.setitem(sys.modules, path, mod)


def _absent(monkeypatch, path: str) -> None:
    """Make importing ``path`` raise, the way a wheel that omits it would."""

    class _Blocker:
        def find_module(self, name, path=None):  # pragma: no cover - legacy hook
            return None

        def find_spec(self, name, path=None, target=None):
            if name == globals()["_blocked"]:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None  # noqa: R501 (finder protocol: None means "not mine")

    globals()["_blocked"] = path
    monkeypatch.delitem(sys.modules, path, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)


def test_the_new_path_is_preferred(monkeypatch):
    _stub(monkeypatch, _NEW, "new")
    _stub(monkeypatch, _OLD, "old")
    engine, session = load_grammar_backend()
    assert engine.came_from == "new"
    assert session.came_from == "new"


def test_an_older_engine_wheel_still_works(monkeypatch):
    _absent(monkeypatch, _NEW)
    _stub(monkeypatch, _OLD, "old")
    engine, session = load_grammar_backend()
    assert engine.came_from == "old"
    assert session.came_from == "old"


def test_with_neither_path_the_error_names_no_model(monkeypatch):
    _absent(monkeypatch, _NEW)
    monkeypatch.delitem(sys.modules, _OLD, raising=False)
    for parent in _PARENTS:
        monkeypatch.delitem(sys.modules, parent, raising=False)
    with pytest.raises(ModuleNotFoundError) as e:
        load_grammar_backend()
    msg = str(e.value)
    assert "xgrammar host backend" in msg
    assert "glm" not in msg.lower(), "the client-visible message must not name another model"
