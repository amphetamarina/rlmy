"""
Tests for the interpreter factory / selection (Unit B).

Focus: resolution priority (arg > env > default), aliases, unknown handling,
and that make_interpreter wires snapshot_path/tools to the right interpreter.
The Deno builder is only constructed when explicitly selected, so most tests
stay on the default Monty path (no Deno dependency required).
"""

import pytest

from rlmy.agent.interpreters import (
    DEFAULT_INTERPRETER,
    KNOWN_INTERPRETERS,
    MontyInterpreter,
    make_interpreter,
    resolve_interpreter_kind,
)


class TestResolveKind:
    def test_default_is_monty(self):
        assert DEFAULT_INTERPRETER == "monty"
        assert resolve_interpreter_kind() == "monty"

    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("RLM_INTERPRETER", "deno")
        assert resolve_interpreter_kind("monty") == "monty"

    def test_env_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("RLM_INTERPRETER", "deno")
        assert resolve_interpreter_kind() == "deno"

    def test_pyodide_aliases_to_deno(self):
        assert resolve_interpreter_kind("pyodide") == "deno"

    def test_case_and_whitespace_insensitive(self):
        assert resolve_interpreter_kind("  MONTY ") == "monty"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_interpreter_kind("nonsense")

    def test_known_interpreters_listed(self):
        assert "monty" in KNOWN_INTERPRETERS
        assert "deno" in KNOWN_INTERPRETERS


class TestMakeInterpreter:
    def test_default_builds_monty(self):
        interp = make_interpreter()
        assert isinstance(interp, MontyInterpreter)

    def test_monty_receives_snapshot_path(self, tmp_path):
        snap = tmp_path / "repl.snapshot"
        interp = make_interpreter("monty", snapshot_path=snap)
        assert isinstance(interp, MontyInterpreter)
        assert interp.restores_state is True  # snapshot path → will restore

    def test_monty_without_snapshot_does_not_restore(self):
        interp = make_interpreter("monty")
        assert interp.restores_state is False

    def test_tools_passed_through(self):
        interp = make_interpreter("monty", tools={"greet": lambda name: f"hi {name}"})
        assert interp.execute("print(greet(name='x'))") == "hi x"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            make_interpreter("ec2-someday")
