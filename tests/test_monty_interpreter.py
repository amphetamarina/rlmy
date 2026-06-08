"""
Tests for rlmy's MontyInterpreter.

Focus: the CodeInterpreter contract (execute returns/raises), tool dispatch,
SUBMIT shaping, in-process state persistence, and disk-backed snapshotting
(the headline feature — session survives a fresh interpreter / process).

No LLM, no MCP, no Deno — pure interpreter behavior against pydantic-monty.
"""

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from rlmy.agent.interpreters import MontyInterpreter


class TestExecuteContract:
    def test_print_output_returned_as_string(self):
        interp = MontyInterpreter()
        assert interp.execute("print(1 + 2)") == "3"

    def test_no_output_returns_none(self):
        interp = MontyInterpreter()
        assert interp.execute("x = 5") is None

    def test_syntax_error_raises_syntaxerror(self):
        interp = MontyInterpreter()
        with pytest.raises(SyntaxError):
            interp.execute("def (")

    def test_runtime_error_raises_codeinterpretererror(self):
        interp = MontyInterpreter()
        with pytest.raises(CodeInterpreterError):
            interp.execute("undefined_name")

    def test_execution_halts_at_raising_tool(self):
        # A raising tool must stop the snippet — later lines do NOT run.
        ran_after = []

        def boom():
            raise ValueError("tool blew up")

        def after():
            ran_after.append(True)
            return "nope"

        interp = MontyInterpreter(tools={"boom": boom, "after": after})
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute('print("starting...")\nboom()\nafter()')
        assert "tool blew up" in str(exc.value)
        assert ran_after == [], "code after the raising tool should not execute"

    def test_partial_output_preserved_in_error(self):
        # Output printed BEFORE the exception is surfaced alongside the error,
        # so the LLM sees how far it got, not just the failure.
        def boom():
            raise ValueError("kaboom")

        interp = MontyInterpreter(tools={"boom": boom})
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute('print("starting...")\nboom()')
        msg = str(exc.value)
        assert "starting..." in msg, "pre-exception print should be preserved"
        assert "kaboom" in msg, "error message should be present"

    def test_inline_error_also_preserves_partial_output(self):
        interp = MontyInterpreter()
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute('print("before")\nx = 1 / 0')
        msg = str(exc.value)
        assert "before" in msg
        assert "ZeroDivisionError" in msg or "division" in msg

    def test_inputs_injected_as_variables(self):
        interp = MontyInterpreter()
        out = interp.execute("print(n * 2)", variables={"n": 21})
        assert out == "42"


class TestStatePersistsInProcess:
    def test_variable_persists_across_calls(self):
        interp = MontyInterpreter()
        interp.execute("acc = 100")
        interp.execute("acc = acc + 23")
        assert interp.execute("print(acc)") == "123"

    def test_function_persists_across_calls(self):
        interp = MontyInterpreter()
        interp.execute("def double(x):\n    return x * 2")
        assert interp.execute("print(double(21))") == "42"


class TestTools:
    def test_tool_callable_from_code(self):
        seen = []

        def lookup(key: str) -> str:
            seen.append(key)
            return f"VAL[{key}]"

        interp = MontyInterpreter(tools={"lookup": lookup})
        out = interp.execute("print(lookup(key='alpha'))")
        assert out == "VAL[alpha]"
        assert seen == ["alpha"]

    def test_tools_property_is_mutable_for_rlm_injection(self):
        # RLM injects its own tools by calling .update() on this dict.
        interp = MontyInterpreter()
        interp.tools.update({"greet": lambda name: f"hi {name}"})
        assert interp.execute("print(greet(name='bob'))") == "hi bob"


class TestSubmit:
    def test_submit_kwargs(self):
        interp = MontyInterpreter(output_fields=[{"name": "answer"}])
        result = interp.execute("SUBMIT(answer='done')")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "done"}

    def test_submit_positional_maps_to_output_fields(self):
        interp = MontyInterpreter(output_fields=[{"name": "answer"}])
        result = interp.execute("SUBMIT('hello')")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "hello"}

    def test_submit_passthrough_dict(self):
        interp = MontyInterpreter(output_fields=[{"name": "answer"}])
        result = interp.execute("SUBMIT({'answer': 7})")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": 7}

    def test_submit_positional_count_mismatch_raises_helpful_error(self):
        # Too many positionals for the declared fields → educate, don't silently drop.
        interp = MontyInterpreter(output_fields=[{"name": "answer"}])
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute("SUBMIT('a', 'b')")
        msg = str(exc.value)
        assert "2 positional" in msg and "1 field" in msg


class TestSnapshotting:
    def test_session_survives_fresh_interpreter(self, tmp_path):
        """The headline behavior: a new interpreter on the same path resumes state."""
        snap = tmp_path / "repl.snapshot"

        first = MontyInterpreter(snapshot_path=snap)
        first.execute("saved = 'survived'")
        first.execute("def helper(x):\n    return x + 1")
        assert snap.exists()

        # Simulate a process restart: brand-new object, same snapshot path.
        second = MontyInterpreter(snapshot_path=snap)
        assert second.execute("print(saved)") == "survived"
        assert second.execute("print(helper(41))") == "42"

    def test_no_snapshot_path_means_no_file_and_no_cross_instance_state(self):
        a = MontyInterpreter()
        a.execute("ghost = 1")
        b = MontyInterpreter()
        with pytest.raises(CodeInterpreterError):
            b.execute("print(ghost)")

    def test_corrupt_snapshot_starts_fresh_not_crash(self, tmp_path):
        snap = tmp_path / "repl.snapshot"
        snap.write_bytes(b"not a valid monty snapshot")
        interp = MontyInterpreter(snapshot_path=snap)
        # Should not raise on load; just runs against a fresh REPL.
        assert interp.execute("print('ok')") == "ok"

    def test_tools_are_fresh_each_restart_not_serialized(self, tmp_path):
        snap = tmp_path / "repl.snapshot"

        first = MontyInterpreter(
            snapshot_path=snap, tools={"t": lambda key: f"v1-{key}"}
        )
        first.execute("kept = 'data'")

        # Restart with a DIFFERENT tool implementation.
        second = MontyInterpreter(
            snapshot_path=snap, tools={"t": lambda key: f"V2-{key.upper()}"}
        )
        out = second.execute("print(kept, t(key='x'))")
        assert out == "data V2-X"  # data restored, but NEW tool impl ran

    def test_submit_still_saves_snapshot(self, tmp_path):
        snap = tmp_path / "repl.snapshot"
        interp = MontyInterpreter(snapshot_path=snap, output_fields=[{"name": "answer"}])
        interp.execute("pre = 'before submit'")
        interp.execute("SUBMIT(answer='x')")
        # A later process can still recover state defined before SUBMIT.
        resumed = MontyInterpreter(snapshot_path=snap)
        assert resumed.execute("print(pre)") == "before submit"
