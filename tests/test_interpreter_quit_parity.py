"""
Cross-backend parity tests: control-flow signals from host tools.

WHY: When the user types /quit, the `ask_user_guidance` host tool raises
SystemExit. That signal is raised *inside* the code interpreter while a tool
is running. The interpreter must let it ESCAPE as a real SystemExit so it
propagates past the RLM loop (which only catches CodeInterpreterError /
SyntaxError) and exits the process — instead of being swallowed and fed back to
the LLM as a plain "[Error] SystemExit: 0", which would trap the user in a loop.

Monty needed an explicit fix for this (it wraps tool exceptions as
MontyRuntimeError, losing the type — see interpreters/monty.py). The Deno/Pyodide
builtin (dspy.PythonInterpreter) runs host tools in-process, so BaseException
control-flow signals propagate natively. These tests assert BOTH backends behave
identically, locking the parity in as a regression guard:

    - SystemExit from a tool      -> escapes as SystemExit (not wrapped/swallowed)
    - KeyboardInterrupt from tool -> escapes as KeyboardInterrupt
    - ordinary ValueError         -> surfaces as CodeInterpreterError

Deno is not installed everywhere (and the user no longer tests it locally), so
the Deno/Pyodide cases SKIP cleanly when the `deno` binary is unavailable rather
than failing.
"""

import os
import shutil

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from rlmy.agent.interpreters import MontyInterpreter


# ── Deno detection ───────────────────────────────────────────────────────────
#
# Deno may be installed at ~/.deno/bin/deno but not on PATH. dspy launches it as
# the bare command "deno", so if the binary only lives in ~/.deno/bin we put that
# dir on PATH for this test process (guarded by the binary actually existing).
# This is scoped to the test module's process and is acceptable here.
_DENO_HOME_BIN = os.path.expanduser("~/.deno/bin")
_DENO_HOME_DENO = os.path.join(_DENO_HOME_BIN, "deno")

if shutil.which("deno") is None and os.path.exists(_DENO_HOME_DENO):
    os.environ["PATH"] = _DENO_HOME_BIN + os.pathsep + os.environ.get("PATH", "")

DENO_AVAILABLE = shutil.which("deno") is not None or os.path.exists(_DENO_HOME_DENO)

requires_deno = pytest.mark.skipif(
    not DENO_AVAILABLE,
    reason="Deno is not installed (checked PATH and ~/.deno/bin/deno); "
    "skipping Deno/Pyodide parity tests.",
)


# ── Host tools that raise from inside the interpreter ────────────────────────

def _quit_tool():
    # Mirrors ask_user_guidance on /quit.
    raise SystemExit(0)


def _interrupt_tool():
    raise KeyboardInterrupt()


def _boom_tool():
    raise ValueError("real failure")


# ── Monty backend (always runs) ──────────────────────────────────────────────

class TestMontyControlFlow:
    def test_systemexit_escapes(self):
        interp = MontyInterpreter(tools={"t": _quit_tool})
        with pytest.raises(SystemExit) as exc:
            interp.execute("t()")
        # It must be a true SystemExit (a BaseException), NOT a normal Exception
        # the RLM loop would catch and retry.
        assert not isinstance(exc.value, Exception)

    def test_keyboardinterrupt_escapes(self):
        interp = MontyInterpreter(tools={"t": _interrupt_tool})
        with pytest.raises(KeyboardInterrupt) as exc:
            interp.execute("t()")
        assert not isinstance(exc.value, Exception)

    def test_normal_error_wrapped(self):
        interp = MontyInterpreter(tools={"t": _boom_tool})
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute("t()")
        assert "real failure" in str(exc.value)


# ── Deno/Pyodide backend (skips when Deno unavailable) ───────────────────────

@requires_deno
class TestDenoControlFlow:
    @pytest.fixture
    def make_interp(self):
        """Build a PythonInterpreter and guarantee the Deno subprocess is freed."""
        from dspy.primitives.python_interpreter import PythonInterpreter

        created = []

        def _make(tool):
            interp = PythonInterpreter(tools={"t": tool})
            created.append(interp)
            return interp

        yield _make

        for interp in created:
            interp.shutdown()

    def test_systemexit_escapes(self, make_interp):
        # Deno/Pyodide runs the tool in-process, so SystemExit propagates as the
        # real type rather than being wrapped as a CodeInterpreterError.
        interp = make_interp(_quit_tool)
        with pytest.raises(SystemExit) as exc:
            interp.execute("t()")
        assert not isinstance(exc.value, Exception)

    def test_keyboardinterrupt_escapes(self, make_interp):
        interp = make_interp(_interrupt_tool)
        with pytest.raises(KeyboardInterrupt) as exc:
            interp.execute("t()")
        assert not isinstance(exc.value, Exception)

    def test_normal_error_wrapped(self, make_interp):
        # Parity with Monty: an ordinary error still surfaces as a
        # CodeInterpreterError the RLM loop feeds back to the LLM.
        interp = make_interp(_boom_tool)
        with pytest.raises(CodeInterpreterError) as exc:
            interp.execute("t()")
        assert "real failure" in str(exc.value)
