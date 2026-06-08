"""
Code interpreters for the RLM REPL.

Each interpreter satisfies DSPy's CodeInterpreter protocol (start/execute/
shutdown/tools), so RLM can drive any of them interchangeably. Construction is
the only thing that differs between them — call sites build one via
`make_interpreter(...)` and otherwise stay identical.

    from rlmy.agent.interpreters import make_interpreter, resolve_interpreter_kind
    interp = make_interpreter(resolve_interpreter_kind(), snapshot_path=ws / "repl.snapshot")

Adding a new interpreter (e.g. a remote EC2 sandbox) is: implement the protocol,
register it in _BUILDERS below. No call site changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from rlmy.agent.interpreters.monty import MontyInterpreter

__all__ = [
    "MontyInterpreter",
    "make_interpreter",
    "resolve_interpreter_kind",
    "DEFAULT_INTERPRETER",
    "KNOWN_INTERPRETERS",
    "INTERPRETER_CHOICES",
]

# The default interpreter: pip-only, no Deno, with on-disk snapshotting.
DEFAULT_INTERPRETER = "monty"

# Accepted aliases → canonical kind. "deno"/"pyodide" both mean DSPy's builtin
# Deno/Pyodide PythonInterpreter.
_ALIASES = {
    "monty": "monty",
    "deno": "deno",
    "pyodide": "deno",
}

KNOWN_INTERPRETERS = sorted(set(_ALIASES.values()))

# All accepted names INCLUDING aliases — the single source of truth for CLI
# `--interpreter` choices, so adding an interpreter only touches _ALIASES.
INTERPRETER_CHOICES = sorted(_ALIASES)


def _build_monty(
    snapshot_path: Path | str | None,
    tools: dict[str, Callable[..., Any]] | None,
) -> MontyInterpreter:
    return MontyInterpreter(tools=tools, snapshot_path=snapshot_path)


def _build_deno(
    snapshot_path: Path | str | None,
    tools: dict[str, Callable[..., Any]] | None,
) -> Any:
    # Imported lazily: the Deno/Pyodide interpreter pulls in a heavier stack and
    # is only needed when explicitly selected. It has no snapshot support, so
    # snapshot_path is intentionally ignored (the builtin can't persist state
    # across restarts — see `restores_state` defaulting to False at call sites).
    from dspy.primitives.python_interpreter import PythonInterpreter

    return PythonInterpreter(tools=tools)


# kind → builder. Register new interpreters here; nothing else changes.
_BUILDERS = {
    "monty": _build_monty,
    "deno": _build_deno,
}


def resolve_interpreter_kind(explicit: str | None = None) -> str:
    """Resolve which interpreter to use. Priority: explicit arg > env > default.

    Mirrors config.py's resolution style (env over config over default). The
    `RLM_INTERPRETER` env var lets users pin a choice without a CLI flag.

    Raises:
        ValueError: if the resolved name isn't a known interpreter.
    """
    choice = explicit or os.getenv("RLM_INTERPRETER") or DEFAULT_INTERPRETER
    canonical = _ALIASES.get(choice.strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unknown interpreter '{choice}'. "
            f"Choose one of: {', '.join(KNOWN_INTERPRETERS)} (aliases: pyodide=deno)."
        )
    return canonical


def make_interpreter(
    kind: str | None = None,
    *,
    snapshot_path: Path | str | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> Any:
    """Construct a CodeInterpreter by kind.

    Args:
        kind: "monty" (default) or "deno"/"pyodide". None → resolve_interpreter_kind().
        snapshot_path: Where to persist/restore REPL state. Honored by interpreters
            that support snapshotting (Monty); ignored by those that don't (Deno).
        tools: Initial tool callables (RLM injects its own before each run too).

    Returns:
        An interpreter satisfying DSPy's CodeInterpreter protocol.
    """
    canonical = resolve_interpreter_kind(kind)
    return _BUILDERS[canonical](snapshot_path, tools)
