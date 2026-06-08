"""
Purpose: A CodeInterpreter backed by Monty (pydantic-monty), with optional
         on-disk session snapshotting so REPL state survives process restarts.
Usage:
    from rlmy.agent.interpreters.monty import MontyInterpreter
    interp = MontyInterpreter(snapshot_path=workspace / "repl.snapshot")
    rlm = dspy.RLM("question -> answer", interpreter=interp)

Key Components:
    MontyInterpreter — satisfies DSPy's CodeInterpreter protocol (start/execute/
        shutdown/tools). Runs each snippet against a persistent Monty heap.

Design notes:
    - Monty binds host functions (tools) PER execute() call via `external_functions`,
      so there is nothing to "register" once — we pass tools fresh every time. DSPy's
      RLM resets an interpreter's `_tools_registered` flag before each run only if the
      attribute exists (`hasattr`); we don't define it, so nothing extra happens. The
      live heap simply persists across calls within a process.
    - Snapshotting is fully internal: `snapshot_path` is the only extra knob. We load
      it lazily on the first execute() (recovering a prior process's session) and
      save after each successful execute() (keeping disk current for the next restart).
      No caller ever needs to know whether snapshotting is on — swapping in a non-
      snapshotting interpreter changes construction only, never call sites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from pydantic_monty import (
    MontyError,
    MontyRepl,
    MontyRuntimeError,
    MontySyntaxError,
    ResourceLimits,
)

logger = logging.getLogger(__name__)

# Sentinel used to detect whether code actually called SUBMIT(). Monty's SUBMIT
# shim records its args here; we read it after the run.
_SUBMIT = "SUBMIT"


class MontyInterpreter:
    """CodeInterpreter backed by Monty, with optional disk-backed session snapshots.

    State (variables, functions, imported-module state) persists across execute()
    calls via Monty's incremental no-replay REPL. When `snapshot_path` is set, that
    state also persists across process restarts.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        snapshot_path: Path | str | None = None,
        resource_limits: ResourceLimits | None = None,
    ) -> None:
        """
        Args:
            tools: Host callables exposed to sandbox code by name. RLM injects its
                own (llm_query, user tools) into this dict before each run, so it
                must stay mutable and accessible via the `tools` property.
            output_fields: Output field definitions (name/type dicts) used to shape
                a positional SUBMIT(...) call into the expected output mapping. RLM
                sets this before each run.
            snapshot_path: If given, the REPL session is loaded from this file on the
                first execute() and saved to it after each successful execute().
            resource_limits: Optional Monty resource limits (memory, duration, etc.).
        """
        self._tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self._snapshot_path: Path | None = Path(snapshot_path) if snapshot_path else None
        self._resource_limits: ResourceLimits | None = resource_limits

        self._repl: MontyRepl = self._new_repl()
        # Whether we've attempted the one-time load from snapshot_path yet.
        self._restored: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _new_repl(self) -> MontyRepl:
        return MontyRepl(limits=self._resource_limits)

    def start(self) -> None:
        """No-op: Monty needs no warmup. Present to satisfy the protocol."""

    def shutdown(self) -> None:
        """Drop the live session. A fresh REPL is created on next use.

        We do NOT delete the snapshot file — it's the persistent record a future
        process restores from. Shutdown only releases the in-memory session.
        """
        self._repl = self._new_repl()
        self._restored = False

    def __enter__(self) -> MontyInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    # ── Tools (CodeInterpreter protocol) ─────────────────────────────────

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        """Mutable tool registry. RLM calls .update() on this before each run."""
        return self._tools

    @property
    def restores_state(self) -> bool:
        """Whether a prior session's REPL state comes back on restart.

        True only when we have a snapshot file to load from. Read by the UI's
        session-recovery message so it can tell the truth per interpreter
        (interpreters that don't define this attribute are assumed to be False).
        """
        return self._snapshot_path is not None

    # ── Snapshot persistence (fully internal) ────────────────────────────

    def _restore_once(self) -> None:
        """Load a prior session from snapshot_path the first time we run.

        Corrupt or unreadable snapshots are ignored — we start fresh rather than
        crash, since a lost session is recoverable but a crash loses the turn.
        """
        if self._restored or self._snapshot_path is None:
            self._restored = True
            return
        self._restored = True
        if not self._snapshot_path.exists():
            return
        try:
            self._repl = MontyRepl.load(self._snapshot_path.read_bytes())
        except (ValueError, OSError) as e:
            # Corrupt/incompatible snapshot, or read failure → keep the fresh REPL.
            # Log it: the user is silently starting a new session instead of
            # resuming, and a breadcrumb is the only way to notice that happened.
            logger.warning(
                "Could not restore REPL snapshot %s (%s); starting a fresh session.",
                self._snapshot_path, type(e).__name__,
            )
            self._repl = self._new_repl()

    def _save(self) -> None:
        """Persist the live session to snapshot_path (best-effort)."""
        if self._snapshot_path is None:
            return
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self._snapshot_path.write_bytes(self._repl.dump())
        except OSError:
            # Never let a snapshot-write failure break the turn.
            pass

    # ── Execution (CodeInterpreter protocol) ─────────────────────────────

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Run one snippet against the persistent heap and return its result.

        Returns:
            FinalOutput  — if the snippet called SUBMIT(...)
            str          — captured print() output (trailing newline trimmed)
            None         — if nothing was printed and no value produced

        Raises:
            SyntaxError          — invalid Python (from MontySyntaxError)
            CodeInterpreterError — runtime errors (from MontyRuntimeError)
        """
        self._restore_once()

        prints: list[str] = []

        def on_print(_stream: Literal["stdout"], text: str) -> None:
            # _stream is always "stdout" today; we only care about the text.
            prints.append(text)

        # SUBMIT records its call into this box and returns None so the VM keeps
        # going; we interpret the box after the run completes.
        submitted: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def submit(*args: Any, **kwargs: Any) -> None:
            submitted.append((args, kwargs))

        external = dict(self._tools)
        external[_SUBMIT] = submit

        try:
            output = self._repl.feed_run(
                code,
                inputs=variables or None,
                external_functions=external,
                print_callback=on_print,
            )
        except MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except MontyRuntimeError as e:
            # A host tool may raise a control-flow signal — e.g. ask_user_guidance
            # raises SystemExit when the user types /quit. Monty catches that and
            # re-wraps it as MontyRuntimeError; if we surfaced it as a normal error
            # the LLM would see "[Error] SystemExit: 0" and loop forever, so the
            # user could never leave. Let those signals escape instead.
            self._reraise_control_flow(e)
            # If SUBMIT ran before the error, honor it — the agent's intent to
            # finish should win over a trailing exception.
            if submitted:
                self._save()
                return self._to_final(submitted[0])
            # Surface any output printed BEFORE the exception alongside the error.
            # Monty's print_callback has already received those lines by the time
            # feed_run raises, so we don't lose the agent's own diagnostics — the
            # LLM sees what ran up to the failing line, then why it failed.
            raise CodeInterpreterError(
                self._error_with_partial_output(e.display("type-msg"), prints)
            ) from e
        except MontyError as e:
            # Defensive catch-all: any other Monty failure (e.g. a typing error if
            # type-checking is ever enabled) still becomes a CodeInterpreterError
            # rather than escaping as a raw library exception the RLM loop can't
            # classify. Today only MontySyntaxError/MontyRuntimeError occur here.
            if submitted:
                self._save()
                return self._to_final(submitted[0])
            raise CodeInterpreterError(
                self._error_with_partial_output(str(e), prints)
            ) from e

        self._save()

        if submitted:
            return self._to_final(submitted[0])
        return self._build_output(output, prints)

    @staticmethod
    def _reraise_control_flow(error: MontyRuntimeError) -> None:
        """Re-raise a swallowed SystemExit/KeyboardInterrupt so it can escape.

        Monty runs host tools (our callables) inside `feed_run` and catches any
        exception they raise — including BaseException control-flow signals — and
        re-wraps them as MontyRuntimeError, losing the original type. We reconstruct
        the underlying exception via `MontyRuntimeError.exception()`; if it's a quit
        (SystemExit) or interrupt (KeyboardInterrupt) signal, we re-raise it so it
        propagates past the RLM loop (which only catches CodeInterpreterError /
        SyntaxError) and actually exits, instead of being fed back to the LLM as a
        plain error it would just retry. Anything else is left for the caller to
        wrap as a normal CodeInterpreterError.
        """
        try:
            original = error.exception()
        except Exception:
            # Older/newer Monty without a reconstructable exception — fall back to
            # leaving it as a normal error rather than masking it.
            return
        if isinstance(original, (SystemExit, KeyboardInterrupt)):
            raise original

    # ── SUBMIT / output shaping ──────────────────────────────────────────

    def _to_final(self, call: tuple[tuple[Any, ...], dict[str, Any]]) -> FinalOutput:
        """Turn a recorded SUBMIT(...) call into a FinalOutput dict.

        RLM expects FinalOutput.output to be a dict keyed by the signature's output
        field names. We support both call styles the prompt teaches:
            SUBMIT(answer=...)        → kwargs become the dict directly
            SUBMIT(value1, value2)    → positionals zip onto output_fields by order
        """
        args, kwargs = call
        if kwargs:
            return FinalOutput(dict(kwargs))

        names = [f["name"] for f in (self.output_fields or [])]
        # A single dict positional that already matches the schema is passed through
        # (e.g. SUBMIT({"answer": x})) rather than nested under a field name.
        if len(args) == 1 and isinstance(args[0], dict) and names and set(args[0]).issubset(names):
            return FinalOutput(dict(args[0]))
        if names and args:
            # zip() would silently truncate a count mismatch; surface it instead so
            # the agent learns it passed the wrong number of values to SUBMIT.
            if len(args) != len(names):
                raise CodeInterpreterError(
                    f"SUBMIT() got {len(args)} positional value(s) but the output has "
                    f"{len(names)} field(s) {names}. Pass one value per field, or use "
                    f"keywords: SUBMIT({'=..., '.join(names)}=...)."
                )
            return FinalOutput(dict(zip(names, args)))
        if len(args) == 1:
            return FinalOutput(args[0])
        if not args:
            return FinalOutput(None)
        # Multiple positionals with no declared output fields: we can't map them to
        # names, so keep them all as a tuple rather than silently dropping all but
        # the first. (Unreachable via RLM, which always sets output_fields.)
        return FinalOutput(tuple(args))

    @staticmethod
    def _trim_trailing_newline(prints: list[str]) -> str:
        """Join captured print fragments and drop a single trailing newline."""
        captured = "".join(prints)
        if captured.endswith("\n"):
            captured = captured[:-1]
        return captured

    @classmethod
    def _error_with_partial_output(cls, error_text: str, prints: list[str]) -> str:
        """Combine output printed before a crash with the error message.

        Without this, a snippet like `print("step 1"); boom()` would surface only
        the exception, hiding the agent's own progress markers. We keep both so the
        LLM sees how far it got and why it stopped.
        """
        captured = cls._trim_trailing_newline(prints)
        if not captured:
            return error_text
        return f"{captured}\n{error_text}"

    @classmethod
    def _build_output(cls, value: Any, prints: list[str]) -> Any:
        """Prefer captured print output; fall back to the snippet's value."""
        captured = cls._trim_trailing_newline(prints)
        if captured:
            return captured
        if value is not None:
            return str(value)
        return None
