"""
Purpose: Reusable RLM components — InterruptableRLM, cooperative SIGINT, trajectory persistence.
Usage:
    # As a library (from mainmcp.py or elsewhere):
    from rlmy.agent.rlm import InterruptableRLM, InterruptFlag, save_trajectory, load_trajectory

    # As a standalone CLI:
    python prototype/cli_proto.py
Key Components:
    InterruptableRLM — interrupt-safe RLM subclass with prior trajectory injection
    InterruptFlag — cooperative SIGINT handler (flag instead of exception)
    save_trajectory / load_trajectory — opt-in disk persistence
Conventions: No module-level side effects (LM config, logging setup, etc.) — all in main().
"""

import inspect
import logging
import shutil
import signal
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ClassVar, List

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

import dspy
from pydantic import BaseModel, ConfigDict
from rich.console import Console, Group
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Themed console: inline `code` in Markdown renders as bold coral instead of
# the default "bold cyan on black" which is hard to read in dark terminals.
_MD_THEME = Theme({"markdown.code": "bold #e06c75"})
rprint = Console(theme=_MD_THEME).print

# Max panel width: 120 chars or terminal width, whichever is smaller.
# Prevents panels from becoming unreadably wide on large monitors.
_MAX_PANEL_WIDTH = 120


def _panel_width() -> int:
    """Panel width capped at _MAX_PANEL_WIDTH, respecting terminal size."""
    return min(_MAX_PANEL_WIDTH, shutil.get_terminal_size().columns)

from dspy.predict.rlm import _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLHistory, REPLEntry

logger = logging.getLogger(__name__)


# =============================================================================
# Trajectory Persistence — delegated to prototype.trajectory
# =============================================================================
# Re-exported here for backward compatibility (existing import sites).
from rlmy.agent.trajectory import (
    save_trajectory,
    load_trajectory,
    clear_trajectory,
    compact_trajectory,
    format_compact_stats,
    format_compact_stats_for_llm,
    estimate_tokens,
    DEFAULT_TRAJECTORY_FILE,
    DEFAULT_TOTAL_BUDGET,
)


# =============================================================================
# Cooperative Interrupt Flag
# =============================================================================

class InterruptFlag:
    """
    Purpose: Cooperative SIGINT handling — converts Ctrl+C from async exception to
             a flag checked at safe iteration boundaries.

    Why: KeyboardInterrupt can fire on ANY Python bytecode instruction. Lines outside
         try/except blocks (list(), dict(), assignment) are all vulnerable. A signal
         handler converts SIGINT into a flag that's checked cooperatively, so the
         current operation always finishes cleanly before we react.

    Usage Patterns:
        flag = InterruptFlag()
        flag.install()       # replaces SIGINT handler
        ...
        if flag.is_set():    # check at safe boundary
            flag.clear()
            # handle interrupt
        flag.restore()       # restore original handler
    """

    def __init__(self):
        self._interrupted = False
        self._original_handler = None

    def install(self):
        """Replace SIGINT handler with flag-setting handler.
        Only works from main thread; silently skips if called from worker thread.
        """
        import threading
        self._interrupted = False
        if threading.current_thread() is not threading.main_thread():
            logger.debug("InterruptFlag.install() skipped — not main thread")
            return
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)

    def restore(self):
        """Restore original SIGINT handler."""
        if self._original_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._original_handler)
            except ValueError:
                pass  # not main thread
            self._original_handler = None

    def _handler(self, signum, frame):
        """Signal handler: set flag, print notice."""
        self._interrupted = True
        # Print immediately so user sees feedback
        rprint("\n[bold yellow]⚡ Interrupt received — finishing current step...[/bold yellow]")

    def is_set(self) -> bool:
        return self._interrupted

    def clear(self):
        self._interrupted = False


# =============================================================================
# User Input
# =============================================================================
#
# CRITICAL INTERACTION MODEL — read this before modifying prompt_user():
#
# prompt_user() is THE single funnel for ALL user<>system interaction. It is called:
#
# 1. MOST COMMONLY: from contextual_ask_user_guidance() — the LLM's ask_user_guidance
#    tool. This happens many times PER RLM TURN (inside forward/aforward iteration loop).
#    The LLM generates a question, the user responds, the LLM sees the response in
#    repl_history and continues. This is the PRIMARY interaction path.
#
# 2. RARELY: from the between-turns prompt in mainmcp.py's conversation loop
#    ("What would you like to do next?"). This only fires when the RLM returns
#    (via SUBMIT or max iterations), which is infrequent.
#
# 3. FROM broadcast_user_update: print_only=True mode. No user input collected.
#
# Therefore:
# - Slash commands (/quit, etc.) are checked HERE, not in mainmcp.py's loop
# - RichMarkdown wrapping is applied HERE for LLM-generated questions
# - SIGINT handling is managed HERE because the cooperative flag is active during RLM
#

# Slash commands: decoupled into prototype.commands. prompt_user() calls
# REGISTRY.match() and either handles self-contained commands (exit, help)
# or raises SlashCommandSignal for context-dependent ones (compact, reset).
from rlmy.agent.commands import REGISTRY, SlashCommandSignal


# =============================================================================
# prompt_toolkit Session
# =============================================================================
# Replaces the while+input() loop in prompt_user() with multiline editing.
# Enter = newline (safe for pasting). Alt+Enter (or Esc,Enter) = submit.
# ↑/↓ arrow history persists across prompt_user() calls within a session.
# Tab autocomplete for all registered slash commands (including aliases).

_completer = WordCompleter(
    list(REGISTRY._lookup.keys()),
    ignore_case=True,
    sentence=True,  # complete the whole input, not just the last word
)

_session = PromptSession(
    history=InMemoryHistory(),
    multiline=True,
    completer=_completer,
    complete_while_typing=False,  # only complete on Tab press
)


def _render_panel_body(question: str, header: str = "") -> Group | RichMarkdown:
    """
    Build the panel body: optional Rich-markup header + Markdown question.

    Why separate: build_status_header() returns Rich console markup ([dim]...[/dim]).
    RichMarkdown treats those tags as literal text. By rendering them as separate
    renderables via Group(), each uses its own rendering engine.

    Args:
        question: LLM-generated text, rendered as Markdown.
        header: Rich console markup (e.g., from build_status_header). Optional.

    Returns:
        Group(header_text, markdown) if header is non-empty, else just RichMarkdown.
    """
    md = RichMarkdown(question)
    if header:
        return Group(Text.from_markup(header), Text(""), md)
    return md


def prompt_user(question: str, print_only=False, header: str = ""):
    """
    Purpose: THE single funnel for all user<>system interaction in the RLM pipeline.

    This function is called most frequently from contextual_ask_user_guidance() —
    the LLM's ask_user_guidance tool — many times per RLM turn. The between-turns
    prompt in mainmcp.py is the RARE caller. All user input flows through here.

    Features:
    - Multiline input via prompt_toolkit (Enter = newline, Alt+Enter = submit)
    - Arrow-key history recall across prompts (InMemoryHistory)
    - Tab autocomplete for slash commands (WordCompleter from REGISTRY)
    - Rich Markdown rendering for LLM-generated question text
    - Rich markup header (optional) rendered separately from Markdown body
    - Slash commands (via CommandRegistry) intercepted before returning to caller
    - SIGINT temporarily restored to default during input collection
    - Panel width capped at 120 chars

    Args:
        question: The prompt text to display. Usually LLM-generated (from ask_user_guidance).
        print_only: If True, display as non-interactive update panel (broadcast_user_update).
        header: Optional Rich console markup prepended above the Markdown question.
                Used by contextual tools to show status (step/tokens/cost) without
                corrupting the Markdown rendering. Must NOT be embedded in `question`.

    Returns:
        User's multiline response as a single string, or triggers sys.exit for /quit.

    SIGINT handling: This function may be called from inside an RLM tool (ask_user_guidance)
    while our cooperative InterruptFlag is active. We temporarily restore the default SIGINT
    handler so Ctrl+C works normally during user input, then re-install the cooperative handler.
    """
    if print_only:
        # Communication mode: display as update panel
        body = _render_panel_body(question, header)
        rprint(
            Panel(
                body,
                title="[bold cyan]📢 System Update[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
                width=_panel_width(),
            )
        )
        return "Posted and finished. reminder: use ask_user_guidance if you need two-way communication."

    # Loop handles self-contained commands (e.g. /help) without recursion.
    # Previously used recursive call: prompt_user(question, ...) which could
    # hit RecursionError after ~1000 consecutive /help commands.
    while True:
        # User input mode: display as input prompt
        body = _render_panel_body(question, header)
        rprint(
            Panel(
                body,
                title="[bold yellow]❓ User Input Needed[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
                width=_panel_width(),
            )
        )
        rprint(
            "[dim](Enter = newline. Alt+Enter to submit. Ctrl+C to cancel. Tab for commands.)[/dim]"
        )

        # Temporarily restore default SIGINT so Ctrl+C works during input.
        # This is needed because InterruptableRLM's cooperative SIGINT handler
        # may be active (it sets a flag instead of raising KeyboardInterrupt).
        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            user_text = _session.prompt(
                "│ ",  # Left margin indicator
                prompt_continuation="│ ",
            )
        except KeyboardInterrupt:
            rprint("\n[bold yellow]Input cancelled.[/bold yellow]")
            user_text = ""
        except EOFError:
            rprint("\n[bold yellow]Input cancelled (EOF).[/bold yellow]")
            user_text = ""
        finally:
            # Re-install the cooperative handler (or whatever was there before)
            signal.signal(signal.SIGINT, prev_handler)

        # Slash command interception: if the ENTIRE input matches a registered command.
        # Self-contained commands (exit, help) are handled here.
        # Context-dependent commands (compact, reset) raise SlashCommandSignal.
        cmd = REGISTRY.match(user_text.strip())
        if cmd is not None:
            if cmd.action == "exit":
                rprint("\n[bold]Goodbye![/bold]")
                raise SystemExit(0)
            if cmd.action == "help":
                rprint(REGISTRY.help_panel())
                # Re-prompt: help is informational, user still needs to answer.
                # Loop back instead of recursing to avoid RecursionError.
                continue
            # Context-dependent command → signal caller
            raise SlashCommandSignal(cmd)

        rprint("[bold green]✓ Input received. Continuing...[/bold green]")
        return user_text


# =============================================================================
# Contextual Tools Framework
# =============================================================================
# Adapted from the contextual tools pattern (telegram project's tools.py)
# Pattern: contextual_* functions receive hidden context via **kwargs sentinel.
# Factory strips prefix + injects context, producing clean LLM-visible signatures.


class RLMContext(BaseModel):
    """
    Purpose: Holds RLM execution state that tools need but the LLM shouldn't control.

    Attributes:
        rlm: The InterruptableRLM instance
        trajectory: Current trajectory snapshot (list of dicts), updated by callback
        iteration: Current iteration number within forward()
        max_iterations: Max iterations configured on the RLM
        token_usage: Running token count (from lm.history)
        metadata: Extensible dict for tool-specific state

    Conventions:
        Mutable — the trajectory callback updates fields in-place between iterations.
        Tools wrapped at init time see updated context automatically since they hold
        a reference to the same RLMContext instance.
    """
    rlm: Any = None
    trajectory: list[dict] = []
    iteration: int = 0
    max_iterations: int = 100
    metadata: dict = {}

    # Sentinel key for **kwargs injection — tools retrieve context via this key
    _sentinel: ClassVar[str] = "_RLMContext"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def make_contextual_tool(
    contextual_func: Callable,
    context: RLMContext,
) -> Callable:
    """
    Purpose: Wrap a contextual_* function to hide context injection from the LLM.

    Takes a function that expects RLMContext in **kwargs and returns a new function
    with a clean signature. Strips "contextual_" prefix from the name.

    Why this works with dspy.Tool: DSPy's Tool() wrapper calls inspect.signature()
    on the function. We set wrapper.__signature__ to the original (which has **kwargs
    but the LLM only sees the explicit params). The sentinel key is invisible.
    """
    original_sig = inspect.signature(contextual_func)

    @wraps(contextual_func)
    def wrapper(*args, **kwargs):
        bound_args = original_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        injected = {RLMContext._sentinel: context}
        return contextual_func(**bound_args.arguments, **injected)

    # Build clean signature for LLM introspection — strip **kwargs so DSPy Tool
    # doesn't expose it to the LLM. Only explicit params are visible.
    clean_params = [
        p for p in original_sig.parameters.values()
        if p.kind != inspect.Parameter.VAR_KEYWORD
    ]
    wrapper.__signature__ = original_sig.replace(parameters=clean_params)

    # Strip prefix: contextual_ask_user_guidance -> ask_user_guidance
    if wrapper.__name__.startswith("contextual_"):
        wrapper.__name__ = wrapper.__name__[len("contextual_"):]
    else:
        raise ValueError(
            f"Function '{wrapper.__name__}' must start with 'contextual_' prefix"
        )

    return wrapper


def make_contextual_tools(
    contextual_funcs: List[Callable],
    context: RLMContext,
) -> List[Callable]:
    """Wrap multiple contextual functions for LLM use."""
    return [make_contextual_tool(fn, context) for fn in contextual_funcs]


# =============================================================================
# Contextual Tool Implementations
# =============================================================================
# Convention: Functions prefixed with "contextual_" accept **kwargs.
# Retrieve context via: ctx = kwargs.get(RLMContext._sentinel)


def build_status_header(ctx: RLMContext) -> str:
    """
    Purpose: Build a Rich-formatted status line from RLMContext for display in prompts.

    Why extracted: Used by both contextual tools AND the between-turns prompt in mainmcp.py.

    Token display strategy:
    - Primary: trajectory token estimate (always fresh, reflects compaction immediately)
    - Secondary: LLM actual token count from ctx.rlm.history[-1] (canonical per-call usage)
    - The trajectory estimate is what the LLM will see on its next call, so it's the
      most honest representation of "how big is the prompt going to be".
    - After /compact, rlm.history[-1] is stale (from the pre-compaction turn). The
      trajectory estimate reflects the compacted state immediately.

    Auto-warning: When trajectory token estimate exceeds DEFAULT_TOTAL_BUDGET,
    appends "⚠️ /compact" to nudge the user.

    Returns:
        Rich-formatted dim string like "[dim]Step 5/100 | 12 entries | ~4,200 traj tokens | $0.0312[/dim]"
        or empty string if ctx is None.
    """
    if ctx is None:
        return ""

    n_steps = len(ctx.trajectory)
    parts = [f"Step {ctx.iteration}/{ctx.max_iterations}", f"{n_steps} entries"]

    # Trajectory token estimate — always fresh, reflects compaction immediately.
    # This is what the LLM will actually see on its next call.
    traj_tokens = sum(estimate_tokens(
        (e.get("reasoning", "") + e.get("code", "") + e.get("output", ""))
    ) for e in ctx.trajectory)
    if traj_tokens:
        parts.append(f"~{traj_tokens:,} traj tokens")

    # Cost from the RLM's own history — canonical path
    try:
        last_entry = ctx.rlm.history[-1]
        cost = last_entry.get("cost", 0) or 0
        if cost > 0:
            parts.append(f"${cost:.4f}")
    except (IndexError, KeyError, TypeError, AttributeError):
        pass

    # Auto-warning when trajectory is over budget
    if traj_tokens > DEFAULT_TOTAL_BUDGET:
        parts.append("⚠️ /compact")

    return f"[dim]{' | '.join(parts)}[/dim]"


def contextual_ask_user_guidance(question: str, **kwargs) -> str:
    """
    Purpose: Request guidance from user with visual emphasis and execution context.

    The LLM sees: ask_user_guidance(question: str) -> str
    Hidden context provides iteration count, trajectory length, and token usage
    via build_status_header() — passed as `header` to prompt_user() so Rich markup
    renders separately from the LLM's Markdown question text.

    Return value is wrapped with a directive tag so the LLM prioritizes it over
    the original `question` variable (which may be stale in long sessions).
    See journal entry "Stale-Input Bug Root Cause Analysis" for details.

    SlashCommandSignal handling: If the user types a context-dependent command
    (e.g., /compact), prompt_user() raises SlashCommandSignal. We catch it here
    because we have access to the RLMContext needed for dispatch.
    """
    ctx: RLMContext = kwargs.get(RLMContext._sentinel)
    header = build_status_header(ctx)

    try:
        user_response = prompt_user(question, header=header)
    except SlashCommandSignal as sig:
        return _handle_slash_command(sig, ctx, question, header)

    # Wrap the response so it stands out in repl_history as a fresh user directive.
    # Without this, at 70+ trajectory entries the LLM sometimes latches onto the
    # stale `question` variable instead of the latest tool output.
    return f"[USER DIRECTIVE — this supersedes the original `question` variable]\n{user_response}"


def _handle_slash_command(sig: SlashCommandSignal, ctx: RLMContext,
                          question: str, header: str) -> str:
    """
    Purpose: Dispatch context-dependent slash commands inside ask_user_guidance.

    CRITICAL DESIGN INSIGHT (2026-04-21):
    history is a LOCAL variable in forward()/aforward(). It's what the LLM sees
    (via repl_history=history in the prompt). Modifying ctx.trajectory or the disk
    file does NOT change what the LLM sees within the current turn. history is built
    once at _init_history() and only grows.

    Therefore: /compact and /reset MUST end the current turn by setting the RLM's
    interrupt flag. The compacted/reset trajectory is fed to the NEXT turn via
    set_prior_trajectory(). The LLM restarts with a smaller history.

    Flow:
    1. Tool returns (this function's return value goes into repl_history)
    2. _process_execution_result appends it to history
    3. Top of next iteration → interrupt flag checked → _make_interrupted_prediction
    4. Turn ends with interrupted=True
    5. Post-turn code: compacts result.trajectory, feeds to next turn
    """
    traj_path = ctx.metadata.get("trajectory_file") if ctx else None

    if sig.action == "compact":
        if not ctx or not ctx.trajectory:
            rprint(Panel(
                "Trajectory is empty. Nothing to compact.",
                title="[bold]📦 Compact[/bold]",
                border_style="blue",
                padding=(1, 2),
                width=_panel_width(),
            ))
            return "[SYSTEM] Trajectory is empty. Nothing to compact. Continue with your current task."

        # Dry-run to show stats (actual compaction happens post-turn on result.trajectory)
        _, stats = compact_trajectory(list(ctx.trajectory), dry_run=False)

        # Flag for post-turn compaction of result.trajectory
        ctx.metadata["compaction_requested"] = True

        # CRITICAL: Set the interrupt flag to END this turn. The compacted trajectory
        # can only take effect when fed to a NEW turn via set_prior_trajectory().
        # history (the LLM's actual prompt) is a local variable in forward() —
        # we cannot shrink it from here.
        if ctx.rlm and hasattr(ctx.rlm, '_interrupt_flag'):
            ctx.rlm._interrupt_flag._interrupted = True

        rprint(Panel(
            format_compact_stats(stats) + "\n\n[bold]Ending current turn to apply compaction...[/bold]",
            title="[bold]📦 Trajectory Compacted[/bold]",
            border_style="blue",
            padding=(1, 2),
            width=_panel_width(),
        ))
        return format_compact_stats_for_llm(stats)

    if sig.action == "reset":
        if ctx:
            # Flag for post-turn handling
            ctx.metadata["reset_requested"] = True
            # Set interrupt flag to end the turn
            if ctx.rlm and hasattr(ctx.rlm, '_interrupt_flag'):
                ctx.rlm._interrupt_flag._interrupted = True
        rprint(Panel(
            "Trajectory will be cleared after this turn ends.",
            title="[bold]🗑️ Reset[/bold]",
            border_style="red",
            padding=(1, 2),
            width=_panel_width(),
        ))
        return "[SYSTEM] Trajectory reset requested. Turn ending."

    # Unknown context-dependent command — shouldn't happen if REGISTRY is correct
    return f"[SYSTEM] Unknown command: {sig.command.name}"


def contextual_broadcast_user_update(message: str, **kwargs) -> str:
    """
    Purpose: Communicate system updates to user (non-blocking, display only).

    The LLM sees: broadcast_user_update(message: str) -> str
    Hidden context provides execution stats via build_status_header() —
    passed as `header` to prompt_user() for proper Rich/Markdown separation.
    """
    ctx: RLMContext = kwargs.get(RLMContext._sentinel)
    header = build_status_header(ctx)

    if ctx:
        ctx.metadata.setdefault("broadcast_count", 0)
        ctx.metadata["broadcast_count"] += 1

    return prompt_user(message, print_only=True, header=header)


# Default contextual tools list — import and wrap with make_contextual_tools()
DEFAULT_CONTEXTUAL_TOOLS = [
    contextual_ask_user_guidance,
    contextual_broadcast_user_update,
]


# =============================================================================
# InterruptableRLM
# =============================================================================

class InterruptableRLM(dspy.RLM):
    """
    Purpose: RLM subclass with cooperative SIGINT handling, prior trajectory injection,
             and disk-persisted trajectory for crash recovery.

    Attributes:
        _prior_trajectory: list[dict] — trajectory dicts to pre-populate REPLHistory
        _interrupt_flag: InterruptFlag — cooperative SIGINT flag
        _trajectory_callback: callable — called with trajectory after every mutation

    Usage Patterns:
        rlm = InterruptableRLM(signature, ...)
        rlm.set_prior_trajectory(accumulated_trajectory)
        result = rlm(question="...")
        # result.interrupted == True if Ctrl+C was pressed
        # result.trajectory always available (partial or complete)

    Conventions:
        - SIGINT handler is installed only during forward/aforward execution
        - Original handler is restored after execution (so Ctrl+C at prompt works normally)
        - Interrupt flag is checked at the top of each iteration (safe boundary)
        - trajectory_callback is called after every history mutation for disk persistence
    """

    def __init__(self, *args, trajectory_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prior_trajectory: list[dict] = []
        self._interrupt_flag = InterruptFlag()
        self._trajectory_callback = trajectory_callback or (lambda t: None)

    def set_prior_trajectory(self, trajectory: list[dict]):
        """Set prior trajectory to inject as initial REPLHistory on next forward() call."""
        self._prior_trajectory = list(trajectory)

    def _make_interrupted_prediction(self, history, output_field_names):
        """Build a Prediction capturing partial trajectory on interrupt."""
        rprint("\n[bold red]⚡ Interrupted! Saving trajectory...[/bold red]")
        trajectory = [e.model_dump() for e in history]
        self._trajectory_callback(trajectory)
        return Prediction(
            interrupted=True,
            trajectory=trajectory,
            **{name: "[interrupted]" for name in output_field_names},
        )

    def _init_history(self):
        """Consume prior trajectory and build initial REPLHistory."""
        prior_trajectory = self._prior_trajectory
        self._prior_trajectory = []
        if prior_trajectory:
            return REPLHistory(entries=[REPLEntry(**e) for e in prior_trajectory])
        return REPLHistory()

    def _notify_trajectory(self, history):
        """Persist current trajectory state via callback."""
        self._trajectory_callback([e.model_dump() for e in history])

    def forward(self, **input_args) -> Prediction:
        """
        Purpose: Execute RLM with cooperative SIGINT handling and disk persistence.

        Uses InterruptFlag (signal handler) instead of try/except KeyboardInterrupt.
        SIGINT sets a flag checked at the TOP of each iteration — the current LLM call
        or code execution always finishes cleanly before we react.

        Trajectory is persisted via callback after every history mutation.
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)
        history = self._init_history()

        self._interrupt_flag.install()
        try:
            with self._interpreter_context(execution_tools) as repl:
                for iteration in range(self.max_iterations):
                    # Check interrupt flag at safe boundary
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        return self._make_interrupted_prediction(history, output_field_names)

                    variables_info = [v.format() for v in variables]

                    # Phase 1: LLM call — runs to completion, flag checked after
                    action = self.generate_action(
                        variables_info=variables_info,
                        repl_history=history,
                        iteration=f"{iteration + 1}/{self.max_iterations}",
                    )

                    if self.verbose:
                        logger.info(
                            f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                            f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
                        )

                    # Check flag after LLM call (might have been pressed during it)
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        # Record the reasoning the LLM produced before we stop
                        history = history.append(
                            reasoning=action.reasoning,
                            code=_strip_code_fences(action.code),
                            output="[interrupted before execution]",
                        )
                        return self._make_interrupted_prediction(history, output_field_names)

                    # Phase 2: Code execution — runs to completion, flag checked after
                    code = _strip_code_fences(action.code)
                    try:
                        result = repl.execute(code, variables=dict(input_args))
                    except (CodeInterpreterError, SyntaxError) as e:
                        result = f"[Error] {e}"

                    # Check flag after code execution
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        history = history.append(
                            reasoning=action.reasoning,
                            code=code,
                            output="[interrupted after execution]",
                        )
                        return self._make_interrupted_prediction(history, output_field_names)

                    # Phase 3: Process result and persist
                    processed = self._process_execution_result(
                        action, result, history, output_field_names
                    )
                    if isinstance(processed, Prediction):
                        return processed
                    history = processed
                    self._notify_trajectory(history)

                # Max iterations reached — use extract fallback
                return self._extract_fallback(variables, history, output_field_names)
        finally:
            self._interrupt_flag.restore()

    async def aforward(self, **input_args) -> Prediction:
        """
        Purpose: Async version of forward() with same cooperative interrupt handling.

        Required for MCP usage (mainmcp.py) where the event loop is async.
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)
        history = self._init_history()

        self._interrupt_flag.install()
        try:
            with self._interpreter_context(execution_tools) as repl:
                for iteration in range(self.max_iterations):
                    # Check interrupt flag at safe boundary
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        return self._make_interrupted_prediction(history, output_field_names)

                    variables_info = [v.format() for v in variables]

                    # Phase 1: Async LLM call
                    action = await self.generate_action.acall(
                        variables_info=variables_info,
                        repl_history=history,
                        iteration=f"{iteration + 1}/{self.max_iterations}",
                    )

                    if self.verbose:
                        logger.info(
                            f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                            f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
                        )

                    # Check flag after LLM call
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        history = history.append(
                            reasoning=action.reasoning,
                            code=_strip_code_fences(action.code),
                            output="[interrupted before execution]",
                        )
                        return self._make_interrupted_prediction(history, output_field_names)

                    # Phase 2: Code execution (sync even in async path)
                    code = _strip_code_fences(action.code)
                    try:
                        result = repl.execute(code, variables=dict(input_args))
                    except (CodeInterpreterError, SyntaxError) as e:
                        result = f"[Error] {e}"

                    # Check flag after code execution
                    if self._interrupt_flag.is_set():
                        self._interrupt_flag.clear()
                        history = history.append(
                            reasoning=action.reasoning,
                            code=code,
                            output="[interrupted after execution]",
                        )
                        return self._make_interrupted_prediction(history, output_field_names)

                    # Phase 3: Process result and persist
                    processed = self._process_execution_result(
                        action, code, result, history, output_field_names
                    )
                    if isinstance(processed, Prediction):
                        return processed
                    history = processed
                    self._notify_trajectory(history)

                # Max iterations reached — use async extract fallback
                return await self._aextract_fallback(variables, history, output_field_names)
        finally:
            self._interrupt_flag.restore()


# =============================================================================
# Main CLI Loop
# =============================================================================


def _handle_slash_command_between_turns(
    sig: SlashCommandSignal,
    rlm_context: "RLMContext",
    conversation_trajectory: list[dict],
    rlm: "InterruptableRLM",
    persist_to_disk: bool,
):
    """
    Purpose: Dispatch context-dependent slash commands in the standalone between-turns loop.

    Why separate from _handle_slash_command: That one runs inside ask_user_guidance (during RLM)
    and returns a string to the LLM. This one runs outside the RLM loop and has direct access
    to the conversation_trajectory list and rlm instance for updating set_prior_trajectory.
    """
    if sig.action == "compact":
        if not conversation_trajectory:
            rprint(Panel(
                "Trajectory is empty. Nothing to compact.",
                title="[bold]📦 Compact[/bold]",
                border_style="blue",
                padding=(1, 2),
                width=_panel_width(),
            ))
            return

        result = compact_trajectory(conversation_trajectory)
        compacted_traj, stats = result
        conversation_trajectory[:] = compacted_traj
        rlm_context.trajectory = conversation_trajectory
        rlm_context.iteration = len(conversation_trajectory)
        if persist_to_disk:
            save_trajectory(conversation_trajectory)

        rprint(Panel(
            format_compact_stats(stats),
            title="[bold]📦 Trajectory Compacted[/bold]",
            border_style="blue",
            padding=(1, 2),
            width=_panel_width(),
        ))
        return

    if sig.action == "reset":
        conversation_trajectory.clear()
        rlm_context.trajectory = conversation_trajectory
        rlm_context.iteration = 0
        if persist_to_disk:
            save_trajectory(conversation_trajectory)
        rprint(Panel(
            "Trajectory cleared. Starting fresh.",
            title="[bold]🗑️ Reset[/bold]",
            border_style="red",
            padding=(1, 2),
            width=_panel_width(),
        ))
        return

    rprint(f"[yellow]Unknown command: {sig.command.name}[/yellow]")


def main(persist_to_disk: bool = False):
    """
    Purpose: Main CLI loop with cooperative interrupt handling and conversation continuity.
    Args:
        persist_to_disk: If True, trajectory is saved to .trajectory_state.jsonl after every
                         mutation and recovered on startup. Opt-in because REPL state (variables,
                         functions) is NOT recoverable across process restarts — the LLM would
                         see prior code steps but the variables wouldn't exist.
    Conventions: Ctrl+C during RLM sets flag (checked at iteration boundary).
                 Ctrl+C at the prompt exits the program (original handler restored).
                 REPL state (variables, functions, imports) persists across turns within a session.
    """
    # --- Setup (only runs when used as standalone CLI) ---
    # Required for RLM verbose output (reasoning + code at each iteration).
    # InterruptableRLM's forward/aforward use logger.info() when verbose=True,
    # but Python's logging module silently drops messages if no handler is configured.
    # Each entrypoint (mainmcp.py, cli_proto.py main()) must call basicConfig itself.
    # RichHandler gives syntax-highlighted code blocks and styled log output.
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=Console(theme=_MD_THEME), show_path=False, show_time=False, rich_tracebacks=True)],
    )

    OPUS = "bedrock/us.anthropic.claude-opus-4-6-v1"
    lm = dspy.LM(model=OPUS, cache=True)
    dspy.settings.configure(lm=lm)

    class LongContextWithDictQA(dspy.Signature):
        """Answer the question using the provided context and data dictionary.

        Communication Tools:
        - Use `broadcast_user_update` (1-way) between significant chunks of work (not within one block)
        - Use `ask_user_guidance` (2-way) when you need human decision or are unsure

        Filesystem Tools:
        - `list_files(path="")` - Returns list of file paths (Python list)
        - `read_file(path="example.md")` - Read from sandbox (relative) or system (absolute path)
        - `write_file(path="output/result.md", contents="...")` - Write to sandbox only

        MCP Tools (with typed parameters):
        - `mcp_slack_search(query='search term')` - Search Slack messages
        - `mcp_slack_batch_get_thread_replies(threads=['id1', 'id2'])` - Get thread replies
        - `mcp_builder_ReadInternalWebsites(inputs=['url1', 'url2'])` - Read internal websites
        - `mcp_outlook_email_folders()` - List email folders (no params)
        - `mcp_outlook_email_read(id='email-id')` - Read email by ID

        Sandbox: ./sandbox/<workspace>/ (unrestricted access)
        System: Absolute paths (e.g., "/Users/...") require user permission on first access

        **Important**: Take an out if necessary. You are allowed to not know the answer.
        Feel free to say "I don't know" whenever applicable.
        Ask for user guidance whenever you're unsure.

        Also, any `print()`s you do are for your own internal reasoning. I can't see it.
        Use broadcast_user_update to report findings (non-blocking).
        When you're done, always ask for the next task (there is always a next task).
        Use the ask_user_guidance tool (blocking) for that. otherwise you will be stuck in an infinite loop.
        """

        question: str = dspy.InputField(desc="Your starting point")
        answer: str = dspy.OutputField(desc="Concise final answer")

    persistent_repl = PythonInterpreter()
    conversation_trajectory: list[dict] = []

    # Opt-in: recover trajectory from disk
    if persist_to_disk:
        conversation_trajectory = load_trajectory()
        if conversation_trajectory:
            n = len(conversation_trajectory)
            rprint(
                Panel(
                    f"Recovered {n} trajectory steps from previous session.\n"
                    "⚠️  Note: REPL variables from prior session are NOT recovered.",
                    title="[bold cyan]🔄 Session Recovered[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                    width=_panel_width(),
                )
            )

    # Mutable context — tools hold a reference, see updates automatically.
    # The trajectory callback updates fields in-place between iterations.
    rlm_context = RLMContext(
        trajectory=list(conversation_trajectory),
        max_iterations=100,
    )

    # Wrap contextual tools — LLM sees clean signatures, context injected at call time
    tools = make_contextual_tools(DEFAULT_CONTEXTUAL_TOOLS, rlm_context)

    # Callback: simple pass-through. /compact and /reset set the interrupt flag
    # to end the turn, then post-turn code compacts result.trajectory.
    def on_trajectory_update(trajectory: list[dict]):
        nonlocal conversation_trajectory
        conversation_trajectory = trajectory
        rlm_context.trajectory = trajectory
        rlm_context.iteration = len(trajectory)
        if persist_to_disk:
            save_trajectory(trajectory)

    rlm = InterruptableRLM(
        LongContextWithDictQA,
        verbose=True,
        tools=tools,
        max_iterations=100,
        max_llm_calls=350,
        interpreter=persistent_repl,
        trajectory_callback=on_trajectory_update,
    )

    # Back-reference: context holds the RLM instance (for tools that need config access)
    rlm_context.rlm = rlm

    try:
        while True:
            try:
                # Route through prompt_user for slash command support.
                # SlashCommandSignal for context-dependent commands caught below.
                status = build_status_header(rlm_context)
                user_input = prompt_user("What would you like to do?", header=status)
            except (KeyboardInterrupt, EOFError):
                rprint("\n[bold]Goodbye![/bold]")
                break
            except SlashCommandSignal as sig:
                _handle_slash_command_between_turns(
                    sig, rlm_context, conversation_trajectory, rlm, persist_to_disk
                )
                continue

            if not user_input.strip():
                continue

            # Record user's question BEFORE the RLM call — it triggered the steps that follow,
            # so it must appear before them in the chronological trajectory.
            conversation_trajectory.append({
                "reasoning": f"<user-input>{user_input}</user-input>",
                "code": "",
                "output": "",
            })

            # Inject prior context (including the new user input) so the RLM sees full history
            rlm.set_prior_trajectory(conversation_trajectory)
            result = rlm(question=user_input)

            # Update trajectory from result — RLM steps now follow the user input entry
            conversation_trajectory = list(result.trajectory)

            # Post-turn compaction: /compact set interrupt flag to end the turn.
            # Now compact result.trajectory and auto-restart with smaller history.
            if rlm_context.metadata.pop("compaction_requested", False):
                compacted, _stats = compact_trajectory(conversation_trajectory)
                conversation_trajectory = compacted
                rlm_context.trajectory = conversation_trajectory
                if persist_to_disk:
                    save_trajectory(conversation_trajectory)
                rprint("[bold cyan]🔄 Restarting with compacted trajectory...[/bold cyan]")
                continue  # skip interrupted/answer panel, go to prompt

            # Post-turn reset: clear everything and restart fresh.
            if rlm_context.metadata.pop("reset_requested", False):
                conversation_trajectory = []
                rlm_context.trajectory = conversation_trajectory
                rlm_context.iteration = 0
                if persist_to_disk:
                    clear_trajectory()
                continue

            if persist_to_disk:
                save_trajectory(conversation_trajectory)

            if getattr(result, "interrupted", False):
                n_steps = len(result.trajectory)
                rprint(
                    Panel(
                        f"Progress saved ({n_steps} steps in trajectory).\n"
                        "Type your next message to continue with full context.",
                        title="[bold red]⚡ Session Interrupted[/bold red]",
                        border_style="red",
                        padding=(1, 2),
                        width=_panel_width(),
                    )
                )
            else:
                rprint(
                    Panel(
                        RichMarkdown(result.answer or "*(no answer produced)*"),
                        title="[bold green]✅ Final Answer[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                        width=_panel_width(),
                    )
                )
    finally:
        # Clean up the persistent interpreter on exit
        persistent_repl.shutdown()


if __name__ == "__main__":
    main()
