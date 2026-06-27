"""
Purpose: MCP-enabled RLM agent with interrupt handling and conversation continuity.
Usage: python -m rlmy.agent.main [--sandbox-root PATH] [--cache-path PATH]
Key Components: MCPConnector (persistent MCP connections), InterruptableRLM (from rlmy.agent.rlm),
                SandboxManager (workspace selection), conversation loop with cooperative SIGINT.
Conventions: Ctrl+C during RLM sets flag (checked at iteration boundary).
             Ctrl+C at the prompt exits the program.
             Workspace selected at startup via interactive picker (sandbox_manager.py).
             Trajectory always persisted to workspace/trajectory.jsonl (JSONL format).
"""

import argparse
import asyncio
import readline  # This often resolves the character limit for input()
from pathlib import Path

import dspy
import nest_asyncio
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel

from rlmy.agent.ui import _MD_THEME, _panel_width, rprint

readline
nest_asyncio.apply()  # Allow nested event loops for sync MCP calls

# Required for RLM verbose output (reasoning + code at each iteration).
# InterruptableRLM's forward/aforward use logger.info() when verbose=True,
# but Python's logging module silently drops messages if no handler is configured.
# Each entrypoint (main.py, cli_proto.py main()) must call basicConfig itself.
# RichHandler gives syntax-highlighted code blocks and styled log output.
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=Console(theme=_MD_THEME),
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
        )
    ],
)


# prompt_user is centralized in cli_proto — imported in run_agent() alongside other cli_proto components.
# Do NOT duplicate it here. See prototype/cli_proto.py:141.
# Standard tools (non-MCP)
#
# AGENT VISIBILITY PATTERN:
# Agents see ONLY function signatures (name + parameters), NOT docstrings.
# Therefore:
#   1. Function names must be self-descriptive (e.g., read_document vs read_binary)
#   2. Error messages must be EDUCATIVE - explaining what went wrong AND how to fix it
#   3. Parameter names should indicate expected values (e.g., `path` not `p`)
#
# This pattern ensures agents can effectively use tools even without documentation access.
#
from rlmy.agent.filesystem import (
    edit_file,
    list_files,
    read_pdf_docx_xlsx,
    read_text_file,
    write_file,
)
from rlmy.agent.models import lm, sub_lm
from rlmy.tools.shell import make_shell_tool
from rlmy.tools.vision import peek_image

standard_tools = [
    list_files,
    read_text_file,
    read_pdf_docx_xlsx,
    write_file,
    edit_file,
    make_shell_tool(),
    peek_image,
]

from rlmy.agent.prompt import INITIAL_QUERY, build_signature
from rlmy.tools.mcp import MCPConnector


def _clear_snapshot(snapshot_file: Path | None) -> None:
    """Best-effort delete of the REPL snapshot file (used on /reset)."""
    if snapshot_file is None:
        return
    try:
        snapshot_file.unlink(missing_ok=True)
    except OSError:
        pass


def _repl_recovery_note(interpreter, snapshot_file: Path) -> str:
    """The truthful 'were REPL variables recovered?' line for the recovery panel.

    Snapshotting interpreters (Monty) restore prior REPL state when a snapshot
    file exists; others (Deno builtin) do not and don't declare `restores_state`,
    so the getattr default keeps the message honest. Pure + injectable so it can
    be unit-tested without standing up the whole agent loop.
    """
    if getattr(interpreter, "restores_state", False) and snapshot_file.exists():
        return "✅ REPL variables restored from snapshot."
    return "⚠️  Note: REPL variables from prior session are NOT recovered."


def _dispatch_between_turns(
    sig,
    rlm_context,
    conversation_trajectory: list[dict],
    trajectory_file: Path,
    snapshot_file: Path | None = None,
    interpreter=None,
):
    """
    Purpose: Handle context-dependent slash commands in the between-turns loop.

    Imported functions (compact_trajectory, etc.) are available from the top-level
    import in run_agent(). This function has access to the mutable trajectory list
    and can persist changes.
    """
    from rlmy.agent.trajectory import (
        clear_trajectory,
        compact_trajectory,
        format_compact_stats,
        save_trajectory,
    )

    if sig.action == "compact":
        if not conversation_trajectory:
            rprint(
                Panel(
                    "Trajectory is empty. Nothing to compact.",
                    title="[bold]📦 Compact[/bold]",
                    border_style="blue",
                    padding=(1, 2),
                    width=_panel_width(),
                )
            )
            return

        result = compact_trajectory(conversation_trajectory)
        compacted_traj, stats = result
        conversation_trajectory[:] = compacted_traj
        rlm_context.trajectory = conversation_trajectory
        rlm_context.iteration = len(conversation_trajectory)
        save_trajectory(conversation_trajectory, trajectory_file)

        rprint(
            Panel(
                format_compact_stats(stats),
                title="[bold]📦 Trajectory Compacted[/bold]",
                border_style="blue",
                padding=(1, 2),
                width=_panel_width(),
            )
        )
        return

    if sig.action == "reset":
        conversation_trajectory.clear()
        rlm_context.trajectory = conversation_trajectory
        rlm_context.iteration = 0
        clear_trajectory(trajectory_file)
        # Reset clears the whole session — drop the REPL snapshot AND the live
        # heap, so neither a restart nor the next turn resurrects old variables.
        _clear_snapshot(snapshot_file)
        if interpreter is not None:
            interpreter.shutdown()
        rprint(
            Panel(
                "Trajectory cleared. Starting fresh.",
                title="[bold]🗑️ Reset[/bold]",
                border_style="red",
                padding=(1, 2),
                width=_panel_width(),
            )
        )
        return

    rprint(f"[yellow]Unknown command: {sig.command.name}[/yellow]")


async def run_agent(
    sandbox_root: Path | None = None,
    cache_path: Path | None = None,
    interpreter: str | None = None,
):
    """
    Purpose: Main async entrypoint for MCP-enabled agent with interrupt handling
             and conversation continuity.

    Args:
        sandbox_root: Override sandbox directory (default: ./sandbox, env: RLM_SANDBOX_ROOT)
        cache_path: Override cache file (default: sandbox/.cache.json, env: RLM_CACHE_PATH)
        interpreter: Which code interpreter to use ("monty" default, or "deno"/"pyodide").
            None resolves via RLM_INTERPRETER env then the default.

    Conventions:
    - Workspace selected at startup via SandboxManager interactive picker
    - FILESYSTEM_ROOT set to selected workspace path (module global, used by tool functions)
    - Trajectory always persisted to workspace/trajectory.jsonl
    - Uses MCPConnector context manager to maintain persistent connections
    - Uses InterruptableRLM for cooperative SIGINT + prior trajectory injection
    - Persistent interpreter for REPL state continuity across turns; with Monty +
      a snapshot path, that state also survives process restarts.
    """
    from rlmy.agent.commands import SlashCommandSignal
    from rlmy.agent.interpreters import make_interpreter, resolve_interpreter_kind
    from rlmy.agent.rlm import (
        DEFAULT_CONTEXTUAL_TOOLS,
        InterruptableRLM,
        RLMContext,
        build_status_header,
        make_contextual_tools,
        prompt_user,
    )
    from rlmy.agent.sandbox import SandboxManager
    from rlmy.agent.trajectory import (
        clear_trajectory,
        compact_trajectory,
        load_trajectory,
        save_trajectory,
    )

    # ── Workspace selection ──────────────────────────────────────────────
    manager = SandboxManager(sandbox_root=sandbox_root, cache_path=cache_path)
    workspace = manager.prompt_selection()
    from rlmy.agent.filesystem import set_filesystem_root

    set_filesystem_root(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Trajectory persistence: per-workspace file
    trajectory_file = workspace / "trajectory.jsonl"
    # REPL session snapshot: per-workspace, the other half of session state.
    # Honored by snapshotting interpreters (Monty); ignored by those without (Deno).
    snapshot_file = workspace / "repl.snapshot"
    # One-time migration: rename legacy .json → .jsonl on first startup after upgrade
    _legacy_json = workspace / "trajectory.json"
    if _legacy_json.exists() and not trajectory_file.exists():
        _legacy_json.rename(trajectory_file)
        # Content is still JSON-array format; load_trajectory()'s legacy fallback handles it

    # Show configured models for transparency
    rprint(f"[bold cyan]🤖 Models:[/bold cyan] Main=[yellow]{lm.model}[/yellow] | Sub=[yellow]{sub_lm.model}[/yellow]")
    
    rprint("[bold cyan]🚀 Initializing MCP Tools...[/bold cyan]")

    # Use context manager to maintain MCP connections
    async with MCPConnector() as mcp_tools:
        rprint(f"[dim]MCP tools loaded: {len(mcp_tools)}[/dim]")

        # Persistent interpreter: REPL state survives across turns. With a
        # snapshotting interpreter (Monty) + snapshot_file, it also survives a
        # restart. Non-snapshotting interpreters (Deno) ignore snapshot_path.
        interp_kind = resolve_interpreter_kind(interpreter)
        persistent_repl = make_interpreter(interp_kind, snapshot_path=snapshot_file)
        capabilities_hint = getattr(persistent_repl, "capabilities_hint", "")
        signature = build_signature(capabilities_hint)
        rprint(f"[dim]Interpreter: {interp_kind}[/dim]")

        # Load any existing trajectory from this workspace
        conversation_trajectory: list[dict] = load_trajectory(trajectory_file)
        if conversation_trajectory:
            n = len(conversation_trajectory)
            repl_note = _repl_recovery_note(persistent_repl, snapshot_file)
            rprint(
                Panel(
                    f"Recovered {n} trajectory steps from previous session.\n"
                    f"{repl_note}",
                    title="[bold cyan]🔄 Session Recovered[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                    width=_panel_width(),
                )
            )

        # Mutable context — contextual tools hold a reference, see updates automatically.
        # trajectory_file stored in metadata so _handle_slash_command() can persist
        # to the correct workspace path (not the default prototype/.trajectory_state.jsonl).
        rlm_context = RLMContext(
            trajectory=list(conversation_trajectory),
            max_iterations=100,
            metadata={"trajectory_file": trajectory_file},
        )

        # Wrap contextual tools — LLM sees clean signatures, context injected at call time
        contextual_tools = make_contextual_tools(DEFAULT_CONTEXTUAL_TOOLS, rlm_context)

        def on_trajectory_update(trajectory: list[dict]):
            nonlocal conversation_trajectory
            # Simple pass-through: persist the RLM's canonical history.
            # /compact and /reset work by setting the interrupt flag to end the
            # turn, then compacting result.trajectory post-turn. No mid-callback
            # compaction needed — history is a local variable in forward() and
            # can't be shrunk from here.
            conversation_trajectory = trajectory
            rlm_context.trajectory = trajectory
            rlm_context.iteration = len(trajectory)
            save_trajectory(trajectory, trajectory_file)

        # Merge standard, contextual, and MCP tools
        all_tools = standard_tools + contextual_tools + list(mcp_tools.values())

        # Create interruptable RLM agent with all tools
        rlm = InterruptableRLM(
            signature,
            verbose=True,
            tools=all_tools,
            max_iterations=100,
            max_llm_calls=350,
            sub_lm=sub_lm,
            interpreter=persistent_repl,
            trajectory_callback=on_trajectory_update,
        )
        rlm_context.rlm = rlm

        rprint(f"[dim]Total tools available: {len(all_tools)}[/dim]")

        rprint("[bold cyan]🚀 Starting RLM Processing...[/bold cyan]")

        try:
            # First turn: use initial query
            query = INITIAL_QUERY
            first_turn = True

            while True:
                if not first_turn:
                    try:
                        # Include token/cost status in between-turns prompt.
                        # header is Rich markup, question is Markdown — kept separate.
                        status = build_status_header(rlm_context)
                        query = prompt_user(
                            "What would you like to do next?",
                            header=status,
                        )
                    except (KeyboardInterrupt, EOFError):
                        rprint("\n[bold]Goodbye![/bold]")
                        break
                    except SlashCommandSignal as sig:
                        # Context-dependent command (e.g., /compact, /reset)
                        _dispatch_between_turns(
                            sig,
                            rlm_context,
                            conversation_trajectory,
                            trajectory_file,
                            snapshot_file,
                            persistent_repl,
                        )
                        continue

                    if not query.strip():
                        continue

                # Record user input BEFORE the RLM call (chronological ordering)
                conversation_trajectory.append(
                    {
                        "reasoning": f"<user-input>{query}</user-input>",
                        "code": "",
                        "output": "",
                    }
                )
                # Persist immediately so user-input survives a crash during rlm.acall().
                # Without this, there's a window where the entry exists only in memory
                # until the first RLM iteration callback fires.
                save_trajectory(conversation_trajectory, trajectory_file)

                # Inject prior context so the RLM sees full conversation history
                rlm.set_prior_trajectory(conversation_trajectory)
                with dspy.context(lm=lm):
                    result = await rlm.acall(question=query)

                # Update trajectory from result — RLM's REPLHistory is the canonical source
                conversation_trajectory = list(result.trajectory)

                # Post-turn compaction: /compact set the interrupt flag to end the turn.
                # Now compact result.trajectory and return control to the user.
                if rlm_context.metadata.pop("compaction_requested", False):
                    compacted, _stats = compact_trajectory(conversation_trajectory)
                    conversation_trajectory = compacted
                    rlm_context.trajectory = conversation_trajectory
                    save_trajectory(conversation_trajectory, trajectory_file)
                    rprint("[bold cyan]📦 Trajectory compacted.[/bold cyan]")
                    first_turn = False
                    continue  # → between-turns prompt, user decides what's next

                # Post-turn reset: clear everything and restart fresh.
                if rlm_context.metadata.pop("reset_requested", False):
                    conversation_trajectory = []
                    rlm_context.trajectory = conversation_trajectory
                    rlm_context.iteration = 0
                    clear_trajectory(trajectory_file)
                    # Drop both halves of the session: disk snapshot AND the live
                    # REPL heap, so a fresh start doesn't keep old variables around.
                    _clear_snapshot(snapshot_file)
                    persistent_repl.shutdown()
                    first_turn = False
                    continue

                # Persist after each turn completes
                save_trajectory(conversation_trajectory, trajectory_file)

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

                first_turn = False
        finally:
            persistent_repl.shutdown()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sandbox/cache path overrides."""
    parser = argparse.ArgumentParser(description="MCP-enabled RLM agent")
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Override sandbox root directory (env: RLM_SANDBOX_ROOT)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override cache file path (env: RLM_CACHE_PATH)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_agent(sandbox_root=args.sandbox_root, cache_path=args.cache_path))
