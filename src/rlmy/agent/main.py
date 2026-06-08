"""
Purpose: MCP-enabled RLM agent with interrupt handling and conversation continuity.
Usage: python mainmcp.py [--sandbox-root PATH] [--cache-path PATH]
Key Components: MCPConnector (persistent MCP connections), InterruptableRLM (from rlmy.agent.rlm),
                SandboxManager (workspace selection), conversation loop with cooperative SIGINT.
Conventions: Ctrl+C during RLM sets flag (checked at iteration boundary).
             Ctrl+C at the prompt exits the program.
             Workspace selected at startup via interactive picker (sandbox_manager.py).
             Trajectory always persisted to workspace/trajectory.jsonl (JSONL format).
"""

import argparse
import asyncio
import os
import readline  # This often resolves the character limit for input()
import shutil
import signal
from pathlib import Path
from textwrap import dedent

import dspy
import nest_asyncio
from markitdown import MarkItDown
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.theme import Theme

# Themed console: inline `code` in Markdown renders as bold coral instead of
# the default "bold cyan on black" which is hard to read in dark terminals.
_MD_THEME = Theme({"markdown.code": "bold #e06c75"})
rprint = Console(theme=_MD_THEME).print

# Max panel width: 120 chars or terminal width, whichever is smaller.
_MAX_PANEL_WIDTH = 120

def _panel_width() -> int:
    """Panel width capped at _MAX_PANEL_WIDTH, respecting terminal size."""
    return min(_MAX_PANEL_WIDTH, shutil.get_terminal_size().columns)

readline
nest_asyncio.apply()  # Allow nested event loops for sync MCP calls

# Required for RLM verbose output (reasoning + code at each iteration).
# InterruptableRLM's forward/aforward use logger.info() when verbose=True,
# but Python's logging module silently drops messages if no handler is configured.
# Each entrypoint (mainmcp.py, cli_proto.py main()) must call basicConfig itself.
# RichHandler gives syntax-highlighted code blocks and styled log output.
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=Console(theme=_MD_THEME), show_path=False, show_time=False, rich_tracebacks=True)],
)


OPUS = "bedrock/us.anthropic.claude-opus-4-6-v1"
SONNET = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Model selection: env vars override hardcoded defaults
_main_model = os.getenv('RLM_MAIN_MODEL', OPUS)
_sub_model = os.getenv('RLM_SUB_MODEL', SONNET)

# Main LM (strategist) — drives RLM reasoning + code generation
lm = dspy.LM(model=_main_model, cache=True)

# Sub-LM (worker) — used by llm_query()/llm_query_batched() inside the REPL
sub_lm = dspy.LM(model=_sub_model, cache=True)

dspy.settings.configure(lm=lm)

# prompt_user is centralized in cli_proto — imported in run_agent() alongside other cli_proto components.
# Do NOT duplicate it here. See prototype/cli_proto.py:141.


# FILESYSTEM_ROOT is set at runtime by run_agent() after workspace selection.
# Initialized to None as a sentinel — tool functions must not be called before run_agent().
FILESYSTEM_ROOT: Path | None = None

# Global state for trusted filesystem roots (outside sandbox)
TRUSTED_ROOTS = set()

# Trust clipboard image directory (user already consented by pasting via Ctrl+\)
from rlmy.agent.clipboard import RLMY_IMAGE_DIR
TRUSTED_ROOTS.add(RLMY_IMAGE_DIR.resolve())


def _require_filesystem_root() -> Path:
    """
    Purpose: Guard against tool calls before workspace selection.
    Why: FILESYSTEM_ROOT is None until run_agent() sets it. Without this guard,
         tool functions crash with confusing TypeError on `None / path`.
    """
    if FILESYSTEM_ROOT is None:
        raise RuntimeError(
            "FILESYSTEM_ROOT is not set. Call run_agent() first to select a workspace."
        )
    return FILESYSTEM_ROOT

# Read-before-write guard: tracks resolved paths that have been read.
# write_file blocks overwriting existing files that haven't been read first.
# Why: prevents the LLM from blindly overwriting files it hasn't inspected.
_FILES_READ: set[str] = set()


def _request_filesystem_permission(path: Path, action: str) -> tuple[bool, str]:
    """
    Purpose: Request user permission for filesystem access outside sandbox

    Args:
        path: Absolute path being accessed
        action: Description of action (e.g., "read file", "list directory")

    Returns:
        Tuple of (granted: bool, user_response: str)
    """
    # Check if path is within filesystem root - no permission needed
    fs_root = _require_filesystem_root()
    try:
        path.resolve().relative_to(fs_root.resolve())
        return (True, "")  # Within sandbox, always allowed
    except ValueError:
        pass  # Outside sandbox, continue to permission check

    # Determine the root for this request
    if path.is_dir():
        request_root = path.resolve()
    else:
        request_root = path.resolve().parent

    # Check if this root or any parent is already trusted
    for trusted_root in TRUSTED_ROOTS:
        try:
            request_root.relative_to(trusted_root)
            return (True, "")  # request_root is under a trusted root
        except ValueError:
            continue  # Not under this trusted root, try next

    # Request permission
    rprint(f"\n[bold yellow]⚠️  Permission needed to {action}:[/bold yellow]")
    rprint(f"[dim]{path.resolve()}[/dim]")
    rprint("[dim]Type 'y' to allow once, 't' to trust this directory:[/dim]")

    # Temporarily restore default SIGINT so Ctrl+C works during input.
    # During RLM execution, our cooperative SIGINT handler sets a flag instead of
    # raising KeyboardInterrupt, which makes bare input() hang.
    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        response = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        rprint("[bold red]✗ Access denied (cancelled)[/bold red]\n")
        return (False, "cancelled by user")
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if response == "y":
        rprint("[bold green]✓ Allowed once[/bold green]\n")
        return (True, "")
    elif response == "t":
        TRUSTED_ROOTS.add(request_root)
        rprint(f"[bold green]✓ Trusted: {request_root}[/bold green]\n")
        return (True, "")
    else:
        rprint("[bold red]✗ Access denied[/bold red]\n")
        return (False, response)


# Common directories to exclude from file listings
EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".svn",
    "node_modules",
    ".npm",
    ".cache",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "htmlcov",
}


def _should_exclude_path(file_path: Path) -> bool:
    """Check if path contains any excluded directories"""
    return any(excluded in file_path.parts for excluded in EXCLUDED_DIRS)


def list_files(path: str = ""):
    """
    Purpose: List all files in a directory (excludes common noise like .venv, node_modules)
    Attributes: Sandbox access is unrestricted; external access requires permission
    Usage Patterns: Pass absolute path for system-wide, relative for sandbox

    Args:
        path: Absolute path (e.g., "/Users/...") or relative within sandbox (e.g., "notes/")

    Returns:
        List of file paths as strings, or error string if operation fails
    """
    # Determine if absolute or relative path
    if path.startswith("/"):
        # Absolute path - system-wide access with permission
        target_path = Path(path)

        if not target_path.exists():
            raise RuntimeError(f"Directory not found: {path}")

        if not target_path.is_dir():
            raise RuntimeError(f"Path is not a directory: {path}")

        # Request permission if outside sandbox
        granted, user_response = _request_filesystem_permission(
            target_path, "list directory"
        )
        if not granted:
            msg = f"Permission denied to list: {path}"
            if user_response:
                msg += f" (user said: {user_response})"
            raise RuntimeError(msg)

        try:
            files = []
            for file_path in target_path.rglob("*"):
                if file_path.is_file() and not _should_exclude_path(file_path):
                    files.append(str(file_path))
            return sorted(files) if files else []
        except Exception as e:
            raise RuntimeError(f"Could not list directory: {str(e)}") from e
    else:
        # Relative path - sandbox access
        fs_root = _require_filesystem_root()
        files = []
        search_root = fs_root / path if path else fs_root

        for file_path in search_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Sandbox doesn't need excluded dirs check (it's controlled)
            rel_path = file_path.relative_to(fs_root)
            files.append(str(rel_path))

        return sorted(files) if files else []


def read_text_file(path: str) -> str:
    """
    Purpose: Read PLAIN TEXT file contents (txt, md, py, json, yaml, csv, html, xml, log, etc.)

    Why this name: Agents only see signatures. 'read_text_file' distinguishes from
    'read_pdf_docx_xlsx' which handles binary document formats.

    Attributes: Sandbox access is unrestricted; external access requires permission
    Usage Patterns: Pass absolute path for system-wide, relative for sandbox

    Args:
        path: Absolute path (e.g., "/Users/.../file.md") or relative (e.g., "notes/todo.md")

    Returns:
        File contents as string, or error message if file doesn't exist
    """
    # Determine if absolute or relative path
    if path.startswith("/"):
        # Absolute path - system-wide access with permission
        file_path = Path(path)

        if not file_path.exists():
            raise RuntimeError(f"File not found: {path}")

        if not file_path.is_file():
            raise RuntimeError(f"Path is not a file: {path}")

        # Request permission if outside sandbox
        granted, user_response = _request_filesystem_permission(file_path, "read file")
        if not granted:
            msg = f"Permission denied to read: {path}"
            if user_response:
                msg += f" (user said: {user_response})"
            raise RuntimeError(msg)

        try:
            content = file_path.read_text(encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
            return content
        except Exception as e:
            raise RuntimeError(f"Could not read file {path}: {str(e)}") from e
    else:
        # Relative path - sandbox access
        fs_root = _require_filesystem_root()
        file_path = fs_root / path

        # Security: Ensure path doesn't escape the root
        try:
            file_path.resolve().relative_to(fs_root.resolve())
        except ValueError:
            raise RuntimeError(f"Invalid path (attempted path traversal): {path}")

        if not file_path.exists():
            raise RuntimeError(f"File not found: {path}")

        if not file_path.is_file():
            raise RuntimeError(f"Path is not a file: {path}")

        try:
            content = file_path.read_text(encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
            return content
        except Exception as e:
            raise RuntimeError(f"Could not read file {path}: {str(e)}") from e


def write_file(path: str, contents: str) -> str:
    """
    Purpose: Write contents to a file (sandbox or system-wide with permission)
    Attributes: Creates parent directories if needed, prevents path traversal in sandbox
    Usage Patterns: Relative path for sandbox, absolute path for system files (requires permission)

    Args:
        path: Absolute path (e.g., "/Users/.../file.md") or relative (e.g., "output/result.md")
        contents: Content to write to the file

    Returns:
        Success message or error description
    """
    if path.startswith("/"):
        # Absolute path — system-wide access with permission
        file_path = Path(path)

        # Read-before-write guard (same logic as sandbox branch)
        if file_path.exists() and str(file_path.resolve()) not in _FILES_READ:
            raise RuntimeError(
                f"Cannot overwrite '{path}' — you haven't read it yet. "
                f"Use read_text_file('{path}') first to inspect current contents, "
                f"then call write_file again."
            )

        # Request permission if outside sandbox
        granted, user_response = _request_filesystem_permission(file_path, "write file")
        if not granted:
            msg = f"Permission denied to write: {path}"
            if user_response:
                msg += f" (user said: {user_response})"
            raise RuntimeError(msg)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents, encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
            return f"SUCCESS: File written to {path}"
        except Exception as e:
            raise RuntimeError(f"Could not write file {path}: {str(e)}") from e
    else:
        # Relative path — sandbox access
        fs_root = _require_filesystem_root()
        file_path = fs_root / path

        # Security: Ensure path doesn't escape the root
        try:
            file_path.resolve().relative_to(fs_root.resolve())
        except ValueError:
            raise RuntimeError(f"Invalid path (attempted path traversal): {path}")

        # Read-before-write guard: block overwriting existing files that haven't been read.
        # New files (don't exist yet) are always allowed — the guard only protects existing content.
        if file_path.exists() and str(file_path.resolve()) not in _FILES_READ:
            raise RuntimeError(
                f"Cannot overwrite '{path}' — you haven't read it yet. "
                f"Use read_text_file('{path}') first to inspect current contents, "
                f"then call write_file again."
            )

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents, encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
            return f"SUCCESS: File written to {path}"
        except Exception as e:
            raise RuntimeError(f"Could not write file {path}: {str(e)}") from e


def edit_file(path: str, old: str, new: str) -> str:
    """
    Purpose: Apply a single find-and-replace edit to a file (sandbox or system-wide with permission).
    Uses fuzzy matching (8 strategies) so minor whitespace differences are tolerated.

    Workflow: read file → find old text → replace with new → write back.

    Args:
        path: Absolute path (e.g., "/Users/.../file.py") or relative (e.g., "output/result.md")
        old: Text to find (include enough context lines for a unique match)
        new: Replacement text

    Returns:
        Success message with replacement stats, or error description
    """
    if path.startswith("/"):
        # Absolute path — system-wide access with permission
        file_path = Path(path)

        if not file_path.exists():
            raise RuntimeError(f"File not found: {path}")

        if not file_path.is_file():
            raise RuntimeError(f"Path is not a file: {path}")

        # Request permission if outside sandbox
        granted, user_response = _request_filesystem_permission(file_path, "edit file")
        if not granted:
            msg = f"Permission denied to edit: {path}"
            if user_response:
                msg += f" (user said: {user_response})"
            raise RuntimeError(msg)

        try:
            content = file_path.read_text(encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
        except Exception as e:
            raise RuntimeError(f"Could not read file {path}: {str(e)}") from e

        try:
            new_content = _fuzzy_replace(content, old, new)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        try:
            file_path.write_text(new_content, encoding="utf-8")
            return f"SUCCESS: Replaced text in {path} ({len(old)} → {len(new)} chars)"
        except Exception as e:
            raise RuntimeError(f"Could not write file {path}: {str(e)}") from e
    else:
        # Relative path — sandbox access
        fs_root = _require_filesystem_root()
        file_path = fs_root / path

        # Security: prevent path traversal
        try:
            file_path.resolve().relative_to(fs_root.resolve())
        except ValueError:
            raise RuntimeError(f"Invalid path (attempted path traversal): {path}")

        if not file_path.exists():
            raise RuntimeError(f"File not found: {path}")

        if not file_path.is_file():
            raise RuntimeError(f"Path is not a file: {path}")

        # Read current content
        try:
            content = file_path.read_text(encoding="utf-8")
            _FILES_READ.add(str(file_path.resolve()))
        except Exception as e:
            raise RuntimeError(f"Could not read file {path}: {str(e)}") from e

        # Apply fuzzy find-and-replace
        try:
            new_content = _fuzzy_replace(content, old, new)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        # Write back
        try:
            file_path.write_text(new_content, encoding="utf-8")
            return f"SUCCESS: Replaced text in {path} ({len(old)} → {len(new)} chars)"
        except Exception as e:
            raise RuntimeError(f"Could not write file {path}: {str(e)}") from e


def read_pdf_docx_xlsx(path: str) -> str:
    """
    Purpose: Extract text from BINARY DOCUMENTS (PDF, Word, Excel, PowerPoint, images)

    Why this name: Agents only see signatures. 'read_pdf_docx_xlsx' explicitly names
    supported binary formats, distinguishing from 'read_text_file' for plain text.

    How it works: Uses MarkItDown to convert binary document formats into readable
    markdown text.

    Attributes:
        - Supports: PDF, DOCX, XLSX, PPTX, images (with OCR), HTML, and more
        - Returns content as markdown-formatted text
        - Same permission model as read_text_file (sandbox vs system paths)

    Usage Patterns:
        - read_document("reports/quarterly.pdf") -> sandbox relative path
        - read_document("/Users/.../document.docx") -> absolute system path (needs permission)

    IMPORTANT - Agent Visibility Pattern:
        Agents using this tool only see the function SIGNATURE, not this docstring.
        Therefore: function name must be self-explanatory, and error messages must
        be educative (guiding the agent on what went wrong and how to fix it).

    Args:
        path: Absolute path (e.g., "/Users/.../file.pdf") or relative within sandbox

    Returns:
        Document text content as markdown, or descriptive error message
    """
    # Determine if absolute or relative path
    if path.startswith("/"):
        # Absolute path - system-wide access with permission
        file_path = Path(path)

        if not file_path.exists():
            raise RuntimeError(
                f"Document not found at '{path}'. "
                f"Verify the path is correct. Use list_files() to discover available files."
            )

        if not file_path.is_file():
            raise RuntimeError(
                f"Path '{path}' is a directory, not a file. "
                f"Use list_files('{path}') to see files inside this directory."
            )

        # Request permission if outside sandbox
        granted, user_response = _request_filesystem_permission(
            file_path, "read document"
        )
        if not granted:
            msg = (
                f"Permission denied to read document at '{path}'. "
                f"The user declined access to this location."
            )
            if user_response:
                msg += f" User message: {user_response}"
            raise RuntimeError(msg)
    else:
        # Relative path - sandbox access
        fs_root = _require_filesystem_root()
        file_path = fs_root / path

        # Security: Ensure path doesn't escape the root
        try:
            file_path.resolve().relative_to(fs_root.resolve())
        except ValueError:
            raise RuntimeError(
                f"Invalid path '{path}' - attempted to escape sandbox. "
                f"Relative paths must stay within the filesystem sandbox. "
                f"Use absolute paths (starting with '/') for system-wide access."
            )

        if not file_path.exists():
            raise RuntimeError(
                f"Document not found at '{path}' (sandbox path). "
                f"Use list_files() to see available files in the sandbox."
            )

        if not file_path.is_file():
            raise RuntimeError(
                f"Path '{path}' is a directory. "
                f"Use list_files('{path}') to see its contents."
            )

    # Convert document to markdown using MarkItDown
    try:
        md = MarkItDown()
        result = md.convert(str(file_path.resolve()))

        if not result.text_content or result.text_content.strip() == "":
            raise RuntimeError(
                f"Document at '{path}' was processed but appears empty. "
                f"This could mean: (1) the file is genuinely empty, "
                f"(2) the file format is not fully supported, or "
                f"(3) the content is in a format MarkItDown cannot extract (e.g., scanned image without OCR)."
            )

        _FILES_READ.add(str(file_path.resolve()))
        return result.text_content

    except RuntimeError:
        raise  # Re-raise our own RuntimeErrors (e.g., empty document above)
    except Exception as e:
        error_msg = str(e)

        # Provide educative error messages based on common failure modes
        if "password" in error_msg.lower() or "encrypted" in error_msg.lower():
            raise RuntimeError(
                f"Document '{path}' appears to be password-protected or encrypted. "
                f"MarkItDown cannot process protected documents. "
                f"Ask the user to provide an unprotected version."
            ) from e
        elif "corrupt" in error_msg.lower() or "invalid" in error_msg.lower():
            raise RuntimeError(
                f"Document '{path}' appears to be corrupted or has an invalid format. "
                f"The file may be damaged or not actually the format its extension suggests. "
                f"Original error: {error_msg}"
            ) from e
        else:
            # Get file extension for format-specific guidance
            suffix = file_path.suffix.lower()
            supported_formats = (
                "PDF, DOCX, XLSX, PPTX, HTML, images (PNG, JPG), and more"
            )

            raise RuntimeError(
                f"Failed to extract text from '{path}' (format: {suffix}). "
                f"Supported formats include: {supported_formats}. "
                f"If this format should be supported, the file may be corrupted. "
                f"For plain text files, use read_file() instead. "
                f"Technical details: {error_msg}"
            ) from e


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
from rlmy.tools.shell import make_shell_tool
from rlmy.tools.edit import replace as _fuzzy_replace
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

from rlmy.tools.mcp import MCPConnector


class LongContextWithDictQA(dspy.Signature):
    """Answer the question using the provided context and data dictionary.

    Communication Tools:
    - Use `broadcast_user_update` (1-way) between significant chunks of work (not within one block)
    - Use `ask_user_guidance` (2-way) when you need human decision or are unsure

    Filesystem Tools:
    - `list_files(path="")` - Returns list of file paths (Python list)
    - `read_file(path="example.md")` - Read from sandbox (relative) or system (absolute path)
    - `write_file(path="output/result.md", contents="...")` - Write to sandbox (relative) or system (absolute path)
    - `edit_file(path="output/result.md", old="text to find", new="replacement")` - Surgical find-and-replace, sandbox (relative) or system (absolute path)

    **Important**: Take an out if necessary. You are allowed to not know the answer.
    Feel free to say "I don't know" whenever applicable.
    Ask for user guidance whenever you're unsure.

    Also, any `print()`s you do are for your own internal reasoning. Assume I can't see it.
    Use broadcast_user_update to report findings (non-blocking).
    When you're done, always ask for the next task (there is always a next task).
    Use the ask_user_guidance tool (blocking) for that. otherwise you will be stuck in an infinite loop.

    **CRITICAL — ask_user_guidance isolation rule**:
    NEVER call `ask_user_guidance` in a code block that also prints large amounts of data
    (e.g., file contents, LLM outputs, inventories). The user's response gets buried at the
    end of your code block's output, and you WILL miss it on the next iteration.
    Always call `ask_user_guidance` in its own SHORT, DEDICATED code block — ideally with
    no other print() statements. If you need to show the user data AND ask a question,
    use `broadcast_user_update` first in one code block, then `ask_user_guidance` alone
    in the NEXT code block.

    **CRITICAL — fresh user directives override stale `question` variable**:
    The `question` variable contains the ORIGINAL input from the start of this session.
    For ongoing direction, ALWAYS prioritize the most recent `ask_user_guidance` return
    value in your REPL history over the original `question`. The user's latest response
    is their current intent — the `question` variable may be hours old.
    """

    question: str = dspy.InputField(desc="Your starting point")
    answer: str = dspy.OutputField(desc="Concise final answer")


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
    from rlmy.agent.trajectory import compact_trajectory, format_compact_stats, save_trajectory
    from rlmy.agent.trajectory import clear_trajectory

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
        save_trajectory(conversation_trajectory, trajectory_file)

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
        clear_trajectory(trajectory_file)
        # Reset clears the whole session — drop the REPL snapshot AND the live
        # heap, so neither a restart nor the next turn resurrects old variables.
        _clear_snapshot(snapshot_file)
        if interpreter is not None:
            interpreter.shutdown()
        rprint(Panel(
            "Trajectory cleared. Starting fresh.",
            title="[bold]🗑️ Reset[/bold]",
            border_style="red",
            padding=(1, 2),
            width=_panel_width(),
        ))
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
    global FILESYSTEM_ROOT

    from rlmy.agent.rlm import (
        InterruptableRLM,
        RLMContext,
        DEFAULT_CONTEXTUAL_TOOLS,
        make_contextual_tools,
        prompt_user,
        build_status_header,
    )
    from rlmy.agent.interpreters import make_interpreter, resolve_interpreter_kind
    from rlmy.agent.trajectory import save_trajectory, load_trajectory, clear_trajectory, compact_trajectory, format_compact_stats
    from rlmy.agent.commands import SlashCommandSignal
    from rlmy.agent.sandbox import SandboxManager

    # ── Workspace selection ──────────────────────────────────────────────
    manager = SandboxManager(sandbox_root=sandbox_root, cache_path=cache_path)
    workspace = manager.prompt_selection()
    FILESYSTEM_ROOT = workspace
    FILESYSTEM_ROOT.mkdir(parents=True, exist_ok=True)

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

    rprint("[bold cyan]🚀 Initializing MCP Tools...[/bold cyan]")

    # Use context manager to maintain MCP connections
    async with MCPConnector() as mcp_tools:
        rprint(f"[dim]MCP tools loaded: {len(mcp_tools)}[/dim]")

        # Persistent interpreter: REPL state survives across turns. With a
        # snapshotting interpreter (Monty) + snapshot_file, it also survives a
        # restart. Non-snapshotting interpreters (Deno) ignore snapshot_path.
        interp_kind = resolve_interpreter_kind(interpreter)
        persistent_repl = make_interpreter(interp_kind, snapshot_path=snapshot_file)
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
            LongContextWithDictQA,
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

        initial_query = dedent("""
                       Start by reading the file `AGENTS.md` in full.
                       You will obey those instructions.
                       Start by doing:
                       ```python
                       instructions = read_text_file("AGENTS.md")
                       print(instructions)
                       ```
                       **important**: do NOT use `os.path` to read files. They wont exist!! read_text_file works on a virtual remote filesystem.
                       **important**: take an out if necessary. you are allowed to not know the answer. feel free to say "I don't know" whenever applicable.
                       I HAVE A NEW QUESTION
                       IMPORTANT: NEVER use the `SUBMIT` function. let me be the judge. always assume a new task is waiting when you ask for it!
                        you should never use shell unless explicitly told to.
                       starty by asking me a question
                        also... assume the journal (as described in AGENTS.md) already exists, careful not to overwrite it and lose data.
                        so ALWAYS READ THE JOURNAL IN FULL. not just pieces. IN FULL.
        """).strip()

        rprint("[bold cyan]🚀 Starting RLM Processing...[/bold cyan]")

        try:
            # First turn: use initial query
            query = initial_query
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
                            sig, rlm_context, conversation_trajectory,
                            trajectory_file, snapshot_file, persistent_repl,
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
