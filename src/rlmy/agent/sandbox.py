"""
Purpose: Workspace discovery, selection, creation, and cache management for RLM sandboxes.
Usage:
    from rlmy.agent.sandbox import SandboxManager
    manager = SandboxManager()  # uses defaults; override via env or args
    workspace_path = manager.prompt_selection()  # interactive picker
Key Components:
    SandboxManager — discovers workspaces, manages cache, creates new ones from template
Conventions:
    - Cache lives at sandbox_root/.cache.json (overridable via RLM_CACHE_PATH env or arg)
    - Each workspace has: AGENTS.md, session-journal.md, input/, output/, trajectory.jsonl
    - No dependency on mainmcp.py or cli_proto.py — this is a standalone utility
"""

import json
import os
import re
import shutil
from importlib import resources
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel

_MAX_PANEL_WIDTH = 120


def _panel_width() -> int:
    """Panel width capped at _MAX_PANEL_WIDTH, respecting terminal size."""
    return min(_MAX_PANEL_WIDTH, shutil.get_terminal_size().columns)

# Default sandbox root: ~/.config/rlmy/sandboxes/
# Centralized in user's config directory (not relative to package install location)
_DEFAULT_SANDBOX_ROOT = Path.home() / ".config" / "rlmy" / "sandboxes" 

_SESSION_JOURNAL_TEMPLATE = "# Session Journal\n"

# Regex for workspace name sanitization: only allow word chars and hyphens
_VALID_NAME_RE = re.compile(r"[^\w\-]")


def _copy_sample_workspace(dest: Path) -> None:
    """
    Recursively copy the bundled 'sample_workspace/' tree into a new workspace.

    The template folder ships INSIDE the package (rlmy/sample_workspace/) and is
    located via importlib.resources, so it works under any install method
    (editable, `uv tool install`, pip, Docker). Whatever files live in that
    folder get copied automatically — no per-file names hardcoded here.

    Never overwrites existing files (preserves user data on re-entry).
    """
    try:
        src_root = resources.files("rlmy") / "sample_workspace"
    except (ModuleNotFoundError, AttributeError):
        return  # package data unavailable; ensure_structure handles the journal seed

    def _recurse(src_dir, rel: Path) -> None:
        for entry in src_dir.iterdir():
            target = dest / rel / entry.name
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                _recurse(entry, rel / entry.name)
            else:
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        entry.read_text(encoding="utf-8"), encoding="utf-8"
                    )

    if src_root.is_dir():
        _recurse(src_root, Path("."))


class SandboxManager:
    """
    Purpose: Manages workspace discovery, selection, creation, and cache persistence.

    Attributes:
        sandbox_root: Path to directory containing workspace folders
        cache_path: Path to .cache.json storing last_active workspace name

    Usage Patterns:
        manager = SandboxManager()
        workspace = manager.prompt_selection()  # returns Path to selected workspace
        # workspace is now ready with AGENTS.md, session-journal.md, input/, output/
    """

    def __init__(
        self,
        sandbox_root: Path | None = None,
        cache_path: Path | None = None,
    ):
        # Env overrides > explicit args > defaults
        self.sandbox_root = Path(
            os.environ.get("RLM_SANDBOX_ROOT")
            or sandbox_root
            or _DEFAULT_SANDBOX_ROOT
        )
        self.cache_path = Path(
            os.environ.get("RLM_CACHE_PATH")
            or cache_path
            or (self.sandbox_root / ".cache.json")
        )
        # Ensure sandbox root exists
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    # ── Cache ────────────────────────────────────────────────────────────

    def _read_cache(self) -> dict:
        """Read cache file. Returns empty dict on any failure."""
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _write_cache(self, data: dict) -> None:
        """Atomic write of cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(self.cache_path)

    def get_last_active(self) -> str | None:
        """Return the name of the last active workspace, or None."""
        return self._read_cache().get("last_active")

    def set_active(self, name: str) -> None:
        """Update cache to mark workspace as last active."""
        cache = self._read_cache()
        cache["last_active"] = name
        self._write_cache(cache)

    # ── Workspace discovery ──────────────────────────────────────────────

    def list_workspaces(self) -> list[str]:
        """
        Return sorted list of workspace folder names in sandbox_root.
        Excludes hidden files/folders (starting with '.').
        """
        if not self.sandbox_root.exists():
            return []
        return sorted(
            d.name
            for d in self.sandbox_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    # ── Workspace creation ───────────────────────────────────────────────

    def ensure_structure(self, name: str) -> Path:
        """
        Ensure workspace has the expected directory structure and template files.
        Creates missing dirs/files but never overwrites existing ones.

        Returns:
            Path to the workspace root directory.
        """
        ws = self.sandbox_root / name
        ws.mkdir(parents=True, exist_ok=True)

        # Runtime directories (not part of the bundled template folder)
        (ws / "input").mkdir(exist_ok=True)
        (ws / "output").mkdir(exist_ok=True)

        # Copy the bundled sample_workspace/ tree (AGENTS.md, skills/, etc.)
        # Recursive + data-driven: whatever ships in the folder lands here.
        # Never overwrites existing files (preserves user data on re-entry).
        _copy_sample_workspace(ws)

        # Session journal seed — a runtime file, not part of the template folder.
        journal_path = ws / "session-journal.md"
        if not journal_path.exists():
            journal_path.write_text(_SESSION_JOURNAL_TEMPLATE, encoding="utf-8")

        return ws

    def create_workspace(self, name: str) -> Path:
        """Create a new workspace with template structure and mark as active."""
        ws = self.ensure_structure(name)
        self.set_active(name)
        return ws

    # ── Interactive selection ─────────────────────────────────────────────

    @staticmethod
    def _sanitize_name(raw_name: str) -> str | None:
        """
        Sanitize workspace name: lowercase, replace non-word/non-hyphen chars with underscore.
        Returns None if the result is empty or starts with a dot (hidden).

        Why strict: workspace name becomes a directory under sandbox_root.
        Path traversal via '../' or special chars must be impossible.
        """
        name = _VALID_NAME_RE.sub("_", raw_name.strip().lower())
        # Strip leading/trailing underscores and collapse runs
        name = re.sub(r"_+", "_", name).strip("_")
        if not name or name.startswith("."):
            return None
        return name

    def prompt_selection(self) -> Path:
        """
        Interactive workspace picker with Rich UI.

        Shows numbered list of existing workspaces, highlights last active,
        and offers a "Create new" option. Returns the Path to the selected
        (or newly created) workspace.
        """
        workspaces = self.list_workspaces()
        last_active = self.get_last_active()

        # Build display
        lines = []
        default_idx = None

        for i, name in enumerate(workspaces, start=1):
            marker = " ← last active" if name == last_active else ""
            lines.append(f"  {i}. {name}{marker}")
            if name == last_active:
                default_idx = i

        new_idx = len(workspaces) + 1
        lines.append(f"  {new_idx}. [bold]Create new workspace[/bold]")

        # If no last_active match, default to first workspace (or create new if empty)
        if default_idx is None:
            default_idx = 1 if workspaces else new_idx

        header = f"[dim]sandbox root: {self.sandbox_root.resolve()}[/dim]\n\n"
        body = header + "\n".join(lines)

        rprint(Panel(
            body,
            title="[bold cyan]🗂️  Workspace Selection[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            width=_panel_width(),
        ))

        # Get user choice
        try:
            raw = input(f"  Enter number [{default_idx}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            rprint("\n[bold]Goodbye![/bold]")
            raise SystemExit(0)

        try:
            choice = int(raw) if raw else default_idx
        except ValueError:
            rprint(f"[red]Invalid input: '{raw}'. Enter a number.[/red]")
            raise SystemExit(1)

        # Handle "Create new"
        if choice == new_idx:
            try:
                raw_name = input("  Workspace name: ").strip()
            except (KeyboardInterrupt, EOFError):
                rprint("\n[bold]Goodbye![/bold]")
                raise SystemExit(0)

            name = self._sanitize_name(raw_name)
            if not name:
                rprint(f"[yellow]Invalid workspace name: '{raw_name}'. Aborting.[/yellow]")
                raise SystemExit(1)

            if name in workspaces:
                rprint(f"[yellow]Workspace '{name}' already exists. Selecting it.[/yellow]")
            else:
                rprint(f"[green]Creating workspace: {name}[/green]")

            ws = self.create_workspace(name)
            rprint(f"[bold green]✓ Active workspace: {name}[/bold green]\n")
            return ws

        # Handle existing workspace selection
        if 1 <= choice <= len(workspaces):
            name = workspaces[choice - 1]
            ws = self.ensure_structure(name)
            self.set_active(name)
            rprint(f"[bold green]✓ Active workspace: {name}[/bold green]\n")
            return ws

        rprint(f"[red]Invalid choice: {choice}[/red]")
        raise SystemExit(1)
