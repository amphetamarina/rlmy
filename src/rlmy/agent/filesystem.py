"""
Purpose: Sandboxed filesystem tools + the permission model that gates access
         outside the workspace. Extracted from agent/main.py as a cohesive seam.

State: FILESYSTEM_ROOT (the active workspace) is set once at startup by run_agent()
via set_filesystem_root(); tool functions read it through _require_filesystem_root().
TRUSTED_ROOTS / _FILES_READ track per-session permission + read-before-write state.
"""

import signal
from pathlib import Path

from markitdown import MarkItDown

from rlmy.agent.clipboard import RLMY_IMAGE_DIR
from rlmy.agent.ui import rprint
from rlmy.tools.edit import replace as _fuzzy_replace


def set_filesystem_root(path: Path) -> None:
    """Set the active workspace root. Called once by run_agent() at startup.

    FILESYSTEM_ROOT lived as a module global in main.py and was mutated there via
    `global`; now that the tools live here, this setter is the cross-module write.
    """
    global FILESYSTEM_ROOT
    FILESYSTEM_ROOT = path


# FILESYSTEM_ROOT is set at runtime by run_agent() after workspace selection.
# Initialized to None as a sentinel — tool functions must not be called before run_agent().
FILESYSTEM_ROOT: Path | None = None

# Global state for trusted filesystem roots (outside sandbox)
TRUSTED_ROOTS = set()

# Trust clipboard image directory (user already consented by pasting via Ctrl+\)
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


def _resolve_file_path(path: str, action: str) -> Path:
    """Resolve an absolute or relative path, enforcing traversal guard + permissions.

    Used by the four file-oriented tools (read/write/edit/read_pdf_docx_xlsx) which
    share identical resolution logic. list_files has its own because its relative-path
    semantics differ (returns paths relative to root, filters excluded dirs).

    Absolute paths (starting with "/") go through the permission model.
    Relative paths are resolved against FILESYSTEM_ROOT with a traversal guard.

    Returns the resolved Path. Raises RuntimeError on traversal attempt or
    permission denial.
    """
    if path.startswith("/"):
        file_path = Path(path)
        granted, user_response = _request_filesystem_permission(file_path, action)
        if not granted:
            msg = f"Permission denied to {action}: {path}"
            if user_response:
                msg += f" (user said: {user_response})"
            raise RuntimeError(msg)
        return file_path
    else:
        fs_root = _require_filesystem_root()
        file_path = fs_root / path
        try:
            file_path.resolve().relative_to(fs_root.resolve())
        except ValueError:
            raise RuntimeError(f"Invalid path (attempted path traversal): {path}")
        return file_path


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
    file_path = _resolve_file_path(path, "read file")

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
    file_path = _resolve_file_path(path, "write file")

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
    file_path = _resolve_file_path(path, "edit file")

    if not file_path.exists():
        raise RuntimeError(f"File not found: {path}")
    if not file_path.is_file():
        raise RuntimeError(f"Path is not a file: {path}")

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
    file_path = _resolve_file_path(path, "read document")

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
                f"For plain text files, use read_text_file() instead. "
                f"Technical details: {error_msg}"
            ) from e
