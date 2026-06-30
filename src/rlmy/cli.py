"""
Purpose: Main CLI entrypoint for RLMY.
Usage: Called when user runs 'rlmy' command (via pyproject.toml entry point)
Key Components: main() - config check, wizard, agent execution
Conventions: Auto-runs wizard on first use if config missing and terminal is interactive.
             Sets RLM_MAIN_MODEL/RLM_SUB_MODEL env vars before importing agent module
             so module-level dspy config picks up the correct models.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rlmy",
        description="Interactive RLM agent with MCP tools and iterative code execution.",
    )
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Override sandbox directory (default: ./sandbox, env: RLM_SANDBOX_ROOT)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override cache file (default: sandbox/.cache.json, env: RLM_CACHE_PATH)",
    )
    from .agent.interpreters import INTERPRETER_CHOICES

    parser.add_argument(
        "--interpreter",
        choices=INTERPRETER_CHOICES,
        default=None,
        help="Code interpreter to use (default: monty; env: RLM_INTERPRETER). "
        "'monty' is pip-only with no extra setup; 'deno'/'pyodide' needs Deno installed.",
    )

    sub = parser.add_subparsers(dest="command")
    auth_p = sub.add_parser("auth", help="Manage subscription sign-in (ChatGPT)")
    auth_p.add_argument("action", choices=["login", "logout", "status"])
    auth_p.add_argument("provider", nargs="?", default="chatgpt", choices=["chatgpt"])
    return parser.parse_args()


def _run_auth_command(action: str, provider: str, store=None) -> int:
    """
    Purpose: Dispatch `rlmy auth <action> [provider]` against the credential store.
    Usage Patterns: Returns a process exit code. login raises RuntimeError (caught
        by main) with sign-in instructions when no Codex login exists; store is
        injectable for tests.
    """
    from .auth.login import CHATGPT_PROVIDER, login_with_codex
    from .auth.store import AuthStore

    store = store if store is not None else AuthStore()
    if provider != "chatgpt":
        print(f"Unknown auth provider: {provider}", file=sys.stderr)
        return 1

    if action == "login":
        token = login_with_codex(store)
        plan = f" ({token.plan_type})" if token.plan_type else ""
        print(f"Signed in to ChatGPT{plan}.")
        print("Set your model to 'chatgpt-oauth/gpt-5.5' (or gpt-5.4) to use it.")
        return 0
    if action == "status":
        signed_in = store.get(CHATGPT_PROVIDER) is not None
        print(f"chatgpt: {'signed in' if signed_in else 'not signed in'}")
        return 0
    if action == "logout":
        store.remove(CHATGPT_PROVIDER)
        print("Signed out of ChatGPT.")
        return 0
    print(f"Unknown auth action: {action}", file=sys.stderr)
    return 1


def _ensure_config() -> tuple[str, str]:
    """
    Ensure configuration exists. Returns (main_model, sub_model) strings.
    Runs wizard interactively if no config found.
    Raises RuntimeError if non-interactive and no config.
    """
    from .config import RLMYConfig

    config = RLMYConfig()

    try:
        return config.get_models()
    except RuntimeError as e:
        # No config found — try wizard if interactive
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            # Non-interactive: fail with helpful message
            raise

        print(str(e))
        print()

        try:
            response = input("Run setup wizard now? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            sys.exit(1)

        if response not in ("", "y", "yes"):
            print("Setup cancelled. Run \'rlmy\' again when ready.")
            sys.exit(1)

        # Run wizard
        from .wizard import run_wizard
        success = run_wizard(config.config_dir, config.config_file)

        if not success:
            print("Wizard failed. Please configure manually.", file=sys.stderr)
            sys.exit(1)

        # Try loading config again after wizard
        return config.get_models()


def _ensure_deno_or_exit() -> None:
    """Verify Deno is available (only the Deno/Pyodide interpreter needs it).

    Checks PATH and common install locations; adds the dir to PATH for child
    processes if found there. Exits with install instructions if missing.
    """
    import shutil
    from pathlib import Path

    deno_path = shutil.which("deno")
    if not deno_path:
        common_locations = [
            Path.home() / ".deno" / "bin" / "deno",
            Path("/usr/local/bin/deno"),
        ]
        for loc in common_locations:
            if loc.exists():
                deno_path = str(loc)
                # Add to PATH for child processes (DSPy's PythonInterpreter needs it).
                os.environ["PATH"] = str(loc.parent) + os.pathsep + os.environ.get("PATH", "")
                break

    if not deno_path:
        print("❌ Deno is required for the 'deno' interpreter but was not found.", file=sys.stderr)
        print("", file=sys.stderr)
        print("   Install with:", file=sys.stderr)
        print("     curl -fsSL https://deno.land/install.sh | sh", file=sys.stderr)
        print("", file=sys.stderr)
        print("   Then restart your terminal and run rlmy again.", file=sys.stderr)
        print("   Or use the default interpreter (no Deno needed): rlmy --interpreter monty", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point for RLMY."""
    args = _parse_args()

    if getattr(args, "command", None) == "auth":
        try:
            sys.exit(_run_auth_command(args.action, args.provider))
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    print("Starting RLMY...")

    try:
        # Step 0: Deno is only needed for the Deno/Pyodide interpreter. The
        # default (Monty) is pip-only, so we resolve the choice first and only
        # enforce the Deno requirement when that interpreter is actually selected.
        from .agent.interpreters import resolve_interpreter_kind

        interp_kind = resolve_interpreter_kind(args.interpreter)
        if interp_kind == "deno":
            _ensure_deno_or_exit()

        # Step 1: Ensure config exists (wizard if needed)
        main_model, sub_model = _ensure_config()

        # Step 2: Set env vars BEFORE importing agent module
        # (agent/main.py reads these at module-import time)
        os.environ.setdefault("RLM_MAIN_MODEL", main_model)
        os.environ.setdefault("RLM_SUB_MODEL", sub_model)

        # Step 3: Import and run the agent (deferred to pick up env vars)
        from .agent.main import run_agent

        asyncio.run(run_agent(
            sandbox_root=args.sandbox_root,
            cache_path=args.cache_path,
            interpreter=interp_kind,
        ))

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

