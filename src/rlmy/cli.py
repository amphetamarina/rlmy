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
    return parser.parse_args()


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


def main():
    """Main CLI entry point for RLMY."""
    args = _parse_args()

    try:
        # Step 0: Check Deno is installed (required for WASM sandbox)
        import shutil
        deno_path = shutil.which("deno")
        if not deno_path:
            # Check common install locations (user may not have restarted shell)
            from pathlib import Path
            common_locations = [
                Path.home() / ".deno" / "bin" / "deno",
                Path("/usr/local/bin/deno"),
            ]
            for loc in common_locations:
                if loc.exists():
                    deno_path = str(loc)
                    # Add to PATH for child processes (dspy's PythonInterpreter needs it)
                    os.environ["PATH"] = str(loc.parent) + os.pathsep + os.environ.get("PATH", "")
                    break

        if not deno_path:
            print("❌ Deno is required but not found.", file=sys.stderr)
            print("", file=sys.stderr)
            print("   Install with:", file=sys.stderr)
            print("     curl -fsSL https://deno.land/install.sh | sh", file=sys.stderr)
            print("", file=sys.stderr)
            print("   Then restart your terminal and run rlmy again.", file=sys.stderr)
            sys.exit(1)

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

