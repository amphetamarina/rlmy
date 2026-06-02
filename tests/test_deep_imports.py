"""
Test that ALL package modules can be imported without missing dependencies.

This catches the case where pyproject.toml is missing a dependency that a module
imports at the top level. The standard TestModuleImports tests only check
top-level packages (rlmy.agent, rlmy.tools) which only trigger __init__.py,
not the actual module files.

WHY THIS EXISTS:
On 2026-06-02, `rlmy --sandbox-root /tmp/rlmy-fresh-test` crashed with:
    ModuleNotFoundError: No module named 'strands_tools'
because strands-agents-tools was in the old requirements.txt but missing from
pyproject.toml. The existing tests didn't catch it because they only did shallow
imports. This test does DEEP imports of every .py file in the package.
"""

import importlib
import pkgutil
import pytest
import sys
from pathlib import Path


def _find_all_modules(package_name: str) -> list[str]:
    """Discover all importable modules in a package recursively."""
    package = importlib.import_module(package_name)
    package_path = Path(package.__file__).parent

    modules = [package_name]

    for finder, name, ispkg in pkgutil.walk_packages(
        [str(package_path)], prefix=f"{package_name}."
    ):
        modules.append(name)

    return modules


class TestDeepImports:
    """Verify ALL package modules import without errors (catches missing deps)."""

    def test_all_modules_importable(self):
        """
        Import every module in the rlmy package.

        This catches:
        - Missing dependencies in pyproject.toml
        - Broken import paths after file moves
        - Syntax errors in any module
        """
        modules = _find_all_modules("rlmy")

        failures = []
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failures.append(f"{mod_name}: {type(e).__name__}: {e}")

        if failures:
            failure_report = "\n".join(failures)
            pytest.fail(
                f"Failed to import {len(failures)} module(s):\n{failure_report}"
            )

    def test_critical_imports_explicit(self):
        """
        Explicitly import the modules most likely to have external dependencies.
        Gives clearer error messages than the generic walker.
        """
        # These are the modules with third-party imports at module level
        critical_modules = [
            "rlmy.tools.shell",      # strands_tools
            "rlmy.tools.mcp",        # mcp
            "rlmy.tools.edit",       # (stdlib only, but verify)
            "rlmy.agent.rlm",        # dspy, rich, prompt_toolkit
            "rlmy.agent.main",       # dspy, rich, markitdown, nest_asyncio
            "rlmy.agent.trajectory", # (mostly stdlib)
            "rlmy.agent.sandbox",    # rich
            "rlmy.agent.commands",   # rich
            "rlmy.config",           # platformdirs, dspy
            "rlmy.wizard",           # rich
            "rlmy.cli",              # (stdlib only at top level)
        ]

        for mod_name in critical_modules:
            try:
                importlib.import_module(mod_name)
            except ModuleNotFoundError as e:
                pytest.fail(
                    f"Missing dependency when importing {mod_name}: {e}\n"
                    f"→ Add the missing package to pyproject.toml [project.dependencies]"
                )
