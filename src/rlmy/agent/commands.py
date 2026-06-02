"""
Purpose: Slash command registry — decoupled matching, dispatch signaling, help display.
Usage:
    from rlmy.agent.commands import REGISTRY, SlashCommandSignal
    cmd = REGISTRY.match(user_text)       # returns SlashCommand or None
    raise SlashCommandSignal(cmd)         # for context-dependent commands
Key Components:
    SlashCommand — immutable command descriptor (name, aliases, action key, description)
    SlashCommandSignal — exception raised when a context-dependent command is triggered
    CommandRegistry — register, match, help panel
    REGISTRY — module-level singleton with all commands pre-registered
Conventions:
    Commands return action keys (strings), NOT callbacks.
    Self-contained commands (exit, help) are handled by prompt_user() directly.
    Context-dependent commands (compact) raise SlashCommandSignal for the caller to handle.
"""

from dataclasses import dataclass, field
from typing import Optional

from rich.panel import Panel
from rich.text import Text


@dataclass(frozen=True)
class SlashCommand:
    """
    Purpose: Immutable descriptor for a slash command.
    Attributes:
        name: Primary command string (e.g., "/compact")
        aliases: Alternative triggers (e.g., ["/c"])
        action: Action key dispatched to callers (e.g., "compact"). NOT a callback.
        description: Human-readable description for /help display.
        self_contained: If True, prompt_user() handles it directly (e.g., exit, help).
                        If False, SlashCommandSignal is raised for the caller.
    """
    name: str
    action: str
    description: str
    aliases: list[str] = field(default_factory=list)
    self_contained: bool = False


class SlashCommandSignal(Exception):
    """
    Purpose: Raised by prompt_user() when user triggers a context-dependent command.

    Why an exception: prompt_user() doesn't own the conversation loop or trajectory.
    It can't execute context-dependent commands itself. The exception propagates to
    the caller (contextual_ask_user_guidance or mainmcp's between-turns loop) which
    has the necessary context to dispatch.

    Attributes:
        command: The matched SlashCommand
        action: Shortcut to command.action for easy dispatch
    """

    def __init__(self, command: SlashCommand):
        self.command = command
        self.action = command.action
        super().__init__(f"Slash command: {command.name}")


class CommandRegistry:
    """
    Purpose: Central registry for slash commands. Match user input, render help.

    Usage Patterns:
        REGISTRY.register(SlashCommand("/foo", "foo", "Do foo"))
        cmd = REGISTRY.match("/foo")   # returns SlashCommand
        cmd = REGISTRY.match("hello")  # returns None
        panel = REGISTRY.help_panel()  # Rich Panel for /help display
    """

    def __init__(self):
        self._commands: list[SlashCommand] = []
        # Lookup: normalized trigger string → SlashCommand
        self._lookup: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        """Register a command. All triggers (name + aliases) are indexed."""
        self._commands.append(cmd)
        for trigger in [cmd.name] + cmd.aliases:
            key = trigger.strip().lower()
            if key in self._lookup:
                raise ValueError(
                    f"Duplicate trigger '{key}': already registered to '{self._lookup[key].name}'"
                )
            self._lookup[key] = cmd

    def match(self, text: str) -> Optional[SlashCommand]:
        """
        Match user input against registered commands.

        Args:
            text: The full user input text (already stripped by prompt_user).

        Returns:
            SlashCommand if the ENTIRE text matches a trigger, else None.
            Partial matches (e.g., "/comp" for "/compact") do NOT match.
        """
        return self._lookup.get(text.lower())

    def help_panel(self) -> Panel:
        """Build a Rich Panel listing all registered commands."""
        lines = Text()
        for cmd in self._commands:
            triggers = ", ".join([cmd.name] + cmd.aliases)
            lines.append(f"  {triggers}", style="bold cyan")
            lines.append(f"  — {cmd.description}\n", style="dim")
        return Panel(
            lines,
            title="[bold]Available Commands[/bold]",
            border_style="blue",
            padding=(1, 2),
        )


# =============================================================================
# Module-level singleton — all commands registered here
# =============================================================================

REGISTRY = CommandRegistry()

REGISTRY.register(SlashCommand(
    name="/quit",
    action="exit",
    description="Exit the program",
    aliases=["/q"],
    self_contained=True,
))

REGISTRY.register(SlashCommand(
    name="/help",
    action="help",
    description="Show available commands",
    aliases=["/h", "/?"],
    self_contained=True,
))

REGISTRY.register(SlashCommand(
    name="/compact",
    action="compact",
    description="Compact trajectory to free LLM attention (irreversible)",
    aliases=["/c"],
    self_contained=False,
))

REGISTRY.register(SlashCommand(
    name="/reset",
    action="reset",
    description="Clear trajectory and start fresh",
    aliases=[],
    self_contained=False,
))
