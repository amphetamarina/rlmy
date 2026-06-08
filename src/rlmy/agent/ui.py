import shutil

from rich.console import Console
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
