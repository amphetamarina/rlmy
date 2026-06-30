"""
Purpose: Subscription sign-in actions that populate the OAuth credential store.
Usage: login_with_codex(AuthStore())  # imports an existing `codex login`
Key Components: login_with_codex (ChatGPT via Codex CLI import)
Conventions: Sign-in is delegated to the official `codex login` (which writes
             ~/.codex/auth.json); rlmy imports those tokens once and refreshes
             them independently thereafter. No browser flow is implemented here.
"""

from __future__ import annotations

from pathlib import Path

from rlmy.auth.openai_codex import CODEX_AUTH_FILE, import_codex_cli_auth
from rlmy.auth.store import AuthStore, OAuthToken

CHATGPT_PROVIDER = "chatgpt-oauth"


def login_with_codex(store: AuthStore, codex_path: Path = CODEX_AUTH_FILE) -> OAuthToken:
    """
    Purpose: Import an existing Codex CLI ChatGPT login into rlmy's auth store.
    Usage Patterns: Requires the user to have run `codex login` first (the official
        OpenAI flow that writes ~/.codex/auth.json). Raises RuntimeError naming
        that step when no Codex credentials are found.
    """
    token = import_codex_cli_auth(codex_path)
    if token is None:
        raise RuntimeError(
            f"No Codex CLI login found at {codex_path}. "
            "Run `codex login` and sign in with your ChatGPT account, "
            "then re-run `rlmy auth login chatgpt`."
        )
    store.set(CHATGPT_PROVIDER, token)
    return token
