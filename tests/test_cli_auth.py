"""
Tests for the `rlmy auth` command handler (cli._run_auth_command).

Focus: status reflects store state and logout clears it, exercised through the
same function the CLI dispatches to, with an injected store. Login's import path
is covered in test_auth_login.py; we avoid it here so the suite never reads the
real ~/.codex/auth.json.
"""

from rlmy.auth.login import CHATGPT_PROVIDER
from rlmy.auth.store import AuthStore, OAuthToken
from rlmy.cli import _run_auth_command


def test_status_then_logout_roundtrip(tmp_path):
    store = AuthStore(path=tmp_path / "auth.json")
    assert _run_auth_command("status", "chatgpt", store=store) == 0

    store.set(CHATGPT_PROVIDER, OAuthToken(access_token="a", refresh_token="r",
                                           expires_at=0))
    assert _run_auth_command("status", "chatgpt", store=store) == 0
    assert _run_auth_command("logout", "chatgpt", store=store) == 0
    assert store.get(CHATGPT_PROVIDER) is None


def test_unknown_provider_returns_error_code(tmp_path):
    store = AuthStore(path=tmp_path / "auth.json")
    assert _run_auth_command("login", "grok", store=store) == 1
