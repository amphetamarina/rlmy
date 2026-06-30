"""
Tests for subscription sign-in actions (Unit: auth/login.py).

Focus: importing an existing Codex CLI login into the store, and the educational
error when none exists. The Codex auth.json path is injected (no real ~/.codex).
"""

import base64
import json

import pytest

from rlmy.auth.login import CHATGPT_PROVIDER, login_with_codex
from rlmy.auth.store import AuthStore


def _codex_file(tmp_path):
    def seg(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    id_token = f"{seg({'alg': 'RS256'})}.{seg({'exp': 2000000000, 'https://api.openai.com/auth': {'chatgpt_account_id': 'acct', 'chatgpt_plan_type': 'plus'}})}.sig"
    path = tmp_path / "codex.json"
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "id_token": id_token,
            "access_token": "access",
            "refresh_token": "refresh",
            "account_id": "acct",
        },
    }))
    return path


class TestLoginWithCodex:
    def test_imports_codex_login_into_store(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        token = login_with_codex(store, codex_path=_codex_file(tmp_path))
        assert token.access_token == "access"
        assert store.get(CHATGPT_PROVIDER).account_id == "acct"

    def test_missing_codex_login_raises_with_instructions(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        with pytest.raises(RuntimeError, match="codex login"):
            login_with_codex(store, codex_path=tmp_path / "nope.json")
