"""
Tests for the Sign-in-with-ChatGPT auth helpers (Unit: auth/openai_codex.py).

Focus: parsing an existing ~/.codex/auth.json into an OAuthToken, and the OAuth
refresh exchange. HTTP is injected (post=), so no network; id_tokens are
hand-built JWT fixtures (signature ignored, only the payload is read).
"""

import base64
import json

import pytest

from rlmy.auth.openai_codex import (
    CLIENT_ID,
    TOKEN_URL,
    parse_codex_auth,
    refresh,
)
from rlmy.auth.store import OAuthToken


def _jwt(payload: dict) -> str:
    def seg(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


def _codex_auth(exp=2000000000, account_id="acct-uuid", plan="plus"):
    id_token = _jwt({
        "exp": exp,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan,
        },
    })
    return {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": id_token,
            "access_token": "access-abc",
            "refresh_token": "refresh-xyz",
            "account_id": account_id,
        },
        "last_refresh": "2026-06-06T22:06:00Z",
    }


class TestParseCodexAuth:
    def test_extracts_tokens_and_expiry(self):
        tok = parse_codex_auth(_codex_auth(exp=2000000000))
        assert tok.access_token == "access-abc"
        assert tok.refresh_token == "refresh-xyz"
        assert tok.account_id == "acct-uuid"
        assert tok.expires_at == 2000000000

    def test_extracts_plan_type_from_id_token(self):
        assert parse_codex_auth(_codex_auth(plan="pro")).plan_type == "pro"

    def test_account_id_falls_back_to_id_token_claim(self):
        data = _codex_auth(account_id="from-claim")
        data["tokens"].pop("account_id")
        assert parse_codex_auth(data).account_id == "from-claim"

    def test_expiry_prefers_access_token_when_it_is_a_jwt(self):
        # The backend validates the access token, so its exp wins over the
        # id_token's when the access token is itself a JWT.
        data = _codex_auth(exp=2000000000)
        data["tokens"]["access_token"] = _jwt({"exp": 1900000000})
        assert parse_codex_auth(data).expires_at == 1900000000

    def test_expiry_falls_back_to_id_token_for_opaque_access_token(self):
        data = _codex_auth(exp=2000000000)
        data["tokens"]["access_token"] = "opaque-not-a-jwt"
        assert parse_codex_auth(data).expires_at == 2000000000

    def test_missing_oauth_tokens_raises(self):
        data = _codex_auth()
        data["tokens"].pop("refresh_token")
        with pytest.raises(ValueError):
            parse_codex_auth(data)


class TestRefresh:
    def test_sends_correct_exchange_and_returns_new_token(self):
        captured = {}

        def fake_post(url, body, headers=None):
            captured["url"] = url
            captured["body"] = body
            return {
                "id_token": _jwt({"exp": 2111111111}),
                "access_token": "new-access",
                "refresh_token": "new-refresh",
            }

        old = OAuthToken(access_token="old", refresh_token="refresh-xyz",
                         expires_at=0, account_id="acct", plan_type="plus")
        new = refresh(old, post=fake_post)

        assert captured["url"] == TOKEN_URL
        assert captured["body"]["client_id"] == CLIENT_ID
        assert captured["body"]["grant_type"] == "refresh_token"
        assert captured["body"]["refresh_token"] == "refresh-xyz"
        assert new.access_token == "new-access"
        assert new.refresh_token == "new-refresh"
        assert new.expires_at == 2111111111
        assert new.account_id == "acct"

    def test_reuses_old_refresh_token_when_response_omits_it(self):
        def fake_post(url, body, headers=None):
            return {"id_token": _jwt({"exp": 2111111111}), "access_token": "new-access"}

        old = OAuthToken(access_token="old", refresh_token="keep-me", expires_at=0)
        assert refresh(old, post=fake_post).refresh_token == "keep-me"

    def test_rejected_refresh_raises_educational_reauth_error(self):
        import urllib.error

        def fake_post(url, body, headers=None):
            raise urllib.error.HTTPError(url, 400, "invalid_grant", {}, None)

        old = OAuthToken(access_token="old", refresh_token="dead", expires_at=0)
        with pytest.raises(RuntimeError, match="sign in"):
            refresh(old, post=fake_post)
