"""
Purpose: Sign in with ChatGPT (subscription) for the Codex Responses backend —
         import existing Codex CLI credentials and refresh OAuth access tokens.
Usage: tok = import_codex_cli_auth(); fresh = refresh(tok)
Key Components: parse_codex_auth (pure), import_codex_cli_auth (reads
                ~/.codex/auth.json), refresh (OAuth token exchange, HTTP injected)
Conventions: Inference uses BARE model names (e.g. "gpt-5.5"); the ChatGPT-account
             Codex backend rejects provider-prefixed names. Endpoint + headers are
             applied at the LM layer, not here.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable

from rlmy.auth.store import OAuthToken

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
BACKEND_BASE = "https://chatgpt.com/backend-api/codex"
AUTH_CLAIM = "https://api.openai.com/auth"
CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"

PostJson = Callable[..., dict]


def _decode_jwt_payload(token: str) -> dict:
    segment = token.split(".")[1]
    segment += "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(segment))


def parse_codex_auth(data: dict) -> OAuthToken:
    """
    Purpose: Turn a Codex CLI auth.json payload into an OAuthToken.
    Usage Patterns: account_id comes from the tokens block, falling back to the
        id_token's chatgpt_account_id claim; expires_at and plan_type are read from
        the (unverified) id_token payload. Raises ValueError when no OAuth tokens
        are present (e.g. an API-key-mode auth.json).
    """
    tokens = data.get("tokens", data)
    access = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not access or not refresh_token:
        raise ValueError("auth.json has no OAuth tokens (is auth_mode 'chatgpt'?)")
    claims = _decode_jwt_payload(tokens["id_token"]) if tokens.get("id_token") else {}
    auth_claim = claims.get(AUTH_CLAIM, {})
    return OAuthToken(
        access_token=access,
        refresh_token=refresh_token,
        expires_at=float(claims.get("exp", 0)),
        account_id=tokens.get("account_id") or auth_claim.get("chatgpt_account_id"),
        plan_type=auth_claim.get("chatgpt_plan_type"),
    )


def build_codex_headers(token: OAuthToken) -> dict:
    """
    Purpose: Headers the ChatGPT-account Codex Responses backend requires.
    Usage Patterns: Applied per request alongside the bearer. originator marks the
        client as Codex; OpenAI-Beta opts into the Responses surface; the account
        header is sent only when known (scopes the call to the right subscription).
    """
    headers = {
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
    }
    if token.account_id:
        headers["ChatGPT-Account-ID"] = token.account_id
    return headers


def import_codex_cli_auth(path: Path = CODEX_AUTH_FILE) -> OAuthToken | None:
    if not path.exists():
        return None
    return parse_codex_auth(json.loads(path.read_text()))


def _urllib_post_json(url: str, body: dict, headers: dict | None = None) -> dict:
    import urllib.request

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def refresh(token: OAuthToken, post: PostJson = _urllib_post_json) -> OAuthToken:
    """
    Purpose: Exchange a refresh token for a fresh access token via OpenAI OAuth.
    Usage Patterns: post is injected (defaults to a urllib call) so the exchange is
        unit-testable offline. Reuses the prior refresh_token when the response
        omits a new one; preserves account_id from the existing token.
    """
    resp = post(TOKEN_URL, {
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": token.refresh_token,
        "scope": "openid profile email",
    })
    claims = _decode_jwt_payload(resp["id_token"]) if resp.get("id_token") else {}
    return OAuthToken(
        access_token=resp["access_token"],
        refresh_token=resp.get("refresh_token") or token.refresh_token,
        expires_at=float(claims.get("exp", 0)),
        account_id=token.account_id,
        plan_type=token.plan_type or claims.get(AUTH_CLAIM, {}).get("chatgpt_plan_type"),
    )
