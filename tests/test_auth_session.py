"""
Tests for the provider-agnostic token freshness helper (auth/session.py).

Focus: ensure_fresh_token's three paths — not signed in (raise), fresh (return
as-is, no refresh), expired (refresh + persist). The store is real (tmp_path);
the refresher and clock are injected, so there is no network.
"""

import pytest

from rlmy.auth.session import ensure_fresh_token
from rlmy.auth.store import AuthStore, OAuthToken


def _store(tmp_path, **tok):
    store = AuthStore(path=tmp_path / "auth.json")
    if tok:
        store.set("chatgpt-oauth", OAuthToken(**tok))
    return store


class TestEnsureFreshToken:
    def test_raises_when_not_signed_in(self, tmp_path):
        with pytest.raises(RuntimeError, match="chatgpt-oauth"):
            ensure_fresh_token(_store(tmp_path), "chatgpt-oauth",
                               refresher=lambda t: t, now=1000.0)

    def test_returns_existing_when_fresh(self, tmp_path):
        store = _store(tmp_path, access_token="a", refresh_token="r", expires_at=10000.0)
        calls = []
        token = ensure_fresh_token(store, "chatgpt-oauth",
                                   refresher=lambda t: calls.append(t) or t, now=1000.0)
        assert token.access_token == "a"
        assert calls == []

    def test_refreshes_and_persists_when_expired(self, tmp_path):
        store = _store(tmp_path, access_token="old", refresh_token="r", expires_at=1000.0)
        fresh = OAuthToken(access_token="new", refresh_token="r2", expires_at=99999.0)
        token = ensure_fresh_token(store, "chatgpt-oauth",
                                   refresher=lambda t: fresh, now=2000.0)
        assert token.access_token == "new"
        assert store.get("chatgpt-oauth").access_token == "new"
