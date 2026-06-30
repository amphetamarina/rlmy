"""
Functional test for the Sign-in-with-ChatGPT path: config string → LM →
outgoing request, with the DSPy call boundary mocked.

Covers the critical path in one flow: build_lm routes a chatgpt-oauth model to
an OAuthLM that, on each call, refreshes an expired bearer and stamps the live
token + Codex headers onto the request DSPy is about to send. We mock at the
boundary (dspy.LM.forward) rather than inside our own code.
"""

import time

import dspy

from rlmy.auth.lm import build_lm
from rlmy.auth.store import AuthStore, OAuthToken

FUTURE = time.time() + 10_000


def _signed_in(tmp_path, **over):
    store = AuthStore(path=tmp_path / "auth.json")
    base = dict(access_token="tok-1", refresh_token="r", expires_at=FUTURE,
                account_id="acct-9", plan_type="plus")
    base.update(over)
    store.set("chatgpt-oauth", OAuthToken(**base))
    return store


def test_call_stamps_fresh_bearer_and_codex_headers(tmp_path, monkeypatch):
    seen = {}

    def fake_forward(self, prompt=None, messages=None, **kwargs):
        seen["api_key"] = self.kwargs.get("api_key")
        seen["headers"] = dict(self.kwargs.get("headers") or {})
        seen["model"] = self.model
        return "ok"

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    lm = build_lm("chatgpt-oauth/gpt-5.5", cache=False, store=_signed_in(tmp_path))
    lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert seen["api_key"] == "tok-1"
    assert seen["model"] == "openai/gpt-5.5"
    assert seen["headers"]["originator"] == "codex_cli_rs"
    assert seen["headers"]["OpenAI-Beta"] == "responses=experimental"
    assert seen["headers"]["ChatGPT-Account-ID"] == "acct-9"


def test_call_refreshes_an_expired_bearer_first(tmp_path, monkeypatch):
    monkeypatch.setattr(dspy.LM, "forward", lambda self, **kw: None)
    store = _signed_in(tmp_path, access_token="stale", expires_at=1.0)
    fresh = OAuthToken(access_token="refreshed", refresh_token="r2",
                       expires_at=FUTURE, account_id="acct-9")
    lm = build_lm("chatgpt-oauth/gpt-5.5", cache=False, store=store,
                  refresher=lambda t: fresh)
    lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert store.get("chatgpt-oauth").access_token == "refreshed"
