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
from litellm.types.llms.openai import ResponsesAPIStreamEvents

from rlmy.auth.lm import build_lm
from rlmy.auth.store import AuthStore, OAuthToken

FUTURE = time.time() + 10_000


class _Resp:
    """Stand-in for a unary ResponsesAPIResponse (has .output)."""
    output = []


class _Event:
    def __init__(self, type_, response=None, item=None, delta=None):
        self.type = type_
        self.response = response
        self.item = item
        self.delta = delta


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
        seen["store"] = self.kwargs.get("store")
        seen["stream"] = self.kwargs.get("stream")
        return _Resp()

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    lm = build_lm("chatgpt-oauth/gpt-5.5", store=_signed_in(tmp_path))
    lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert seen["api_key"] == "tok-1"
    assert seen["model"] == "openai/gpt-5.5"
    assert seen["headers"]["originator"] == "codex_cli_rs"
    assert seen["headers"]["OpenAI-Beta"] == "responses=experimental"
    assert seen["headers"]["ChatGPT-Account-ID"] == "acct-9"
    assert seen["store"] is False
    assert seen["stream"] is True


def test_call_aggregates_streamed_response_to_completed(tmp_path, monkeypatch):
    final = _Resp()
    stream = [
        _Event("response.output_text.delta"),
        _Event(ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=final),
    ]
    monkeypatch.setattr(dspy.LM, "forward", lambda self, **kw: stream)
    lm = build_lm("chatgpt-oauth/gpt-5.5", store=_signed_in(tmp_path))

    assert lm.forward(messages=[{"role": "user", "content": "hi"}]) is final


def test_call_reconstructs_text_from_item_done_dict_content(tmp_path, monkeypatch):
    # Codex's completed event has empty output; items arrive via OUTPUT_ITEM_DONE
    # with dict content. We extract the text and expose it with a real .text attr.
    completed = _Resp()
    item = {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}
    stream = [
        _Event(ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, item=item),
        _Event(ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=completed),
    ]
    monkeypatch.setattr(dspy.LM, "forward", lambda self, **kw: stream)
    lm = build_lm("chatgpt-oauth/gpt-5.5", store=_signed_in(tmp_path))

    result = lm.forward(messages=[{"role": "user", "content": "hi"}])
    assert result.output[0].type == "message"
    assert result.output[0].content[0].text == "OK"


def test_call_synthesizes_message_from_text_deltas_as_fallback(tmp_path, monkeypatch):
    completed = _Resp()
    stream = [
        _Event(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, delta="Hel"),
        _Event(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, delta="lo"),
        _Event(ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=completed),
    ]
    monkeypatch.setattr(dspy.LM, "forward", lambda self, **kw: stream)
    lm = build_lm("chatgpt-oauth/gpt-5.5", store=_signed_in(tmp_path))

    result = lm.forward(messages=[{"role": "user", "content": "hi"}])
    msg = result.output[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello"


def test_call_refreshes_an_expired_bearer_first(tmp_path, monkeypatch):
    monkeypatch.setattr(dspy.LM, "forward", lambda self, **kw: _Resp())
    store = _signed_in(tmp_path, access_token="stale", expires_at=1.0)
    fresh = OAuthToken(access_token="refreshed", refresh_token="r2",
                       expires_at=FUTURE, account_id="acct-9")
    lm = build_lm("chatgpt-oauth/gpt-5.5", store=store, refresher=lambda t: fresh)
    lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert store.get("chatgpt-oauth").access_token == "refreshed"
