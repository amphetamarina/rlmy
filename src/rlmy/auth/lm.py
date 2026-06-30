"""
Purpose: Bridge subscription OAuth providers into a DSPy LM.
Usage: lm = build_lm("chatgpt-oauth/gpt-5.5")  # or any normal DSPy model string
Key Components: OAuthLM (refreshes the bearer before each call), build_lm (router)
Conventions: OAuthLM holds no request logic of its own beyond applying credentials;
             the request path stays DSPy's. The refresh decision lives in
             auth.session, and provider headers live in the provider module.
"""

from __future__ import annotations

import time
from typing import Callable

import dspy
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from rlmy.auth.openai_codex import BACKEND_BASE, build_codex_headers
from rlmy.auth.openai_codex import refresh as codex_refresh
from rlmy.auth.session import Refresher, ensure_fresh_token
from rlmy.auth.store import AuthStore, OAuthToken

CHATGPT_OAUTH_PREFIX = "chatgpt-oauth/"
_RESPONSE_COMPLETED = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
_OUTPUT_ITEM_DONE = ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
_OUTPUT_TEXT_DELTA = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA

HeaderBuilder = Callable[[OAuthToken], dict]


def _assistant_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_reassembled",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class _ResponseStreamReducer:
    """
    Purpose: Reassemble a Codex Responses stream into one response object.
    Usage Patterns: Feed each streamed event, then call finish(). The Codex
        backend's RESPONSE_COMPLETED event carries usage but an empty `output`;
        the generated items arrive as OUTPUT_ITEM_DONE events (with text deltas
        alongside). We reattach those items so DSPy's responses parser finds the
        text, falling back to a synthetic message built from the text deltas.
    """

    def __init__(self):
        self._completed = None
        self._items = []
        self._text_parts = []

    def feed(self, event) -> None:
        etype = getattr(event, "type", None)
        if etype == _RESPONSE_COMPLETED:
            self._completed = event.response
        elif etype == _OUTPUT_ITEM_DONE:
            item = getattr(event, "item", None)
            if item is not None:
                self._items.append(item)
        elif etype == _OUTPUT_TEXT_DELTA:
            delta = getattr(event, "delta", None)
            if delta:
                self._text_parts.append(delta)

    def finish(self):
        if self._completed is None:
            raise RuntimeError("Codex response stream ended without a completed event.")
        if not getattr(self._completed, "output", None):
            # Codex's completed event has empty output; rebuild the assistant text
            # as one real ResponseOutputMessage — proper .text attribute for DSPy's
            # parser, and a pydantic type so model_dump/inspect_history stay clean.
            # Raw litellm items can't be reused: their content is dicts, not objects.
            text = "".join(self._text_parts) or self._text_from_message_items()
            if not text:
                raise RuntimeError(
                    "Codex stream completed with no assistant text "
                    "(the model returned only reasoning or an empty response)."
                )
            self._completed.output = [_assistant_message(text)]
        return self._completed

    def _text_from_message_items(self) -> str:
        parts = []
        for item in self._items:
            itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if itype != "message":
                continue  # skip reasoning/tool items — only the message is the answer
            content = (
                item.get("content") if isinstance(item, dict)
                else getattr(item, "content", None)
            )
            for block in content or []:
                text = (
                    block.get("text") if isinstance(block, dict)
                    else getattr(block, "text", None)
                )
                if text:
                    parts.append(text)
        return "".join(parts)


def _collect_responses_stream(result):
    # Codex requires stream=true, so litellm returns a stream of events while
    # DSPy's parser expects one response object. Discriminator: litellm's
    # streaming iterator exposes .response/.completed_response but no .output,
    # whereas a unary ResponsesAPIResponse has .output — so .output means "done".
    if hasattr(result, "output"):
        return result
    reducer = _ResponseStreamReducer()
    for event in result:
        reducer.feed(event)
    return reducer.finish()


async def _acollect_responses_stream(result):
    if hasattr(result, "output"):
        return result
    reducer = _ResponseStreamReducer()
    async for event in result:
        reducer.feed(event)
    return reducer.finish()


class OAuthLM(dspy.LM):
    """
    Purpose: A DSPy LM whose bearer token is refreshed and re-applied on every call.
    Attributes: _provider (store key), _store (AuthStore), _refresher (token
        exchange), _header_builder (provider request headers), _time_fn (clock).
    Usage Patterns: Construct via build_lm, not directly. Each forward/aforward
        calls ensure_fresh_token, then stamps the live access token and provider
        headers onto self.kwargs before delegating to the standard DSPy path. The
        subscription backends speak the Responses API, so model_type is "responses".
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        provider: str,
        store: AuthStore,
        refresher: Refresher,
        header_builder: HeaderBuilder,
        time_fn: Callable[[], float] = time.time,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_type="responses",
            api_base=api_base,
            api_key="pending-oauth",
            **kwargs,
        )
        self._provider = provider
        self._store = store
        self._refresher = refresher
        self._header_builder = header_builder
        self._time_fn = time_fn

    def _apply_credentials(self) -> None:
        token = ensure_fresh_token(
            self._store, self._provider, self._refresher, self._time_fn()
        )
        # Mutates shared self.kwargs; safe under concurrent calls (e.g. sub_lm
        # batched queries) because every writer sets the same fresh token/headers.
        self.kwargs["api_key"] = token.access_token
        self.kwargs["headers"] = self._header_builder(token)

    def forward(self, prompt=None, messages=None, **kwargs):
        # Note: DSPy's global usage_tracker reads usage off the pre-reduced stream
        # (so it records {} here); lm.history usage is computed post-reduce and is
        # correct. Acceptable for a subscription (no per-token billing).
        self._apply_credentials()
        return _collect_responses_stream(
            super().forward(prompt=prompt, messages=messages, **kwargs)
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self._apply_credentials()
        return await _acollect_responses_stream(
            await super().aforward(prompt=prompt, messages=messages, **kwargs)
        )


CHATGPT_PROVIDER = CHATGPT_OAUTH_PREFIX.rstrip("/")


def _make_codex_oauth_lm(model: str, *, store=None, refresher=None, **kwargs) -> OAuthLM:
    # Streamed responses can't be coherently DSPy-cached; the backend also sets
    # store=false, so there's nothing to cache. Force it off.
    kwargs["cache"] = False
    lm = OAuthLM(
        # The "openai/" prefix only selects LiteLLM's OpenAI transport; LiteLLM
        # strips it before the wire, so the backend receives the bare name
        # (e.g. "gpt-5.5") it requires.
        model=f"openai/{model}",
        api_base=BACKEND_BASE,
        provider=CHATGPT_PROVIDER,
        store=store or AuthStore(),
        refresher=refresher or codex_refresh,
        header_builder=build_codex_headers,
        **kwargs,
    )
    # The ChatGPT-account Codex backend requires store=false ("Store must be set
    # to false") and stream=true ("Stream must be set to true"). DSPy passes
    # unknown kwargs straight through to litellm.responses, so these land in the
    # Responses request body; the stream is reassembled in _collect_responses_stream.
    lm.kwargs["store"] = False
    lm.kwargs["stream"] = True
    return lm


def build_lm(model_string: str, *, store=None, refresher=None, **kwargs):
    """
    Purpose: Turn a configured model string into the right LM instance.
    Usage Patterns: A "chatgpt-oauth/<model>" prefix yields an OAuthLM bound to the
        ChatGPT Codex Responses backend; any other string is a standard LiteLLM
        model and yields a plain dspy.LM. store/refresher are injection seams for
        tests and are ignored for the plain path.
    """
    if model_string.startswith(CHATGPT_OAUTH_PREFIX):
        model = model_string[len(CHATGPT_OAUTH_PREFIX):]
        return _make_codex_oauth_lm(model, store=store, refresher=refresher, **kwargs)
    if "-oauth/" in model_string:
        raise RuntimeError(
            f"Unknown subscription provider in {model_string!r}. "
            f"Supported: {CHATGPT_OAUTH_PREFIX}<model> (e.g. chatgpt-oauth/gpt-5.5)."
        )
    return dspy.LM(model=model_string, **kwargs)
