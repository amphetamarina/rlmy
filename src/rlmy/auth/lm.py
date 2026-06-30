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

from rlmy.auth.openai_codex import BACKEND_BASE, build_codex_headers
from rlmy.auth.openai_codex import refresh as codex_refresh
from rlmy.auth.session import Refresher, ensure_fresh_token
from rlmy.auth.store import AuthStore, OAuthToken

CHATGPT_OAUTH_PREFIX = "chatgpt-oauth/"

HeaderBuilder = Callable[[OAuthToken], dict]


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
        self.kwargs["api_key"] = token.access_token
        self.kwargs["headers"] = self._header_builder(token)

    def forward(self, prompt=None, messages=None, **kwargs):
        self._apply_credentials()
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self._apply_credentials()
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)


def _make_codex_oauth_lm(model: str, *, store=None, refresher=None, **kwargs) -> OAuthLM:
    return OAuthLM(
        model=f"openai/{model}",
        api_base=BACKEND_BASE,
        provider="chatgpt-oauth",
        store=store or AuthStore(),
        refresher=refresher or codex_refresh,
        header_builder=build_codex_headers,
        **kwargs,
    )


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
    return dspy.LM(model=model_string, **kwargs)
