"""
Purpose: Provider-agnostic OAuth token freshness for subscription sign-in.
Usage: token = ensure_fresh_token(store, "chatgpt-oauth", refresher=refresh, now=time.time())
Key Components: ensure_fresh_token (refresh-if-expired, persist, return)
Conventions: Pure orchestration — the store, the refresher, and the clock are all
             injected, so this has no network or provider knowledge of its own.
"""

from __future__ import annotations

from typing import Callable

from rlmy.auth.store import AuthStore, OAuthToken

Refresher = Callable[[OAuthToken], OAuthToken]


def ensure_fresh_token(
    store: AuthStore, provider: str, refresher: Refresher, now: float
) -> OAuthToken:
    """
    Purpose: Return a usable (non-expiring-soon) token for a provider.
    Usage Patterns: Call immediately before using the bearer. If the stored token
        is within the refresh skew of expiry, `refresher` is invoked and the new
        token is persisted before returning. Raises RuntimeError when the provider
        has no stored credentials (user has not signed in).
    """
    token = store.get(provider)
    if token is None:
        raise RuntimeError(
            f"Not signed in to '{provider}'. "
            f"Run `rlmy auth login {provider.removesuffix('-oauth')}` to sign in."
        )
    if token.needs_refresh(now=now):
        token = refresher(token)
        store.set(provider, token)
    return token
