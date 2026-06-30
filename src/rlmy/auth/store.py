"""
Purpose: Persistent OAuth credential store for subscription sign-in providers.
Usage: store = AuthStore(); store.set("chatgpt-oauth", token); store.get("chatgpt-oauth")
Key Components: OAuthToken (one provider's tokens + expiry), AuthStore (0600 JSON file)
Conventions: Stored at ~/.config/rlmy/auth.json (0600), one entry per provider id.
             Kept separate from config.toml so secrets never live in plain config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_REFRESH_SKEW = 300.0


@dataclass
class OAuthToken:
    """One provider's OAuth credentials. expires_at is epoch seconds."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str | None = None
    plan_type: str | None = None

    def needs_refresh(self, now: float, skew: float = DEFAULT_REFRESH_SKEW) -> bool:
        return now >= self.expires_at - skew

    @classmethod
    def from_dict(cls, data: dict) -> OAuthToken:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class AuthStore:
    """Read/write OAuth tokens keyed by provider id, in a 0600 JSON file."""

    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".config" / "rlmy" / "auth.json")

    def get(self, provider: str) -> OAuthToken | None:
        entry = self._load().get(provider)
        return OAuthToken.from_dict(entry) if entry else None

    def set(self, provider: str, token: OAuthToken) -> None:
        data = self._load()
        data[provider] = asdict(token)
        self._save(data)

    def remove(self, provider: str) -> None:
        data = self._load()
        if data.pop(provider, None) is not None:
            self._save(data)

    def providers(self) -> list[str]:
        return list(self._load().keys())

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
        os.chmod(self.path, 0o600)
