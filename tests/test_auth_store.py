"""
Tests for the OAuth credential store (Unit: auth/store.py).

Focus: round-trip persistence, per-provider isolation, 0600 file perms, and the
pure expiry decision (needs_refresh). No network and no real provider tokens —
the store is pure file + time logic, so a tmp_path and an injected clock cover it.
"""

import json
import stat

from rlmy.auth.store import AuthStore, OAuthToken


def _token(**kw):
    base = dict(access_token="a", refresh_token="r", expires_at=1000.0)
    base.update(kw)
    return OAuthToken(**base)


class TestNeedsRefresh:
    def test_true_when_past_expiry(self):
        assert _token(expires_at=1000.0).needs_refresh(now=2000.0) is True

    def test_false_when_fresh(self):
        assert _token(expires_at=10000.0).needs_refresh(now=1000.0) is False

    def test_true_within_default_skew_window(self):
        # Expires in 60s; default skew is 300s, so it already needs refreshing.
        assert _token(expires_at=1060.0).needs_refresh(now=1000.0) is True

    def test_skew_is_configurable(self):
        assert _token(expires_at=1060.0).needs_refresh(now=1000.0, skew=30.0) is False


class TestRoundTrip:
    def test_set_then_get_returns_equal_token(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        tok = _token(account_id="acct-123", plan_type="plus")
        store.set("chatgpt-oauth", tok)
        assert store.get("chatgpt-oauth") == tok

    def test_get_unknown_provider_returns_none(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        assert store.get("nope") is None

    def test_get_on_missing_file_returns_none(self, tmp_path):
        store = AuthStore(path=tmp_path / "missing" / "auth.json")
        assert store.get("chatgpt-oauth") is None

    def test_providers_are_isolated(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        store.set("chatgpt-oauth", _token(access_token="x"))
        store.set("xai-oauth", _token(access_token="y"))
        assert store.get("chatgpt-oauth").access_token == "x"
        assert store.get("xai-oauth").access_token == "y"
        assert sorted(store.providers()) == ["chatgpt-oauth", "xai-oauth"]

    def test_entry_missing_required_field_reads_as_none(self, tmp_path):
        # A partially-corrupt entry must not crash get(); it reads as absent.
        path = tmp_path / "auth.json"
        path.write_text(json.dumps({"chatgpt-oauth": {"refresh_token": "r"}}))
        assert AuthStore(path=path).get("chatgpt-oauth") is None

    def test_unknown_fields_in_file_are_ignored(self, tmp_path):
        # Forward-compat: a newer schema with extra keys must not break loading.
        path = tmp_path / "auth.json"
        store = AuthStore(path=path)
        store.set("chatgpt-oauth", _token())
        raw = path.read_text().replace('"plan_type": null', '"future_field": 1')
        path.write_text(raw)
        assert store.get("chatgpt-oauth").access_token == "a"


class TestRemove:
    def test_remove_deletes_entry(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        store.set("chatgpt-oauth", _token())
        store.remove("chatgpt-oauth")
        assert store.get("chatgpt-oauth") is None

    def test_remove_unknown_is_noop(self, tmp_path):
        store = AuthStore(path=tmp_path / "auth.json")
        store.remove("nope")  # must not raise


class TestFilePermissions:
    def test_file_written_0600(self, tmp_path):
        path = tmp_path / "auth.json"
        AuthStore(path=path).set("chatgpt-oauth", _token())
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_parent_dir_created(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "auth.json"
        AuthStore(path=path).set("chatgpt-oauth", _token())
        assert path.exists()
