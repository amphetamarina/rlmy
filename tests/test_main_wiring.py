"""
Tests for the interpreter wiring helpers in main.py.

These cover the pure, injectable pieces of run_agent()'s interpreter handling —
the truthful session-recovery message and snapshot clearing — without standing
up the full agent loop (MCP, LM, etc.).
"""

from rlmy.agent.main import _clear_snapshot, _repl_recovery_note


class _FakeSnapshotting:
    """Stand-in for a snapshotting interpreter (e.g. Monty)."""
    restores_state = True


class _FakeNonSnapshotting:
    """Stand-in for the Deno builtin: no restores_state attribute at all."""


class TestRecoveryNote:
    def test_snapshotting_with_existing_file_says_restored(self, tmp_path):
        snap = tmp_path / "repl.snapshot"
        snap.write_bytes(b"x")  # file exists
        note = _repl_recovery_note(_FakeSnapshotting(), snap)
        assert "restored" in note.lower()

    def test_snapshotting_without_file_says_not_recovered(self, tmp_path):
        snap = tmp_path / "repl.snapshot"  # does not exist
        note = _repl_recovery_note(_FakeSnapshotting(), snap)
        assert "NOT recovered" in note

    def test_non_snapshotting_says_not_recovered_even_if_file_exists(self, tmp_path):
        # Deno-like: no restores_state attribute → getattr default False.
        snap = tmp_path / "repl.snapshot"
        snap.write_bytes(b"x")
        note = _repl_recovery_note(_FakeNonSnapshotting(), snap)
        assert "NOT recovered" in note


class TestClearSnapshot:
    def test_deletes_existing_file(self, tmp_path):
        snap = tmp_path / "repl.snapshot"
        snap.write_bytes(b"x")
        _clear_snapshot(snap)
        assert not snap.exists()

    def test_missing_file_is_noop(self, tmp_path):
        snap = tmp_path / "repl.snapshot"  # never created
        _clear_snapshot(snap)  # must not raise
        assert not snap.exists()

    def test_none_is_noop(self):
        _clear_snapshot(None)  # must not raise
