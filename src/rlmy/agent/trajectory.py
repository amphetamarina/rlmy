"""
Purpose: Trajectory persistence and compaction — all trajectory operations in one place.
Usage:
    from rlmy.agent.trajectory import save_trajectory, load_trajectory, compact_trajectory
    save_trajectory(traj, path)
    traj = load_trajectory(path)
    compacted = compact_trajectory(traj, budget=80_000)
Key Components:
    save_trajectory / load_trajectory / clear_trajectory — disk persistence (atomic writes)
    compact_trajectory — 3-tier sliding window compaction algorithm
    estimate_tokens — heuristic token counter (len // 4)
    trajectory_stats — summary stats for display
Conventions:
    Moved from cli_proto.py to avoid responsibility overload.
    All functions are pure (no module-level state) except the default path constant.
    Compaction is destructive and irreversible — empties fields, never restores them.
    Format: JSONL (one JSON object per line) — easy to chop/tail/grep manually.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Persistence
# =============================================================================

# Default trajectory file for standalone cli_proto usage.
# mainmcp.py passes a workspace-specific path instead.
DEFAULT_TRAJECTORY_FILE = Path(__file__).parent / ".trajectory_state.jsonl"


def save_trajectory(trajectory: list[dict], path: Path | None = None):
    """
    Purpose: Persist trajectory to disk for crash recovery.
    Args:
        trajectory: List of trajectory dicts to persist.
        path: File path to write to. Defaults to cli_proto's own .trajectory_state.jsonl.
              mainmcp.py passes workspace/trajectory.jsonl for per-workspace persistence.
    Conventions:
        Called after every trajectory mutation. Atomic write via temp file.
        Format: JSONL — one JSON object per line. Easy to chop/tail/grep manually.
    """
    target = path or DEFAULT_TRAJECTORY_FILE
    tmp = target.with_suffix(".tmp")
    lines = [json.dumps(entry, ensure_ascii=False) for entry in trajectory]
    tmp.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    tmp.rename(target)  # atomic on POSIX


def load_trajectory(path: Path | None = None) -> list[dict]:
    """
    Purpose: Recover trajectory from disk on startup.
    Args:
        path: File path to read from. Defaults to cli_proto's own .trajectory_state.jsonl.
    Returns: List of trajectory dicts, or empty list if no saved state.
    Conventions:
        Reads JSONL (one JSON object per line). Skips blank lines gracefully.
        Also handles legacy JSON array format (single json.loads) for migration.
    """
    target = path or DEFAULT_TRAJECTORY_FILE
    # Auto-migrate: if .jsonl doesn't exist but legacy .json sibling does, rename it
    if not target.exists() and target.suffix == ".jsonl":
        legacy = target.with_suffix(".json")
        if legacy.exists():
            legacy.rename(target)
            logger.info(f"Migrated legacy trajectory: {legacy.name} → {target.name}")
    if target.exists():
        try:
            text = target.read_text(encoding="utf-8")
            if not text.strip():
                return []
            # Try JSONL first (one object per line).
            # Guard: if a line parses to a list instead of a dict, the file is a
            # compact single-line legacy JSON array (e.g. "[{...},{...}]") that was
            # renamed to .jsonl without content conversion. json.loads() succeeds
            # on that line and returns a list — NOT a dict — so JSONDecodeError is
            # never raised and the except-fallback below is never reached.
            # We handle it here explicitly to avoid returning list[list] to Pydantic.
            entries: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    parsed = json.loads(line)
                    if isinstance(parsed, list):
                        # Legacy JSON array on one line — the whole file is the trajectory
                        logger.info("Migrated legacy JSON array trajectory to JSONL format.")
                        return parsed
                    entries.append(parsed)
            return entries
        except (json.JSONDecodeError, OSError) as e:
            # Fallback: try legacy multi-line JSON array format for migration
            # (handles the case where the JSON array was pretty-printed across many lines,
            # so the per-line parse above would fail on the first line "[{..." with a
            # JSONDecodeError before we ever reach the isinstance check)
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    logger.info("Migrated legacy JSON array trajectory to JSONL format.")
                    return data
            except Exception:
                pass
            logger.warning(f"Could not load trajectory state: {e}")
    return []


def clear_trajectory(path: Path | None = None):
    """Remove persisted trajectory (e.g., on explicit reset). Also removes legacy .json if present."""
    target = path or DEFAULT_TRAJECTORY_FILE
    target.unlink(missing_ok=True)
    # Clean up legacy .json file (from before JSONL migration) if it exists
    legacy = target.with_suffix(".json")
    if legacy != target:
        legacy.unlink(missing_ok=True)


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Heuristic token count: ~4 chars per token for English/code mixed content.
    Not precise, but consistent and fast. Used for budget decisions, not billing.
    """
    return len(text) // 4


def _entry_text(entry: dict) -> str:
    """Serialize a trajectory entry to the text the LLM actually sees."""
    parts = []
    r = entry.get("reasoning", "")
    c = entry.get("code", "")
    o = entry.get("output", "")
    if r:
        parts.append(r)
    if c:
        parts.append(c)
    if o:
        parts.append(o)
    return "\n".join(parts)


def _entry_tokens(entry: dict) -> int:
    """Estimate tokens for a single trajectory entry."""
    return estimate_tokens(_entry_text(entry))


# =============================================================================
# Compaction
# =============================================================================

# Default budget constants (tokens). All configurable via function args.
# Total = tier1 + tier2 + tier3. Aggressive target keeps LLM attention sharp.
DEFAULT_TOTAL_BUDGET = 40_000
DEFAULT_TIER1_BUDGET = 20_000   # newest: full (reasoning + code + output)
DEFAULT_TIER2_BUDGET = 10_000   # middle: reasoning + code (output stripped)
DEFAULT_TIER3_BUDGET = 10_000   # oldest: reasoning only (code + output stripped)

# User-input entries are identified by this tag in reasoning.
_USER_INPUT_TAG = "<user-input>"


def _is_user_input_entry(entry: dict) -> bool:
    """Check if entry is a user-input marker (exempt from compaction)."""
    return entry.get("reasoning", "").lstrip().startswith(_USER_INPUT_TAG)


def trajectory_stats(trajectory: list[dict]) -> dict:
    """
    Purpose: Compute summary statistics for a trajectory.

    Returns:
        Dict with keys: entries_total, tokens_total, user_input_entries,
        compactable_entries.
    """
    total_tokens = 0
    user_input_count = 0
    for entry in trajectory:
        total_tokens += _entry_tokens(entry)
        if _is_user_input_entry(entry):
            user_input_count += 1
    return {
        "entries_total": len(trajectory),
        "tokens_total": total_tokens,
        "user_input_entries": user_input_count,
        "compactable_entries": len(trajectory) - user_input_count,
    }


def compact_trajectory(
    trajectory: list[dict],
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    tier1_budget: int = DEFAULT_TIER1_BUDGET,
    tier2_budget: int = DEFAULT_TIER2_BUDGET,
    tier3_budget: int = DEFAULT_TIER3_BUDGET,
    dry_run: bool = False,
) -> tuple[list[dict], dict] | dict:
    """
    Purpose: Degrade older trajectory entries to fit within a token budget.

    Algorithm — 3-tier sliding window, walk oldest→newest:
    1. Estimate total. If ≤ budget → return unchanged.
    2. Degrade oldest compactable entries to Tier 3 (reasoning only) until under budget or Tier 3 full.
    3. If still over: degrade next batch to Tier 2 (reasoning + code) until under budget or Tier 2 full.
    4. If still over and both tiers full: drop oldest Tier 3 entries entirely.
    5. Return compacted trajectory.

    User-input entries (identified by <user-input> tag) are exempt from all degradation.
    Cleaning = set fields to "". Idempotent: degrading "" → "" is a no-op.

    Args:
        trajectory: List of trajectory dicts. Modified in-place (unless dry_run).
        total_budget: Target token budget for the entire trajectory.
        tier1_budget: Max tokens for Tier 1 (full fidelity, newest entries).
        tier2_budget: Max tokens for Tier 2 (code + reasoning, middle entries).
        tier3_budget: Max tokens for Tier 3 (reasoning only, oldest entries).
        dry_run: If True, return stats dict without modifying trajectory.

    Returns:
        If dry_run=False: tuple (trajectory, stats_dict). The trajectory is the same
            reference, modified in-place. stats_dict has action_needed, tokens_before/after, etc.
        If dry_run=True: stats dict only.
    """
    if not trajectory:
        no_op = {"action_needed": False, "reason": "empty"}
        return no_op if dry_run else (trajectory, no_op)

    # --- Estimate current total ---
    tokens_before = sum(_entry_tokens(e) for e in trajectory)
    if tokens_before <= total_budget:
        no_op = {
            "action_needed": False,
            "reason": "within_budget",
            "tokens_before": tokens_before,
            "tokens_after": tokens_before,
            "entries_total": len(trajectory),
        }
        if dry_run:
            return no_op
        return trajectory, no_op

    # --- Build index of compactable entries (oldest first) ---
    # Each item: (index_in_trajectory, current_tier)
    # Tier assignment: 1 = full, 2 = code+reasoning (no output), 3 = reasoning only
    compactable_indices: list[int] = []
    for i, entry in enumerate(trajectory):
        if not _is_user_input_entry(entry):
            compactable_indices.append(i)

    if not compactable_indices:
        # All entries are user-input → nothing to compact
        no_op = {
            "action_needed": False,
            "reason": "all_exempt",
            "tokens_before": tokens_before,
            "tokens_after": tokens_before,
            "entries_total": len(trajectory),
        }
        if dry_run:
            return no_op
        return trajectory, no_op

    def _current_tier(entry: dict) -> int:
        """Determine current degradation tier of an entry."""
        has_code = bool(entry.get("code", ""))
        has_output = bool(entry.get("output", ""))
        if has_code and has_output:
            return 1  # full
        if has_code:
            return 2  # code + reasoning
        return 3      # reasoning only (or already empty)

    # --- Phase 1: Degrade to Tier 3 (reasoning only) from oldest ---
    tier3_tokens_used = 0
    tier3_count = 0
    current_total = tokens_before

    for idx in compactable_indices:
        if current_total <= total_budget:
            break
        if tier3_tokens_used >= tier3_budget:
            break

        entry = trajectory[idx]
        tier = _current_tier(entry)

        if tier <= 2:
            # Tier 1 or 2 — degrade to tier 3 (reasoning only)
            # Calculate savings from stripping code + output
            code_tokens = estimate_tokens(entry.get("code", ""))
            output_tokens = estimate_tokens(entry.get("output", ""))
            savings = code_tokens + output_tokens

            reasoning_tokens = estimate_tokens(entry.get("reasoning", ""))
            if tier3_tokens_used + reasoning_tokens > tier3_budget:
                break  # Tier 3 budget would overflow

            if not dry_run:
                entry["code"] = ""
                entry["output"] = ""

            current_total -= savings
            tier3_tokens_used += reasoning_tokens
            tier3_count += 1
        elif tier == 3:
            # Already tier 3 — just account for its budget usage
            reasoning_tokens = estimate_tokens(entry.get("reasoning", ""))
            if tier3_tokens_used + reasoning_tokens > tier3_budget:
                break
            tier3_tokens_used += reasoning_tokens
            tier3_count += 1

    # --- Phase 2: Degrade to Tier 2 (strip output) from next entries ---
    tier2_tokens_used = 0
    tier2_count = 0

    # Find the first compactable entry that isn't already in tier 3
    phase2_start = tier3_count  # index into compactable_indices

    for ci in range(phase2_start, len(compactable_indices)):
        if current_total <= total_budget:
            break
        if tier2_tokens_used >= tier2_budget:
            break

        idx = compactable_indices[ci]
        entry = trajectory[idx]
        tier = _current_tier(entry)

        if tier == 1:
            # Full entry → degrade to tier 2 (strip output only)
            output_tokens = estimate_tokens(entry.get("output", ""))
            savings = output_tokens

            reasoning_tokens = estimate_tokens(entry.get("reasoning", ""))
            code_tokens = estimate_tokens(entry.get("code", ""))
            entry_tier2_size = reasoning_tokens + code_tokens

            if tier2_tokens_used + entry_tier2_size > tier2_budget:
                break  # Tier 2 budget would overflow

            if not dry_run:
                entry["output"] = ""

            current_total -= savings
            tier2_tokens_used += entry_tier2_size
            tier2_count += 1
        elif tier == 2:
            # Already tier 2 — just account for budget
            reasoning_tokens = estimate_tokens(entry.get("reasoning", ""))
            code_tokens = estimate_tokens(entry.get("code", ""))
            entry_tier2_size = reasoning_tokens + code_tokens

            if tier2_tokens_used + entry_tier2_size > tier2_budget:
                break
            tier2_tokens_used += entry_tier2_size
            tier2_count += 1

    # --- Phase 3: If still over budget, drop oldest Tier 3 entries entirely ---
    dropped_count = 0
    if current_total > total_budget and not dry_run:
        # Walk from oldest, drop tier 3 entries (they only have reasoning)
        indices_to_drop = []
        for ci in range(tier3_count):
            if current_total <= total_budget:
                break
            idx = compactable_indices[ci]
            entry = trajectory[idx]
            entry_tokens = _entry_tokens(entry)
            current_total -= entry_tokens
            indices_to_drop.append(idx)
            dropped_count += 1

        # Remove dropped entries (reverse order to preserve indices)
        for idx in sorted(indices_to_drop, reverse=True):
            trajectory.pop(idx)
    elif current_total > total_budget and dry_run:
        # Estimate how many would be dropped
        for ci in range(tier3_count):
            if current_total <= total_budget:
                break
            idx = compactable_indices[ci]
            entry = trajectory[idx]
            reasoning_tokens = estimate_tokens(entry.get("reasoning", ""))
            current_total -= reasoning_tokens
            dropped_count += 1

    # --- Compute Tier 1 count ---
    # Tier 1 = all compactable entries not in tier 2 or tier 3
    remaining_compactable = len(compactable_indices) - tier3_count - tier2_count - dropped_count
    tier1_count = max(0, remaining_compactable)
    user_input_count = len(trajectory) - len(compactable_indices) + dropped_count

    tokens_after = current_total

    stats = {
        "action_needed": True,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "entries_total": len(trajectory) if not dry_run else len(trajectory) - dropped_count,
        "tier1_entries": tier1_count,
        "tier2_entries": tier2_count,
        "tier3_entries": tier3_count - dropped_count,
        "dropped_entries": dropped_count,
        "user_input_entries": user_input_count if not dry_run else sum(
            1 for e in trajectory if _is_user_input_entry(e)
        ),
    }

    if dry_run:
        return stats

    return trajectory, stats


def format_compact_stats(stats: dict) -> str:
    """
    Purpose: Format compaction stats into a human-readable string for Rich panel display.

    Args:
        stats: Dict returned by compact_trajectory (either dry_run or actual).

    Returns:
        Formatted string suitable for Rich Panel body.
    """
    if not stats.get("action_needed", False):
        reason = stats.get("reason", "unknown")
        if reason == "within_budget":
            tok = stats.get("tokens_before", 0)
            return f"Already within budget (~{tok:,} tokens). Nothing to compact."
        if reason == "empty":
            return "Trajectory is empty. Nothing to compact."
        if reason == "all_exempt":
            return "All entries are user-input markers (exempt from compaction). Nothing to compact."
        return "Nothing to compact."

    before = stats.get("tokens_before", 0)
    after = stats.get("tokens_after", 0)
    lines = [
        f"Before: {stats['entries_total'] + stats['dropped_entries']} entries, ~{before:,} tokens",
        f"After:  {stats['entries_total']} entries, ~{after:,} tokens",
        "",
        f"  Tier 1 (full):            {stats['tier1_entries']} entries",
        f"  Tier 2 (code only):       {stats['tier2_entries']} entries",
        f"  Tier 3 (reasoning only):  {stats['tier3_entries']} entries",
        f"  Dropped:                  {stats['dropped_entries']} entries",
        f"  User-input (exempt):      {stats['user_input_entries']} entries",
        "",
        "⚠️  This is irreversible.",
    ]
    return "\n".join(lines)


def format_compact_stats_for_llm(stats: dict) -> str:
    """
    Purpose: Format compaction stats as a concise message for the LLM to see as tool output.

    Args:
        stats: Dict returned by compact_trajectory.

    Returns:
        One-line summary suitable for returning from contextual_ask_user_guidance.
    """
    if not stats.get("action_needed", False):
        return "[SYSTEM] Trajectory is within budget. No compaction needed."

    before = stats.get("tokens_before", 0)
    after = stats.get("tokens_after", 0)
    t1 = stats.get("tier1_entries", 0)
    t2 = stats.get("tier2_entries", 0)
    t3 = stats.get("tier3_entries", 0)
    dropped = stats.get("dropped_entries", 0)
    return (
        f"[SYSTEM] Trajectory compacted: ~{before:,}→~{after:,} tokens. "
        f"{t1} full, {t2} code-only, {t3} reasoning-only, {dropped} dropped. "
        f"Your recent context is fully intact. Continue with your current task."
    )
