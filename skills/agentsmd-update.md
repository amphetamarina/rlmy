# Skill: Updating AGENTS.md

## Purpose

Keep AGENTS.md lean and relevant by adding only lessons that prevent recurring time-wasting mistakes.

## When to Update AGENTS.md

**Add to AGENTS.md when:**
1. User explicitly corrects you on something non-obvious
2. The same mistake would waste time if repeated in a future session
3. The correction is NOT easily discoverable within ~5 minutes of basic exploration

**Do NOT add to AGENTS.md when:**
- The information is obvious ("update imports when moving files")
- It's easily discoverable through code reading (module locations, first imports)
- It's implementation-specific detail that will go stale quickly
- It's a command/tool you didn't actually use

## What to Add

### Pattern: User Corrected a Mistake

**Example from clipboard session:**
```markdown
### File Organization

**`src/rlmy/tools/`** — LLM-callable tools (shell, edit, MCP connectors)  
**`src/rlmy/agent/`** — Agent internals (RLM, UI utilities, commands, trajectory)

**Rule**: If it's called by the LLM as a tool → `tools/`. If it's UI/infrastructure for the agent itself → `agent/`.

**Example**: `clipboard.py` goes in `agent/` because it's a prompt_toolkit UI utility, not an LLM tool.
```

**Why this was added:** User explicitly corrected placement of `clipboard.py` from `tools/` to `agent/`. This is a non-obvious architectural decision that prevents wasted refactoring time.

### Pattern: User Established a Philosophy

**Example from clipboard session:**
```markdown
### Testing Philosophy

**Functional tests over exhaustive unit tests**. Test critical paths end-to-end, not isolated units.
```

**Why this was added:** User explicitly stated preference for functional tests over unit tests. This prevents wasted effort writing the wrong kind of tests.

### Pattern: Command Correction

**Example from clipboard session:**
```bash
# Run tests (use venv python, not bare python)
.venv/bin/python -m pytest
```

**Why this was added:** Multiple lessons about using `.venv/bin/python` instead of bare `python`. This is non-obvious and causes import errors.

## What to Remove

### Pattern: Information Went Stale

**Example:**
```markdown
### Key Modules

- **`src/rlmy/cli.py`** — Entry point...
- **`src/rlmy/agent/main.py`** — Main conversation loop...
[...20 more modules...]
```

**Why this was removed:** This list will go stale as modules are added/renamed/removed. It's also easily discoverable by reading the first few imports in `cli.py`.

### Pattern: Obvious Advice

**Example:**
```markdown
### Import Paths

After moving files, update **all** import references...
```

**Why this was removed:** This is obvious. If imports break, the error message tells you exactly what to fix.

## Update Process

1. **During the session:** Take note of explicit lessons from the user
2. **Before completion:** Review which lessons meet the criteria above
3. **Update AGENTS.md:** Add a new subsection or update existing one

## Maintenance

Review AGENTS.md and remove sections that:
- Reference deprecated code
- Describe obvious patterns
- Haven't prevented mistakes in recent sessions

The goal is to keep AGENTS.md under few hundred lines focused on high-value lessons.

## Skills to Run Before/After Updates

- **Before adding:** Use `skills/review.md` to check if the proposed addition meets criteria
- **After updating:** Test that AGENTS.md is still parseable and clear

## Example Session Flow

```
User: "why is clipboard.py in the tools folder? it's not a tool."
You: [moves file to agent/]
User: [approves]

→ Action: Add "File Organization" pattern to AGENTS.md
→ Rationale: Non-obvious architectural decision, prevents future mistakes
```

## Non-Example Session Flow

```
You: [creates function with missing import]
Pytest: ImportError: No module named 'pillow'
You: [installs pillow]

→ Action: DO NOT add to AGENTS.md
→ Rationale: Obvious — the error message tells you exactly what to do
```
