# JOURNAL SYSTEM INSTRUCTIONS

## Core Philosophy

This journal system exists because **power can go out at any time**. When you restart, you need to remember everything that happened, all relevant details, paths walked, decisions made, and user corrections. Think of this as your memory system.

## File Structure

- **AGENTS.md** (this file) - Permanent, reusable knowledge about HOW to maintain the journal. Read this on every session restart.
- **session-journal.md** - Your actual memory/diary. Task-specific, chronological, append-only.

## Writing Style: Narrative, Not Bullets

**DO NOT write brief bullet points.** Each entry should read like a mini blog post, tweet, or diary entry. Capture your thoughts and memories in narrative form. The goal is to preserve information, not compress it prematurely.

### Regular Entry (every 2-5 steps)
Think: "It's been a while, I should take note of what I just did in case power goes off."

Write as a narrative capturing:
- What you just accomplished
- What you learned or discovered
- Specific commands, file paths, variable names, error codes
- Your reasoning and thought process
- Any surprises or unexpected results

Example of a good regular entry:
```
{"$schema":"journal-entry","type":"entry","id":15,"ts":"2026-01-21T14:23:00"}
## [ENTRY-015] Successfully extracted vendor IDs from dataset

I just finished parsing the vendor dataset file located at `data/vendors.json`. The file contained 47 vendor records, each with an ID field formatted as "VND-XXXXX". I used a simple regex pattern `r'VND-\d{5}'` to extract them all into a list called `vendor_ids`.

One thing that surprised me: three records had malformed IDs ("VND-12A45", "VND-", "VND-999999"). I decided to filter these out using a validation function that checks for exactly 5 digits. The clean list now has 44 valid IDs stored in `valid_vendor_ids`.

Next step is to cross-reference these IDs against the contract database to find which vendors have active contracts. The user mentioned earlier that they care most about active contracts from 2025-2026, so I'll filter by date range.
```

### Checkpoint Entry (every ~10 regular entries)
Think: "If power goes out and I restart with ONLY this checkpoint visible, do I have the COMPLETE picture?"

A checkpoint is a **full memory dump**. Include:
- **Complete context**: What is the overall task/project? What's the end goal?
- **Full history**: What have you done so far? (Compress older work, but don't lose critical details)
- **Current state**: Where are you right now? What variables exist? What files have been created?
- **What's next**: What remains to be done? What's the plan?
- **User guidance**: Any corrections, preferences, or specific instructions the user gave (VERBATIM when they corrected you)
- **Specifics**: File paths, command syntax, error patterns, data structures - anything you'd need to continue seamlessly

Example of a good checkpoint:
```
{"$schema":"journal-entry","type":"checkpoint","id":20,"ts":"2026-01-21T15:45:00"}
## [CHECKPOINT-020] Vendor Contract Analysis - Midpoint Status

**PROJECT OVERVIEW**: I'm analyzing vendor contracts to identify which vendors have active contracts in 2025-2026 and calculating their total contract values. The end deliverable is a report file `output/vendor-analysis-report.md` with a ranked list of vendors by contract value.

**WHAT I'VE DONE SO FAR** (entries 1-19):
I started by exploring the data structure. The input files are in `data/vendors.json` (47 vendors) and `data/contracts.json` (203 contracts). I discovered that vendor IDs follow the format "VND-XXXXX" with exactly 5 digits, though 3 records were malformed and I filtered them out, leaving 44 valid vendors.

Then I built a cross-reference system. I wrote a function `get_vendor_contracts(vendor_id, start_date, end_date)` that filters contracts by vendor and date range. The user specifically asked me to focus on 2025-2026 contracts (they said: "we only care about recent contracts, anything before 2025 is irrelevant for this analysis"). I stored this in a dictionary `vendor_to_contracts` where keys are vendor IDs and values are lists of contract objects.

I hit an issue with date parsing - some contracts had dates in "MM/DD/YYYY" format while others used "YYYY-MM-DD". I wrote a helper function `normalize_date(date_str)` that handles both formats and converts to datetime objects for comparison.

**CURRENT STATE**:
- Variables in memory: `vendor_ids` (list of 44), `valid_vendor_ids` (list of 44), `vendor_to_contracts` (dict), `contracts_2025_2026` (list of 87 contracts)
- Files created: `output/data-exploration.txt` (initial findings), `output/vendor-contracts-raw.json` (intermediate data)
- Current position: I've filtered down to 87 active contracts across 31 vendors (13 vendors have no active contracts in the target period)

**WHAT'S NEXT**:
1. Calculate total contract value per vendor (sum the `contract_value` field)
2. Rank vendors by total value (descending)
3. Generate the final report with vendor details, contract counts, and total values
4. User asked for a summary section at the top highlighting the top 5 vendors (exact quote: "put a summary at the top, I don't want to scroll through everything")

**IMPORTANT USER CORRECTIONS**:
- Entry 8: I initially tried to sum contract values as strings. User corrected: "those are currency strings with $ and commas, you need to parse them first". I wrote `parse_currency(value_str)` to handle this.
- Entry 12: User clarified that "active" means contract_status == "active" OR "pending", not just "active"

**TECHNICAL NOTES**:
- The contracts.json file is large (2.3MB), so I'm avoiding re-reading it. All data is in memory.
- Some contracts have null values for contract_value - I'm treating these as $0 for now
- File paths are all relative to the sandbox root `dspy_teacher/dspy-rlm/filesystem`
```

## Entry Format

Each entry has two parts:
1. **JSON metadata line** (for programmatic access)
2. **Markdown content** (human-readable narrative)

```
{"$schema":"journal-entry","type":"entry|checkpoint","id":N,"ts":"ISO-8601-timestamp"}
## [TYPE-ID] Title
Narrative content here...
```

## Maintenance Guidelines

1. **Write frequently**: Create a regular entry every 2-5 code execution steps
2. **Checkpoint regularly**: Create a checkpoint every ~10 regular entries
3. **Persist to disk**: After writing 2-3 new entries, call `write_file("session-journal.md", journal)` to save
4. **Keep journal in memory**: Maintain the full journal string in a variable, append to it, then write to disk periodically
5. **Compress intelligently in checkpoints**: When writing a checkpoint, give less space to old/irrelevant work, but preserve critical details

## How to Use on Session Restart

1. Read `AGENTS.md` (this file) to remember the system
2. Read `session-journal.md` to restore your memory
3. Find the most recent checkpoint using regex: `r'^\{"\$schema":"journal-entry".*"type":"checkpoint".*\}$'`
4. Read from that checkpoint to the end to get recent context
5. Continue work from there

## Simple Access Patterns

**Get everything since last checkpoint:**
```python
import re, json
journal = read_file("session-journal.md")
pattern = r'^\{"\$schema":"journal-entry".*\}$'
matches = list(re.finditer(pattern, journal, re.MULTILINE))
for m in reversed(matches):
    entry = json.loads(m.group())
    if entry['type'] == 'checkpoint':
        recent_context = journal[m.start():]
        break
```

**Count total entries:**
```python
entries = re.findall(pattern, journal, re.MULTILINE)
print(f"Total entries: {len(entries)}")
```

## Key Principles

- **Narrative over brevity**: Preserve information, don't compress prematurely
- **Specificity**: Include exact commands, file paths, error codes, user quotes
- **Checkpoints are memory dumps**: Should contain everything needed to resume work
- **User corrections are sacred**: Write them verbatim so you don't repeat mistakes
- **Simple, not brittle**: Keep the system easy to maintain and understand

---
---
---

**important**: take an out if necessary. you are allowed to not know the answer. feel free to say "I don't know" whenever applicable. important!!
important: you will work from now on making extensive use of your planning capabilities. your plans should be well thought, step by step, and employ the strategy described at `skills/mapreduce.md`.
you will work from now on making extensive use of your planning capabilities. your plans should be well thought, step by step, and employ the strategy described at `skills/mapreduce.md`. important!
