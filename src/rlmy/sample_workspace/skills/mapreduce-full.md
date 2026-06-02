# SKILL: Intelligent MapReduce with Sub-LLMs

## Overview

This skill describes how to use sub-LLM calls (`llm_query`, `llm_query_batched`) to process, cluster, and distill large sets of information items (e.g., code review comments, coding practices, requirements, feedback) into actionable, structured output.

It covers the MapReduce pattern, prompt engineering best practices, and common anti-patterns discovered through practical experience.

---

## Core Philosophy

### MapReduce for Sub-Agents

"MapReduce" means intelligently breaking one task into self-contained subtasks (chunks), mapping each to a sub-agent, then reducing (consolidating) the results.

**Key principle**: Sub-agents read files/context **in full**. They do NOT guess structure or use regexes to sift through contents. They receive complete information to do their assigned task.

### The Two Cognitive Modes of LLMs

LLMs have two very different performance profiles:

1. **Holistic pattern recognition** (STRONG): "Read all of these items and tell me what themes you see." The LLM sees the whole picture and naturally finds overlaps, clusters, and patterns.

2. **Repetitive per-item processing** (WEAK): "Go through each of these 107 items and assign it to a category." This is effectively 107 instructions, and quality degrades predictably (see Research section below).

**Always prefer holistic framing over per-item enumeration.**

---

## The MapReduce Pattern

### Phase 1: MAP (Sub-agents process individual units)

Each sub-agent receives ONE unit of work (e.g., one code review, one document, one chunk) with full context, and extracts structured information.

```python
# Example: Extract practices from each CR's comments
prompts = []
for cr in code_reviews:
    conversation = format_human_comments(cr)  # Full context
    prompt = f"Analyze these code review comments and extract coding practices...\n{conversation}"
    prompts.append(prompt)

# Process in parallel batches
results = llm_query_batched(prompts)  # Each sub-agent gets full context
```

**Best practices for MAP phase:**
- Give each sub-agent COMPLETE context (full conversation, not snippets)
- Ask for structured output (JSON) so results are parseable
- Keep the task focused: extract, don't synthesize yet
- Typical batch size: 5-15 concurrent calls via `llm_query_batched`

### Phase 2: REDUCE (Consolidate results)

This is where most mistakes happen. The reduce phase takes the MAP outputs and must synthesize them into a coherent whole.

**⚠️ The reduce phase is NOT "dump everything into one LLM call."** It requires careful prompt engineering (see Anti-Patterns and Best Practices below).

#### Two-Stage Reduce (Recommended)

**Stage 1: Dedup/Merge** — Shrink the item count by merging near-identical or highly overlapping items.

```python
dedup_prompt = f"""Here are {N} items extracted from {M} sources. Many are duplicates or variations.
First, identify groups of near-identical items.
Then, merge each group into one consolidated item, preserving the source references.
Output the deduplicated list with original item IDs mapped to each merged item."""
```

**Stage 2: Cluster/Synthesize** — Organize the smaller set into themes/principles.

```python
cluster_prompt = f"""Here are {N_deduped} consolidated items.
First, think about the ideal way to organize these — what are the natural high-level themes?
Propose an outline: the sections and subsections that best capture these items.
Then, fill in each section concisely. Account for all items. 
If two themes feel very close, merge them and explain why."""
```

**Why two stages?** 
- Stage 1 is a simpler cognitive task (pattern matching for duplicates)
- Stage 2 operates on a much smaller set (30-60 items vs 100+)
- Each stage is independently verifiable

---

## Prompt Engineering Best Practices

### 1. Shrink-Wrap, Don't Prescribe

**Anti-pattern**: "Give me 10-15 clusters" or "Summarize in 3 paragraphs"
- Prescribing output length/count biases the model:
  - Too few → forces compression, loses signal
  - Too many → forces padding with filler

**Best practice**: Let the content determine the shape.
> "First, think of the ideal outline for this output — what are the natural sections and subsections? Then, fill them in concisely."

This "shrink-wraps" the output to the actual content.

### 2. Outline-First, Then Populate

**Anti-pattern**: "Here are 107 items, cluster them." (LLM may skim the middle)

**Best practice**: Force the LLM to think globally BEFORE committing to structure.
> "First, read all items carefully. What themes emerge? Propose an outline. THEN fill it in."

This prevents premature anchoring on early items and ensures the whole set informs the structure.

### 3. Don't Force Per-Item Enumeration

**Anti-pattern**: "Assign each of the 107 items to exactly one cluster. Output: {id: cluster}"
- This is effectively 107 micro-instructions
- Quality degrades in the middle (attention decay)
- Lost items are hard to detect without post-processing

**Best practice**: Let assignment happen naturally as part of synthesis.
> "For each theme, list which item IDs fall under it. Account for all items — no orphans."

The LLM assigns items as a natural byproduct of theme construction, not as a mechanical exercise.

### 4. Process-Forcing Over Pre-Sorting

**Anti-pattern**: Sort items by importance so "the best ones are at the top where the LLM pays attention."
- This is accepting the weakness (middle will be weak) rather than fixing it.

**Best practice**: Structure the prompt so the LLM must systematically engage with all items:
- Stage 1: Propose themes (forces full scan)
- Stage 2: Populate themes (forces coverage)
- Verification: Check all item IDs appear in output

### 5. Allow Breathing Room

**Anti-pattern**: Rigid constraints that force the LLM into unnatural output.

**Best practice**: Add flexibility:
> "If two themes feel very close, merge them and explain why."
> "Prioritize clean, actionable principles over forcing separation."

### 6. Maintain Traceability

Every level of output should link back to its sources:
```
Principle/Theme
  └── Constituent items (by ID)
       └── Source documents/CRs
            └── Original quotes/evidence
```

This chain ensures nothing is lost and everything can be verified.

---

## Research Context

### IFScale Paper (July 2025, arXiv:2507.11538)
- Tested 20 SOTA models on 10-500 simultaneous atomic instructions
- Key findings:
  - No hard cap at "15 instructions" — frontier models handle 100-200 reliably
  - Three degradation patterns: threshold decay, linear decay, exponential decay
  - **Primacy bias**: Models favor earlier instructions (especially at 150-200 density)
  - At 300-500 instructions, even best models drop to ~68% accuracy
  - Most failures are **omissions** (forgetting to include something)

### Key Distinction
- IFScale tested **independent output constraints** (include keyword X, keyword Y, ...)
- **Semantic clustering is fundamentally different** — it's ONE holistic task over many inputs
- LLMs are excellent at holistic pattern recognition over 100+ related items
- The risk is in forcing **per-item enumeration**, not in providing many items as input

### Practical Ceiling
- **< 150-200 short items** in one prompt → very high quality for clustering
- **200-500 items** → "lost in the middle" effects emerge, consider two-stage
- **Our tested case**: 107 practices at ~414 chars each (~44K chars, ~11K tokens) = well within range

---

## Worked Example: Extracting Coding Standards from Code Reviews

### Context
- Package: VMACollaborators
- 53 CRs, 507 total comments, 47 CRs with human comments

### Phase 1: MAP
- 47 sub-LLM calls via `llm_query_batched` (batches of 10)
- Each call received ONE CR's full human conversation threads
- Extracted: 139 raw practices with source CR references

### Phase 2: REDUCE Stage 1 (Dedup/Merge)
- 1 sub-LLM call with all 139 practices
- Merged into 107 deduplicated practices with categories
- Each practice preserved: source CRs, evidence quotes, consensus status

### Phase 2: REDUCE Stage 2 (Cluster/Synthesize) — [TODO: Execute]
- 1 sub-LLM call with all 107 practices
- Prompt uses shrink-wrap + outline-first pattern
- Output: Natural clusters of principles with full traceability chain

### Verification Steps (Programmatic)
After each reduce stage:
1. Check all input item IDs appear in the output
2. Flag any orphans and re-process if needed
3. Compute quantitative metrics (frequency, reviewer spread, conviction) in code, not LLM

---

## Anti-Pattern Checklist

Before sending a clustering/synthesis prompt, check:

| Check | ❌ Anti-Pattern | ✅ Fix |
|-------|----------------|--------|
| Output length | "Give me 10-15 clusters" | "Propose natural sections, then fill in" |
| Assignment method | "Assign each of N items to a cluster" | "For each theme, list which items belong" |
| Attention decay | Pre-sort by importance | Process-forcing (outline-first) |
| Cognitive load | One giant task | Two-stage (dedup, then cluster) |
| Flexibility | Rigid constraints | "Merge if close, explain why" |
| Traceability | Summarize without sources | "List item IDs, preserve evidence quotes" |
| Sub-agent context | Regex/snippet extraction | Read files in full |

---

## Template: Clustering Prompt

```
Here are {N} items extracted from {M} sources. Each item has an ID, description, 
and source reference.

First, read all items carefully.

Think about the ideal way to organize these — what are the natural high-level themes 
or principles that emerge? Propose an outline: the sections and subsections that would 
best capture these items. Don't force a specific number of sections — let the content 
determine the shape.

Then, fill in each section:
- Write a consolidated principle (concise but complete)
- List which item IDs fall under this section
- Include 1-3 of the strongest original evidence quotes
- Note the overall consensus (accepted vs debated)

Account for all {N} items — no orphans. If two themes feel very close, merge them 
and explain why. Prioritize clean, actionable output over forced separation.

[ITEMS]
{formatted_items}
```