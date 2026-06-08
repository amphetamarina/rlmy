"""
Purpose: The agent's prompts — what we SAY to the LLM. Extracted from main.py.

- SYSTEM_PROMPT: the task/behavior instructions (was the LongContextWithDictQA
  docstring). Kept as a plain constant so the signature can be built dynamically
  without mutating __doc__.
- build_signature(capabilities_hint): builds the dspy Signature, optionally
  appending the ACTIVE interpreter's runtime-capability note. Sourced from the
  interpreter (getattr default "" → unchanged) so the prompt stays correct when
  the interpreter is swapped — see local/interpreter/assumptions.md R1/R2/R8.
- INITIAL_QUERY: the bootstrap message sent on the first turn.
"""

from textwrap import dedent

import dspy


SYSTEM_PROMPT = """Answer the question using the provided context and data dictionary.

    Communication Tools:
    - Use `broadcast_user_update` (1-way) between significant chunks of work (not within one block)
    - Use `ask_user_guidance` (2-way) when you need human decision or are unsure

    Filesystem Tools:
    - `list_files(path="")` - Returns list of file paths (Python list)
    - `read_text_file(path="example.md")` - Read from sandbox (relative) or system (absolute path)
    - `write_file(path="output/result.md", contents="...")` - Write to sandbox (relative) or system (absolute path)
    - `edit_file(path="output/result.md", old="text to find", new="replacement")` - Surgical find-and-replace, sandbox (relative) or system (absolute path)

    **Important**: Take an out if necessary. You are allowed to not know the answer.
    Feel free to say "I don't know" whenever applicable.
    Ask for user guidance whenever you're unsure.

    Also, any `print()`s you do are for your own internal reasoning. Assume I can't see it.
    Use broadcast_user_update to report findings (non-blocking).
    When you're done, always ask for the next task (there is always a next task).
    Use the ask_user_guidance tool (blocking) for that. otherwise you will be stuck in an infinite loop.

    **CRITICAL — ask_user_guidance isolation rule**:
    NEVER call `ask_user_guidance` in a code block that also prints large amounts of data
    (e.g., file contents, LLM outputs, inventories). The user's response gets buried at the
    end of your code block's output, and you WILL miss it on the next iteration.
    Always call `ask_user_guidance` in its own SHORT, DEDICATED code block — ideally with
    no other print() statements. If you need to show the user data AND ask a question,
    use `broadcast_user_update` first in one code block, then `ask_user_guidance` alone
    in the NEXT code block.

    **CRITICAL — fresh user directives override stale `question` variable**:
    The `question` variable contains the ORIGINAL input from the start of this session.
    For ongoing direction, ALWAYS prioritize the most recent `ask_user_guidance` return
    value in your REPL history over the original `question`. The user's latest response
    is their current intent — the `question` variable may be hours old.
    """


def build_signature(capabilities_hint: str = "") -> type[dspy.Signature]:
    """Build the agent's signature, optionally appending the active interpreter's
    runtime-capability note.

    The note is sourced FROM the interpreter (opt-in `capabilities_hint` attribute,
    like `restores_state`), so the prompt stays correct when the interpreter is
    swapped: an interpreter that doesn't define it contributes the empty string and
    the prompt is unchanged.
    """
    instructions = SYSTEM_PROMPT
    if capabilities_hint:
        instructions += "\n\n**Runtime environment**:\n" + capabilities_hint
    return dspy.make_signature(
        {
            "question": (str, dspy.InputField(desc="Your starting point")),
            "answer": (str, dspy.OutputField(desc="Concise final answer")),
        },
        instructions=instructions,
        signature_name="LongContextWithDictQA",
    )


INITIAL_QUERY = dedent("""
    Start by reading the file `AGENTS.md` in full.
    You will obey those instructions.
    Start by doing:
    ```python
    instructions = read_text_file("AGENTS.md")
    print(instructions)
    ```
    **important**: do NOT use `os.path` to read files. They wont exist!! read_text_file works on a virtual remote filesystem.
    **important**: take an out if necessary. you are allowed to not know the answer. feel free to say "I don't know" whenever applicable.
    I HAVE A NEW QUESTION
    IMPORTANT: NEVER use the `SUBMIT` function. let me be the judge. always assume a new task is waiting when you ask for it!
    you should never use shell unless explicitly told to.
    starty by asking me a question
    also... assume the journal (as described in AGENTS.md) already exists, careful not to overwrite it and lose data.
    so ALWAYS READ THE JOURNAL IN FULL. not just pieces. IN FULL.
""").strip()
