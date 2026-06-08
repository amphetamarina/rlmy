import os

import dspy

OPUS = "bedrock/us.anthropic.claude-opus-4-6-v1"
SONNET = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Model selection: env vars override hardcoded defaults
_main_model = os.getenv('RLM_MAIN_MODEL', OPUS)
_sub_model = os.getenv('RLM_SUB_MODEL', SONNET)

# Main LM (strategist) — drives RLM reasoning + code generation
lm = dspy.LM(model=_main_model, cache=True)

# Sub-LM (worker) — used by llm_query()/llm_query_batched() inside the REPL
sub_lm = dspy.LM(model=_sub_model, cache=True)

