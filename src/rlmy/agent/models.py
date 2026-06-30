"""
Purpose: Construct the main and sub LMs for the agent from configured model strings.
Usage: from rlmy.agent.models import lm, sub_lm
Key Components: build_lm() routes a model string to a plain DSPy LM or a
                subscription OAuth LM (e.g. "chatgpt-oauth/gpt-5.5").
Conventions: Model strings come from RLM_MAIN_MODEL / RLM_SUB_MODEL (set by the CLI
             from config). Subscription providers use a "<provider>-oauth/<model>"
             prefix; everything else is a standard LiteLLM model string.
"""

import os

from rlmy.auth.lm import build_lm

OPUS = "bedrock/us.anthropic.claude-opus-4-6-v1"
SONNET = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

_main_model = os.getenv('RLM_MAIN_MODEL', OPUS)
_sub_model = os.getenv('RLM_SUB_MODEL', SONNET)

# Main LM (strategist) drives RLM reasoning + code generation; sub-LM (worker)
# backs llm_query()/llm_query_batched() inside the REPL.
lm = build_lm(_main_model, cache=True)
sub_lm = build_lm(_sub_model, cache=True)
