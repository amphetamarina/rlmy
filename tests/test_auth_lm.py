"""
Tests for the DSPy OAuth LM router (Unit: auth/lm.py).

Focus: build_lm routing and the static configuration it stamps on an OAuth LM
(model string, Codex responses endpoint, model_type). The credential-refresh
behavior is covered end to end in tests/functional/test_chatgpt_oauth_flow.py,
where the DSPy boundary is mocked — per this repo's "functional over exhaustive
unit" testing rule.
"""

import dspy
import pytest

from rlmy.auth.lm import OAuthLM, build_lm
from rlmy.auth.openai_codex import BACKEND_BASE


class TestBuildLm:
    def test_plain_model_string_returns_plain_dspy_lm(self):
        lm = build_lm("openai/gpt-4o-mini", cache=False)
        assert isinstance(lm, dspy.LM)
        assert not isinstance(lm, OAuthLM)

    def test_unknown_oauth_provider_raises_educational_error(self):
        # An unsupported "*-oauth/" prefix must fail loudly, not silently fall
        # through to LiteLLM as a bogus provider.
        with pytest.raises(RuntimeError, match="chatgpt-oauth"):
            build_lm("xai-oauth/grok-4", cache=False)

    def test_chatgpt_oauth_returns_oauth_lm_on_codex_responses_backend(self):
        lm = build_lm("chatgpt-oauth/gpt-5.5", cache=False)
        assert isinstance(lm, OAuthLM)
        assert lm.model == "openai/gpt-5.5"
        assert lm.kwargs["api_base"] == BACKEND_BASE
        assert lm.model_type == "responses"

    def test_chatgpt_oauth_lm_sets_codex_responses_flags(self):
        # The ChatGPT Codex backend requires store=false and stream=true.
        lm = build_lm("chatgpt-oauth/gpt-5.5")
        assert lm.kwargs["store"] is False
        assert lm.kwargs["stream"] is True
        assert lm.cache is False
