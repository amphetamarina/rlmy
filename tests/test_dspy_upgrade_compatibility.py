"""
Test suite for DSPy upgrade compatibility issues.

This captures breaking changes when upgrading DSPy versions.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import sys

# Add prototype to path
sys.path.insert(0, str(Path(__file__).parent.parent / "prototype"))

from cli_proto import InterruptableRLM
from dspy.primitives.code_interpreter import FinalOutput


@pytest.mark.anyio
async def test_process_execution_result_signature_compatibility():
    """
    Test that InterruptableRLM.aforward() correctly calls _process_execution_result
    with the signature expected by DSPy 3.1.3.
    
    DSPy 3.1.3 changed _process_execution_result to require 5 args:
        (self, pred, code, result, history, output_field_names)
    
    Previously it was:
        (self, action, result, history, output_field_names)
    
    This test ensures the call includes the 'code' parameter.
    """
    # Setup: Create minimal InterruptableRLM instance
    rlm = InterruptableRLM("question -> answer", max_iterations=1)
    
    # Mock the interpreter to avoid actual code execution
    mock_interpreter = MagicMock()
    mock_interpreter.execute = MagicMock(return_value=FinalOutput({"answer": "test answer"}))
    mock_interpreter.shutdown = MagicMock()
    mock_interpreter.tools = {}
    mock_interpreter.output_fields = []
    mock_interpreter._tools_registered = False
    
    # Mock the generate_action call to return a SUBMIT action
    mock_prediction = MagicMock()
    mock_prediction.reasoning = "I will submit the answer"
    mock_prediction.code = "```python\nSUBMIT(answer='test answer')\n```"
    
    with patch.object(rlm, '_interpreter', mock_interpreter):
        with patch.object(rlm.generate_action, 'acall', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_prediction
            
            # Spy on _process_execution_result to verify it's called correctly
            original_process = rlm._process_execution_result
            call_args_captured = []
            
            def spy_process(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                return original_process(*args, **kwargs)
            
            with patch.object(rlm, '_process_execution_result', side_effect=spy_process):
                # Execute: This should trigger _process_execution_result without hitting extract fallback
                result = await rlm.aforward(question="test question")
            
            # Verify: _process_execution_result was called with correct number of args
            assert len(call_args_captured) > 0, "_process_execution_result was never called"
            
            args, kwargs = call_args_captured[0]
            
            # The method signature in DSPy 3.1.3 is:
            # _process_execution_result(self, pred, code, result, history, output_field_names)
            # The spy function receives args without 'self', so we expect 5 args
            assert len(args) == 5, (
                f"_process_execution_result called with {len(args)} args, "
                f"expected 5 (pred, code, result, history, output_field_names). "
                f"Args: {[type(a).__name__ for a in args]}"
            )
            
            # Verify the 'code' parameter is present (2nd arg in spy = 3rd in actual method)
            # args[0] = pred
            # args[1] = code (the one we're checking - THIS IS THE FIX!)
            # args[2] = result
            # args[3] = history
            # args[4] = output_field_names
            assert isinstance(args[1], str), (
                f"Expected 'code' (2nd arg) to be a string, got {type(args[1]).__name__}"
            )
            
            # Verify result completed successfully
            assert hasattr(result, 'answer')


@pytest.mark.anyio
async def test_interruptable_rlm_basic_execution():
    """
    Integration test: verify InterruptableRLM can execute a simple forward pass.
    
    This is a broader test to ensure the DSPy upgrade didn't break basic functionality.
    """
    # Create a simple RLM that should work
    rlm = InterruptableRLM("question -> answer", max_iterations=1, verbose=False)
    
    # Mock interpreter for controlled execution
    mock_interpreter = MagicMock()
    mock_interpreter.tools = {}
    mock_interpreter.output_fields = []
    mock_interpreter._tools_registered = False
    mock_interpreter.shutdown = MagicMock()
    
    # Mock to return FinalOutput
    mock_interpreter.execute = MagicMock(return_value=FinalOutput({"answer": "test answer"}))
    
    # Mock generate_action to return valid action
    mock_prediction = MagicMock()
    mock_prediction.reasoning = "I will submit the answer"
    mock_prediction.code = "```python\nSUBMIT(answer='test answer')\n```"
    
    with patch.object(rlm, '_interpreter', mock_interpreter):
        with patch.object(rlm.generate_action, 'acall', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_prediction
            
            # This should complete without raising TypeError
            result = await rlm.aforward(question="test question")
            
            assert hasattr(result, 'answer')
            assert result.answer == "test answer"


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v", "-s"])
