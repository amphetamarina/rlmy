"""
Purpose: Configuration loader for RLMY.
Usage: config = RLMYConfig(); main_model, sub_model = config.get_models()
Key Components: RLMYConfig (env vars > config file > fail)
Conventions: Config file is optional. Env vars always take precedence.
             Users manage credentials themselves (ANTHROPIC_API_KEY, AWS_PROFILE, etc.)
"""

import os
from pathlib import Path
import sys
from textwrap import dedent

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python 3.10

import platformdirs


class RLMYConfig:
    """
    Load RLMY configuration with priority: env vars > config file > fail.
    
    Configuration locations:
    - Config dir: ~/.config/rlmy/ (XDG-compliant via platformdirs)
    - Config file: ~/.config/rlmy/config.toml
    - Sandboxes: ~/.config/rlmy/sandboxes/
    
    Environment variables (highest priority):
    - RLM_MAIN_MODEL: Main LM model string (e.g., "anthropic/claude-opus-4")
    - RLM_SUB_MODEL: Sub LM model string (optional, defaults to main)
    
    Config file format:
        [llm]
        main_model = "anthropic/claude-opus-4"
        sub_model = "bedrock/us.anthropic.claude-opus-4-8"
    """
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "rlmy"
        self.config_file = self.config_dir / "config.toml"
        self.sandboxes_dir = self.config_dir / "sandboxes"
    
    def get_models(self) -> tuple[str, str]:
        """
        Return (main_model, sub_model) strings for DSPy.
        Priority: RLM_MAIN_MODEL env var > config file > raise RuntimeError.
        
        Returns:
            tuple[str, str]: (main_model_string, sub_model_string)
        
        Raises:
            RuntimeError: If no configuration found
        """
        # Priority 1: Environment variables
        main = os.getenv("RLM_MAIN_MODEL")
        sub = os.getenv("RLM_SUB_MODEL")
        if main:
            return main, sub or main
        
        # Priority 2: Config file
        if self.config_file.exists():
            try:
                with open(self.config_file, "rb") as f:
                    config = tomllib.load(f)
                
                main = config.get("llm", {}).get("main_model")
                sub = config.get("llm", {}).get("sub_model")
                if main:
                    return main, sub or main
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}", file=sys.stderr)
        
        # Priority 3: Fail with helpful message
        raise RuntimeError(dedent(f"""
            No RLMY configuration found.

            Run the setup wizard:
              rlmy
              (First run will prompt for configuration)

            Or create config manually:
              mkdir -p {self.config_dir}
              cat > {self.config_file} << 'EOF'
              [llm]
              main_model = 'anthropic/claude-opus-4'
              sub_model = 'bedrock/us.anthropic.claude-opus-4-8'
              EOF

            Or use environment variables:
              export RLM_MAIN_MODEL='anthropic/claude-opus-4'
              export RLM_SUB_MODEL='bedrock/us.anthropic.claude-opus-4-8'
        """).strip())
    
    def ensure_sandboxes_dir(self):
        """Ensure sandboxes directory exists."""
        self.sandboxes_dir.mkdir(parents=True, exist_ok=True)
