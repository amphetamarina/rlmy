"""
Tests for the rlmy package core functionality.

Focus: config loading, package initialization, module imports.
These are the "safety net" tests before any refactoring.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPackageInit:
    """Verify package imports and exposes expected attributes."""

    def test_import_rlmy(self):
        """Package imports without error."""
        import rlmy
        assert hasattr(rlmy, "__version__")

    def test_version_is_string(self):
        """Version is a non-empty string."""
        import rlmy
        assert isinstance(rlmy.__version__, str)
        assert len(rlmy.__version__) > 0


class TestRLMYConfig:
    """Test configuration loading priority chain."""

    def test_instantiation(self):
        """RLMYConfig instantiates with correct default paths."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()
        assert config.config_dir is not None
        assert config.config_file is not None
        assert config.sandboxes_dir is not None
        assert config.config_file.name == "config.toml"

    def test_config_dir_is_xdg_compliant(self):
        """Config directory uses platformdirs (XDG-compliant)."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()
        # Should be under user config directory (platform-specific)
        assert "rlmy" in str(config.config_dir)

    def test_get_models_from_env_vars(self):
        """Environment variables take highest priority."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()

        with patch.dict(os.environ, {
            "RLM_MAIN_MODEL": "anthropic/claude-opus-4",
            "RLM_SUB_MODEL": "bedrock/us.anthropic.claude-opus-4-8",
        }):
            main, sub = config.get_models()
            assert main == "anthropic/claude-opus-4"
            assert sub == "bedrock/us.anthropic.claude-opus-4-8"

    def test_get_models_sub_defaults_to_main(self):
        """If RLM_SUB_MODEL not set, sub defaults to main."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()

        env = {"RLM_MAIN_MODEL": "anthropic/claude-opus-4"}
        with patch.dict(os.environ, env, clear=False):
            # Remove RLM_SUB_MODEL if it exists
            with patch.dict(os.environ, {"RLM_SUB_MODEL": ""}, clear=False):
                os.environ.pop("RLM_SUB_MODEL", None)
                main, sub = config.get_models()
                assert main == "anthropic/claude-opus-4"
                assert sub == "anthropic/claude-opus-4"

    def test_get_models_raises_when_no_config(self, tmp_path):
        """RuntimeError raised when no env vars and no config file."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()

        # Point config to a non-existent location
        config.config_file = tmp_path / "nonexistent" / "config.toml"

        # Remove env vars
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No RLMY configuration found"):
                config.get_models()

    def test_ensure_sandboxes_dir_creates_directory(self, tmp_path):
        """ensure_sandboxes_dir() creates the directory if missing."""
        from rlmy.config import RLMYConfig
        config = RLMYConfig()
        config.sandboxes_dir = tmp_path / "test_sandboxes"

        assert not config.sandboxes_dir.exists()
        config.ensure_sandboxes_dir()
        assert config.sandboxes_dir.exists()


class TestModuleImports:
    """Verify all package modules import cleanly."""

    def test_import_config(self):
        from rlmy import config
        assert hasattr(config, "RLMYConfig")

    def test_import_wizard(self):
        from rlmy import wizard
        assert hasattr(wizard, "run_wizard")

    def test_import_cli(self):
        from rlmy import cli
        assert hasattr(cli, "main")

