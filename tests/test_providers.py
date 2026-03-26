"""Tests for provider config and cost lookup in providers.py."""

import os

from providers import get_model_cost, validate_model_credentials, DEFAULT_COST


class TestGetModelCost:
    def test_cli_tools_are_free(self):
        assert get_model_cost("codex/latest") == {"input": 0.0, "output": 0.0}
        assert get_model_cost("gemini-cli/latest") == {"input": 0.0, "output": 0.0}

    def test_unknown_model_returns_default(self):
        cost = get_model_cost("totally-unknown-model-xyz")
        assert cost == DEFAULT_COST

    def test_foundry_model_returns_default(self):
        # Foundry models aren't in litellm registry — should fall back to default
        cost = get_model_cost("foundry/my-deployment")
        assert cost == DEFAULT_COST


class TestValidateModelCredentials:
    def test_openai_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        valid, invalid = validate_model_credentials(["gpt-5.4"])
        assert "gpt-5.4" in valid
        assert invalid == []

    def test_openai_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["gpt-5.4"])
        assert "gpt-5.4" in invalid

    def test_anthropic_valid(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        valid, invalid = validate_model_credentials(["claude-sonnet-4-6"])
        assert "claude-sonnet-4-6" in valid

    def test_foundry_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["foundry/gpt-5-mini"])
        assert "foundry/gpt-5-mini" in valid

    def test_foundry_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["foundry/gpt-5-mini"])
        assert "foundry/gpt-5-mini" in invalid

    def test_gemini_api(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        valid, _ = validate_model_credentials(["gemini/gemini-2.5-flash"])
        assert "gemini/gemini-2.5-flash" in valid

    def test_unknown_prefix_assumed_valid(self, monkeypatch):
        """Models with no recognized prefix are assumed valid (e.g. openrouter passthrough)."""
        valid, invalid = validate_model_credentials(["some-random-model"])
        assert "some-random-model" in valid
        assert invalid == []

    def test_multiple_models_mixed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["gpt-5.4", "claude-sonnet-4-6"])
        assert "gpt-5.4" in valid
        assert "claude-sonnet-4-6" in invalid

    def test_o_series_needs_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        valid, _ = validate_model_credentials(["o3", "o4-mini"])
        assert "o3" in valid
        assert "o4-mini" in valid

    def test_zai_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["zai/glm-5"])
        assert "zai/glm-5" in valid

    def test_zai_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["zai/glm-4.7"])
        assert "zai/glm-4.7" in invalid

    def test_moonshot_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["moonshot/kimi-k2.5"])
        assert "moonshot/kimi-k2.5" in valid

    def test_moonshot_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["moonshot/kimi-k2-thinking"])
        assert "moonshot/kimi-k2-thinking" in invalid
