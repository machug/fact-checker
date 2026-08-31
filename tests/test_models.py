"""Tests for model utilities in models.py."""

from models import (
    is_reasoning_model,
    uses_max_completion_tokens,
    claude_version,
    CostTracker,
    parse_triage_response,
)


class TestIsFixedTemperatureModel:
    def test_o1_model(self):
        assert is_reasoning_model("o1") is True
        assert is_reasoning_model("o1-preview") is True

    def test_o3_model(self):
        assert is_reasoning_model("o3") is True
        assert is_reasoning_model("o3-mini") is True

    def test_o4_model(self):
        assert is_reasoning_model("o4-mini") is True

    def test_o_series_with_prefix(self):
        assert is_reasoning_model("openai/o3-mini") is True
        assert is_reasoning_model("openrouter/o4-mini") is True

    def test_gpt5(self):
        assert is_reasoning_model("gpt-5.4") is True
        assert is_reasoning_model("gpt-5-mini") is True
        assert is_reasoning_model("openai/gpt-5.4") is True

    def test_xai_reasoning_models(self):
        assert is_reasoning_model("xai/grok-4-1-fast-reasoning") is True
        assert is_reasoning_model("xai/grok-4-fast-reasoning") is True
        assert is_reasoning_model("xai/grok-4.20-0309-reasoning") is True
        assert is_reasoning_model("xai/grok-4-0709") is False
        assert is_reasoning_model("xai/grok-4-1-fast-non-reasoning") is False

    def test_regular_models_are_not_fixed(self):
        assert is_reasoning_model("claude-sonnet-4-6") is False
        assert is_reasoning_model("gemini/gemini-3.1-pro-preview") is False
        assert is_reasoning_model("gpt-4o") is False

    def test_case_insensitive(self):
        assert is_reasoning_model("O3-Mini") is True
        assert is_reasoning_model("GPT-5.4") is True

    def test_claude_47_and_newer_are_fixed_temperature(self):
        assert is_reasoning_model("claude-opus-4-7") is True
        assert is_reasoning_model("claude-opus-4-8") is True
        assert is_reasoning_model("claude-opus-5") is True
        assert is_reasoning_model("claude-sonnet-5") is True
        assert is_reasoning_model("claude-fable-5") is True

    def test_claude_46_and_older_still_take_temperature(self):
        assert is_reasoning_model("claude-sonnet-4-6") is False
        assert is_reasoning_model("claude-opus-4-6") is False
        assert is_reasoning_model("claude-haiku-4-5") is False
        assert is_reasoning_model("claude-3-5-sonnet-20241022") is False

    def test_claude_id_variants(self):
        assert is_reasoning_model("claude-opus-4-7-20260214") is True
        assert is_reasoning_model("anthropic/claude-opus-5") is True
        assert is_reasoning_model("anthropic.claude-opus-4-7-v1:0") is True


class TestClaudeVersion:
    def test_parses_versions(self):
        assert claude_version("claude-opus-5") == (5, 0)
        assert claude_version("claude-sonnet-4-6") == (4, 6)
        assert claude_version("anthropic/claude-opus-4-7-20260214") == (4, 7)

    def test_non_claude_is_none(self):
        assert claude_version("gpt-5.6-sol") is None
        assert claude_version("claude-3-5-sonnet-20241022") is None


class TestUsesMaxCompletionTokens:
    def test_claude_uses_max_tokens(self):
        assert uses_max_completion_tokens("claude-opus-5") is False
        assert uses_max_completion_tokens("anthropic/claude-opus-4-7") is False

    def test_gpt5_uses_max_completion_tokens(self):
        assert uses_max_completion_tokens("gpt-5.6-sol") is True

    def test_xai_uses_max_tokens(self):
        assert uses_max_completion_tokens("xai/grok-4.20-0309-reasoning") is False


class TestCostTracker:
    def test_add_accumulates(self):
        tracker = CostTracker()
        tracker.add("codex/latest", 1000, 500)  # free model
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost == 0.0

    def test_multiple_models(self):
        tracker = CostTracker()
        tracker.add("codex/latest", 1000, 500)
        tracker.add("codex/latest", 2000, 1000)
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert len(tracker.by_model) == 1

    def test_summary_format(self):
        tracker = CostTracker()
        tracker.add("codex/latest", 1000, 500)
        summary = tracker.summary()
        assert "Cost Summary" in summary
        assert "1,000" in summary

    def test_breakdown_str(self):
        tracker = CostTracker()
        tracker.add("codex/latest", 1000, 500)
        breakdown = tracker.breakdown_str()
        assert "latest" in breakdown  # short name from split("/")

    def test_breakdown_empty(self):
        tracker = CostTracker()
        assert tracker.breakdown_str() == "N/A"
