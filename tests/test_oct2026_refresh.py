"""Tests for the October 2026 model refresh: new defaults, the GPT-6
Sol/Luna lineup, Claude 5.5 temperature handling, Gemini 4 Argon, and the
gpt-5.5 retirement from ChatGPT-account Codex."""

from __future__ import annotations

import models
import providers


def test_new_defaults(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "k")
    monkeypatch.setattr(providers, "CODEX_AVAILABLE", True)
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", True)
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["OpenAI"] == "gpt-6.1-sol"
    assert defaults["Anthropic"] == "claude-opus-5-5"
    assert defaults["Google"] == "gemini/gemini-4-argon"
    assert defaults["Codex CLI"] == "codex/gpt-6.1-sol"
    assert defaults["Antigravity CLI"] == "antigravity/gemini-3.8-flash-high"


def test_gemini4_argon_keeps_temperature():
    # verified 2026-10-14: gemini-4-argon accepts temperature and max_tokens
    assert not models.is_reasoning_model("gemini/gemini-4-argon")


def test_gpt6_sol_luna_are_reasoning_models():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "codex/gpt-6.1-sol"):
        assert models.is_reasoning_model(m)
        assert models.uses_max_completion_tokens(m)


def test_claude_55_omits_temperature():
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert models.claude_version(m) == (5, 5)
        assert models.is_reasoning_model(m)
        assert not models.uses_max_completion_tokens(m)


def test_codex_chatgpt_lineup():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra"):
        assert m in providers.CODEX_CHATGPT_MODELS
    assert "gpt-5.5" not in providers.CODEX_CHATGPT_MODELS
    assert "gpt-6.1-sol" in models.CODEX_CHATGPT_HINT
    assert "gpt-5.5 (until" not in models.CODEX_CHATGPT_HINT
