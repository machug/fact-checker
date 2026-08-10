"""Tests for the August 2026 refresh: non-retryable error fast-fail,
Codex ChatGPT-account preflight, and the Antigravity CLI provider."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import models
import providers
from models import (
    CODEX_CHATGPT_HINT,
    _call_cli_provider_with_retries,
    is_non_retryable_error,
    is_reasoning_model,
    resolve_antigravity_model,
)
from providers import (
    CODEX_CHATGPT_MODELS,
    codex_auth_mode,
    warn_codex_chatgpt_model_support,
    warn_openai_base_url_override,
)


class TestNonRetryableErrors:
    def test_chatgpt_account_model_rejection(self):
        msg = (
            '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
            '"message":"The \'gpt-5.3-codex\' model is not supported when using '
            'Codex with a ChatGPT account."}}'
        )
        assert is_non_retryable_error(msg)

    def test_invalid_request_error(self):
        assert is_non_retryable_error("litellm.BadRequestError: invalid_request_error")

    def test_model_not_found(self):
        assert is_non_retryable_error("model_not_found: no such model")
        assert is_non_retryable_error(
            "The model `gpt-9` does not exist or you do not have access to it"
        )

    def test_auth_errors(self):
        assert is_non_retryable_error("litellm.AuthenticationError: bad key")
        assert is_non_retryable_error("Incorrect API key provided")

    def test_azure_missing_deployment(self):
        assert is_non_retryable_error(
            "litellm.BadRequestError: OpenAIException - The API deployment for "
            "this resource does not exist."
        )

    def test_antigravity_deterministic_errors(self):
        assert is_non_retryable_error(
            "Antigravity CLI is not authenticated. Run `agy` interactively once"
        )
        assert is_non_retryable_error(
            "Antigravity CLI returned status ERROR: invalid model selection"
        )

    def test_transient_errors_still_retry(self):
        assert not is_non_retryable_error("rate limit exceeded")
        assert not is_non_retryable_error("connection reset by peer")
        assert not is_non_retryable_error("timed out after 600s")
        assert not is_non_retryable_error("500 Internal Server Error")

    def test_retry_loop_fails_fast_on_non_retryable(self):
        calls = []

        def failing_call():
            calls.append(1)
            raise RuntimeError(
                "The 'gpt-5.3-codex' model is not supported when using Codex "
                "with a ChatGPT account."
            )

        with patch("models.time.sleep") as mock_sleep:
            result = _call_cli_provider_with_retries(
                "codex/gpt-5.3-codex", failing_call
            )

        assert len(calls) == 1  # single attempt, no retries
        mock_sleep.assert_not_called()
        assert result.error is not None
        assert CODEX_CHATGPT_HINT in result.error

    def test_retry_loop_retries_transient_errors(self):
        calls = []

        def failing_call():
            calls.append(1)
            raise RuntimeError("rate limit exceeded")

        with patch("models.time.sleep"):
            result = _call_cli_provider_with_retries("codex/gpt-5.5", failing_call)

        assert len(calls) == models.MAX_RETRIES
        assert result.error is not None
        assert "rate limit" in result.error

    def test_retry_loop_success_parses_verdicts(self):
        content = (
            "[TRIAGE]\nid: 1\nverdict: CONFIDENT\nreason: well-known fact\n[/TRIAGE]"
        )
        result = _call_cli_provider_with_retries(
            "codex/gpt-5.5", lambda: (content, 10, 5)
        )
        assert result.error is None
        assert result.verdicts["1"]["verdict"] == "CONFIDENT"
        assert result.input_tokens == 10


class TestCodexChatGPTPreflight:
    def test_chatgpt_lineup_membership(self):
        assert "gpt-5.6-sol" in CODEX_CHATGPT_MODELS
        assert "gpt-5.5" in CODEX_CHATGPT_MODELS
        assert "gpt-5.3-codex" not in CODEX_CHATGPT_MODELS
        assert "gpt-5.5-pro" not in CODEX_CHATGPT_MODELS

    def test_auth_mode_chatgpt(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"x": 1}})
        )
        monkeypatch.setattr(providers.Path, "home", staticmethod(lambda: tmp_path))
        assert codex_auth_mode() == "chatgpt"

    def test_auth_mode_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers.Path, "home", staticmethod(lambda: tmp_path))
        assert codex_auth_mode() is None

    def test_warns_on_unsupported_model(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.3-codex", "gpt-5.5"])
        err = capsys.readouterr().err
        assert "gpt-5.3-codex" in err
        assert "ChatGPT account" in err

    def test_silent_on_supported_models(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.6-sol", "codex/gpt-5.5"])
        assert capsys.readouterr().err == ""

    def test_silent_on_apikey_auth(self, capsys):
        with patch("providers.codex_auth_mode", return_value="apikey"):
            warn_codex_chatgpt_model_support(["codex/gpt-5.3-codex"])
        assert capsys.readouterr().err == ""

    def test_silent_without_codex_models(self, capsys):
        with patch("providers.codex_auth_mode", return_value="chatgpt"):
            warn_codex_chatgpt_model_support(["gpt-5.5", "claude-opus-5"])
        assert capsys.readouterr().err == ""


class TestOpenAIBaseURLWarning:
    def test_warns_on_foreign_base_url(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://foo.azure.example/v1")
        warn_openai_base_url_override(["gpt-5.6-sol"])
        assert "OPENAI_BASE_URL" in capsys.readouterr().err

    def test_silent_on_official_base_url(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        warn_openai_base_url_override(["gpt-5.6-sol"])
        assert capsys.readouterr().err == ""

    def test_silent_without_openai_models(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://foo.azure.example/v1")
        warn_openai_base_url_override(["claude-opus-5", "gemini/gemini-3.6-flash"])
        assert capsys.readouterr().err == ""


class TestMoonshotReasoningDetection:
    def test_kimi_k25_and_later_are_reasoning(self):
        assert is_reasoning_model("moonshot/kimi-k2.5") is True
        assert is_reasoning_model("moonshot/kimi-k2.7-code") is True
        assert is_reasoning_model("moonshot/kimi-k3") is True

    def test_older_kimi_are_not_reasoning(self):
        assert is_reasoning_model("moonshot/kimi-k2") is False
        assert is_reasoning_model("moonshot/kimi-k1.5") is False


class TestAntigravityProvider:
    def test_resolve_slug_passthrough(self):
        assert (
            resolve_antigravity_model("antigravity/gemini-3.1-pro-high")
            == "gemini-3.1-pro-high"
        )
        assert (
            resolve_antigravity_model("antigravity/claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )

    def test_resolve_bare_antigravity_uses_default(self):
        assert resolve_antigravity_model("antigravity") is None

    def test_call_parses_agy_json(self):
        payload = {
            "conversation_id": "abc",
            "status": "SUCCESS",
            "response": "OK\n",
            "usage": {"input_tokens": 17579, "output_tokens": 26},
        }
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake) as mock_run,
        ):
            text, inp, out = models.call_antigravity_model(
                "sys", "user", "antigravity/gemini-3.1-pro-high"
            )
        assert text == "OK"
        assert (inp, out) == (17579, 26)
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd and "gemini-3.1-pro-high" in cmd
        assert "--output-format" in cmd and "json" in cmd

    def test_call_raises_on_error_status(self):
        payload = {
            "status": "ERROR",
            "response": "",
            "error": "invalid model selection",
        }
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="invalid model selection"):
                models.call_antigravity_model("sys", "user", "antigravity/bogus")

    def test_call_detects_unauthenticated(self):
        fake = type(
            "P",
            (),
            {
                "returncode": 1,
                "stdout": "Waiting for authentication (timeout 30s)...",
                "stderr": "",
            },
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="not authenticated"):
                models.call_antigravity_model("sys", "user", "antigravity")

    def test_call_rejects_json_without_response_field(self):
        payload = {"conversation_id": "abc", "status": "SUCCESS"}
        fake = type(
            "P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            with pytest.raises(RuntimeError, match="No response text"):
                models.call_antigravity_model("sys", "user", "antigravity")

    def test_call_oversized_prompt_raises_clear_error(self):
        import errno as _errno

        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch(
                "models.subprocess.run",
                side_effect=OSError(_errno.E2BIG, "Argument list too long"),
            ),
        ):
            with pytest.raises(RuntimeError, match="argument-size limit") as exc:
                models.call_antigravity_model("sys", "x" * 200_000, "antigravity")
        # Deterministic — must not be retried
        assert models.is_non_retryable_error(str(exc.value))

    def test_call_falls_back_to_raw_text_output(self):
        fake = type(
            "P", (), {"returncode": 0, "stdout": "plain text answer", "stderr": ""}
        )()
        with (
            patch("models.ANTIGRAVITY_AVAILABLE", True),
            patch("models.ANTIGRAVITY_PATH", "/usr/bin/agy"),
            patch("models.subprocess.run", return_value=fake),
        ):
            text, _, _ = models.call_antigravity_model("sys", "user", "antigravity")
        assert text == "plain text answer"

    def test_validate_credentials_antigravity(self):
        with patch("providers.ANTIGRAVITY_AVAILABLE", True):
            valid, invalid = providers.validate_model_credentials(
                ["antigravity/gemini-3.1-pro-high"]
            )
        assert valid == ["antigravity/gemini-3.1-pro-high"]
        with patch("providers.ANTIGRAVITY_AVAILABLE", False):
            valid, invalid = providers.validate_model_credentials(
                ["antigravity/gemini-3.1-pro-high"]
            )
        assert invalid == ["antigravity/gemini-3.1-pro-high"]

    def test_antigravity_cost_is_free(self):
        assert providers.get_model_cost("antigravity/gemini-3.1-pro-high") == {
            "input": 0.0,
            "output": 0.0,
        }
        assert providers.get_model_cost("antigravity") == {
            "input": 0.0,
            "output": 0.0,
        }

    def test_minimax_credentials(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        valid, _ = providers.validate_model_credentials(["minimax/MiniMax-M3"])
        assert "minimax/MiniMax-M3" in valid
        monkeypatch.delenv("MINIMAX_API_KEY")
        _, invalid = providers.validate_model_credentials(["minimax/MiniMax-M3"])
        assert "minimax/MiniMax-M3" in invalid
