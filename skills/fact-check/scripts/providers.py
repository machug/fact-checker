"""Provider configuration, cost tracking, and profile management for fact-checker."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

PROFILES_DIR = Path.home() / ".config" / "fact-checker" / "profiles"
GLOBAL_CONFIG_PATH = Path.home() / ".claude" / "fact-checker" / "config.json"

# Cost per 1M tokens (approximate, as of 2026)
MODEL_COSTS = {
    # OpenAI GPT-5 family
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5.4-pro": {"input": 10.00, "output": 40.00},
    "gpt-5-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-nano": {"input": 0.10, "output": 0.40},
    # OpenAI legacy
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o3": {"input": 10.00, "output": 40.00},
    # Anthropic Claude
    "claude-opus-4-6-20250627": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6-20250627": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # Google Gemini
    "gemini/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini/gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    # xAI Grok
    "xai/grok-4-0709": {"input": 3.00, "output": 15.00},
    "xai/grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "xai/grok-3": {"input": 3.00, "output": 15.00},
    "xai/grok-3-mini": {"input": 0.30, "output": 0.50},
    # Other providers
    "mistral/mistral-large": {"input": 2.00, "output": 6.00},
    "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    # Codex CLI (subscription-based)
    "codex/gpt-5.3-codex": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.2-codex": {"input": 0.0, "output": 0.0},
    # Gemini CLI (account-based)
    "gemini-cli/gemini-3.1-pro-preview": {"input": 0.0, "output": 0.0},
    "gemini-cli/gemini-3-flash-preview": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}

# Check CLI tool availability
CODEX_AVAILABLE = shutil.which("codex") is not None
GEMINI_CLI_AVAILABLE = shutil.which("gemini") is not None
DEFAULT_CODEX_REASONING = "xhigh"


def load_global_config() -> dict:
    """Load global config from ~/.claude/fact-checker/config.json."""
    if not GLOBAL_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(GLOBAL_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in global config: {e}", file=sys.stderr)
        return {}


def save_global_config(config: dict):
    """Save global config."""
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def get_available_providers() -> list[tuple[str, Optional[str], str]]:
    """Get list of providers with configured API keys.

    Returns list of (provider_name, env_var, default_model) tuples.
    """
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.4"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6-20250627"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
        ("xAI", "XAI_API_KEY", "xai/grok-4-0709"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.4"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/gpt-5.3-codex"))
    if GEMINI_CLI_AVAILABLE:
        available.append(("Gemini CLI", None, "gemini-cli/gemini-3.1-pro-preview"))

    return available


def validate_model_credentials(models: list[str]) -> tuple[list[str], list[str]]:
    """Validate API keys for requested models. Returns (valid, invalid)."""
    provider_map = {
        "gpt-": "OPENAI_API_KEY",
        "o1": "OPENAI_API_KEY",
        "o3": "OPENAI_API_KEY",
        "o4": "OPENAI_API_KEY",
        "claude-": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
        "xai/": "XAI_API_KEY",
        "mistral/": "MISTRAL_API_KEY",
        "groq/": "GROQ_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "codex/": None,
        "gemini-cli/": None,
    }

    valid = []
    invalid = []

    for model in models:
        if model.startswith("codex/"):
            (valid if CODEX_AVAILABLE else invalid).append(model)
            continue
        if model.startswith("gemini-cli/"):
            (valid if GEMINI_CLI_AVAILABLE else invalid).append(model)
            continue

        required_key = None
        for prefix, key in provider_map.items():
            if model.startswith(prefix):
                required_key = key
                break

        if required_key is None:
            valid.append(model)
        elif os.environ.get(required_key):
            valid.append(model)
        else:
            invalid.append(model)

    return valid, invalid


def load_profile(profile_name: str) -> dict:
    """Load a saved profile by name."""
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        print(f"Error: Profile '{profile_name}' not found", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(profile_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in profile: {e}", file=sys.stderr)
        sys.exit(2)


def save_profile(profile_name: str, config: dict):
    """Save a profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    profile_path.write_text(json.dumps(config, indent=2))
    print(f"Profile saved to {profile_path}")


def list_profiles():
    """List all saved profiles."""
    print("Saved Profiles:\n")
    if not PROFILES_DIR.exists():
        print("  No profiles found.")
        return

    profiles = list(PROFILES_DIR.glob("*.json"))
    if not profiles:
        print("  No profiles found.")
        return

    for p in sorted(profiles):
        try:
            config = json.loads(p.read_text())
            print(f"  {p.stem}")
            print(f"    models: {config.get('models', 'not set')}")
            print()
        except Exception:
            print(f"  {p.stem} [error reading]")


def list_providers():
    """List all supported providers and their API key status."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.4, gpt-5-mini, gpt-5-nano"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6-20250627, claude-opus-4-6-20250627"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.5-pro, gemini/gemini-2.5-flash"),
        ("xAI", "XAI_API_KEY", "xai/grok-4-0709, xai/grok-3"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-5.4"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ]

    print("Supported providers:\n")
    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Models: {models}")
        print()

    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    print()

    gemini_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(Google account)':24} {gemini_status}")
    print()
