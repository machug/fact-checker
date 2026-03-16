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

# Use LiteLLM's community-maintained model cost registry at runtime.
# This stays current as users update their litellm package.
try:
    from litellm import model_cost as _litellm_model_cost
except ImportError:
    _litellm_model_cost = {}

# CLI tools aren't in LiteLLM's registry (subscription/account-based, no per-token cost)
_CLI_COSTS = {
    "codex/": {"input": 0.0, "output": 0.0},
    "gemini-cli/": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}


def get_model_cost(model: str) -> dict[str, float]:
    """Get cost per 1M tokens for a model, using LiteLLM's registry.

    Falls back to DEFAULT_COST for unknown models.
    """
    # CLI tools — free (subscription-based)
    for prefix, cost in _CLI_COSTS.items():
        if model.startswith(prefix):
            return cost

    # Look up in LiteLLM's registry (keys use per-token costs, we convert to per-1M)
    litellm_key = model.split("/", 1)[1] if "/" in model and model.split("/")[0] in (
        "gemini", "xai", "mistral", "groq", "deepseek", "openrouter"
    ) else model
    for key in (model, litellm_key):
        if key in _litellm_model_cost:
            entry = _litellm_model_cost[key]
            return {
                "input": entry.get("input_cost_per_token", 0) * 1_000_000,
                "output": entry.get("output_cost_per_token", 0) * 1_000_000,
            }

    return DEFAULT_COST

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
    Cost lookups are dynamic via LiteLLM's registry — any model ID
    that LiteLLM supports will work regardless of what's listed here.
    """
    providers = [
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/auto"),
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.4"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6-20250627"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
        ("xAI", "XAI_API_KEY", "xai/grok-4-0709"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/latest"))
    if GEMINI_CLI_AVAILABLE:
        available.append(("Gemini CLI", None, "gemini-cli/latest"))

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


def _discover_models_for_provider(provider_prefix: str, max_models: int = 5) -> str:
    """Discover available models from LiteLLM's registry for a provider prefix."""
    if not _litellm_model_cost:
        return "(update litellm for model list)"
    matches = [k for k in _litellm_model_cost if k.startswith(provider_prefix)]
    # Sort by name, show up to max_models
    matches.sort()
    if len(matches) > max_models:
        return ", ".join(matches[:max_models]) + f" (+{len(matches) - max_models} more)"
    return ", ".join(matches) if matches else "(none found in registry)"


def list_providers():
    """List all supported providers and their API key status."""
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-"),
        ("Google", "GEMINI_API_KEY", "gemini/"),
        ("xAI", "XAI_API_KEY", "xai/"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/"),
        ("Groq", "GROQ_API_KEY", "groq/"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/"),
    ]

    print("Supported providers:\n")
    for name, key, prefix in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        models = _discover_models_for_provider(prefix)
        print(f"  {name:12} {key:24} {status}")
        print(f"             Models: {models}")
        print()

    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    print()

    gemini_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(Google account)':24} {gemini_status}")
    print()
