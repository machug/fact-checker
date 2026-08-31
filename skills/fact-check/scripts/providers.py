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
    "antigravity/": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}


def get_model_cost(model: str) -> dict[str, float]:
    """Get cost per 1M tokens for a model, using LiteLLM's registry.

    Falls back to DEFAULT_COST for unknown models.
    """
    # CLI tools — free (subscription-based)
    if model == "antigravity":
        return _CLI_COSTS["antigravity/"]
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

# Check CLI tool availability — resolve to absolute paths to avoid PATH hijacking
CODEX_PATH = shutil.which("codex")
GEMINI_CLI_PATH = shutil.which("gemini")
ANTIGRAVITY_PATH = shutil.which("agy")
CODEX_AVAILABLE = CODEX_PATH is not None
GEMINI_CLI_AVAILABLE = GEMINI_CLI_PATH is not None
ANTIGRAVITY_AVAILABLE = ANTIGRAVITY_PATH is not None
DEFAULT_CODEX_REASONING = "xhigh"

# Models Codex CLI serves when authenticated with a ChatGPT account (not an
# API key). Rotates with OpenAI's ChatGPT lineup — see
# https://developers.openai.com/codex/models. Last verified 2026-08-31.
CODEX_CHATGPT_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.3-codex-spark",  # ChatGPT Pro only
}


def codex_auth_mode() -> Optional[str]:
    """Return Codex CLI auth mode ("chatgpt" or "apikey") or None if unknown."""
    auth_path = Path.home() / ".codex" / "auth.json"
    try:
        data = json.loads(auth_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    mode = data.get("auth_mode")
    if mode:
        return mode
    if data.get("OPENAI_API_KEY"):
        return "apikey"
    if data.get("tokens"):
        return "chatgpt"
    return None


def warn_codex_chatgpt_model_support(models: list[str]) -> None:
    """Warn upfront when a codex/ model won't work with ChatGPT-account auth.

    ChatGPT-account Codex serves only the current ChatGPT lineup; other models
    (gpt-5.3-codex, gpt-5.5-pro, ...) hard-fail with a 400. The supported set
    rotates, so this warns rather than blocks.
    """
    codex_models = [
        m.split("/", 1)[1] for m in models if m.startswith("codex/") and "/" in m
    ]
    if not codex_models or codex_auth_mode() != "chatgpt":
        return
    unsupported = [m for m in codex_models if m not in CODEX_CHATGPT_MODELS]
    if unsupported:
        print(
            f"Warning: Codex CLI is authenticated with a ChatGPT account, which "
            f"likely rejects: {', '.join(unsupported)}. ChatGPT-account models "
            f"(as of 2026-08-31): gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5. "
            f"Other models need Codex API-key auth or the OPENAI_API_KEY route.\n",
            file=sys.stderr,
        )


def warn_openai_base_url_override(models: list[str]) -> None:
    """Warn when a globally exported OPENAI_BASE_URL reroutes OpenAI models.

    A base URL exported for another tool (e.g. an Azure proxy) silently
    reroutes litellm's OpenAI calls and fails with confusing auth errors on
    models the proxy doesn't serve.
    """
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base_url and "api.openai.com" not in base_url and any(
        m.startswith(("gpt-", "o1", "o3", "o4")) for m in models
    ):
        print(
            f"Warning: OPENAI_BASE_URL is set to {base_url} — OpenAI models will "
            "route there, not to api.openai.com. If that's unintended, run with "
            "OPENAI_BASE_URL=https://api.openai.com/v1\n",
            file=sys.stderr,
        )


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
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    GLOBAL_CONFIG_PATH.chmod(0o600)


def get_available_providers() -> list[tuple[str, Optional[str], str]]:
    """Get list of providers with configured API keys.

    Returns list of (provider_name, env_var, default_model) tuples.
    Cost lookups are dynamic via LiteLLM's registry — any model ID
    that LiteLLM supports will work regardless of what's listed here.
    """
    providers = [
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/auto"),
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.6-sol"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-opus-5"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-3.1-pro-preview"),
        ("xAI", "XAI_API_KEY", "xai/grok-4.6"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-v4-pro"),
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/glm-5.3"),
        ("Moonshot (Kimi)", "MOONSHOT_API_KEY", "moonshot/kimi-k3"),
        ("MiniMax", "MINIMAX_API_KEY", "minimax/MiniMax-M3"),
        # Azure AI Foundry skipped from auto-detect — deployment names are user-specific
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/gpt-5.6-sol"))
    # Antigravity CLI is Gemini CLI's successor (consumer gemini-cli retired
    # 2026-06-18) — prefer it for auto-selection; gemini-cli/ still works if
    # requested explicitly.
    if ANTIGRAVITY_AVAILABLE:
        available.append(("Antigravity CLI", None, "antigravity/gemini-3.1-pro-high"))

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
        "foundry/": "AZURE_AI_API_KEY",
        "mistral/": "MISTRAL_API_KEY",
        "groq/": "GROQ_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "zai/": "ZAI_API_KEY",
        "zhipu/": "ZHIPUAI_API_KEY",  # Legacy prefix, use zai/ instead
        "moonshot/": "MOONSHOT_API_KEY",
        "minimax/": "MINIMAX_API_KEY",
        "codex/": None,
        "gemini-cli/": None,
        "antigravity/": None,  # Uses Google account via agy CLI
    }

    valid = []
    invalid = []

    for model in models:
        if model.startswith("codex/"):
            (valid if CODEX_AVAILABLE else invalid).append(model)
            continue
        if model == "antigravity" or model.startswith("antigravity/"):
            (valid if ANTIGRAVITY_AVAILABLE else invalid).append(model)
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
    PROFILES_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    profile_path.write_text(json.dumps(config, indent=2))
    profile_path.chmod(0o600)
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
        ("ZAI (GLM)", "ZAI_API_KEY", "zai/"),
        ("Moonshot (Kimi)", "MOONSHOT_API_KEY", "moonshot/"),
        ("MiniMax", "MINIMAX_API_KEY", "minimax/"),
    ]

    print("Supported providers:\n")
    for name, key, prefix in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        models = _discover_models_for_provider(prefix)
        print(f"  {name:12} {key:24} {status}")
        print(f"             Models: {models}")
        print()

    # Azure AI Foundry (uses azure-ai-inference SDK, not litellm)
    foundry_status = "[set]" if os.environ.get("AZURE_AI_API_KEY") else "[not set]"
    print(f"  {'Azure AI':12} {'AZURE_AI_API_KEY':24} {foundry_status}")
    print(f"             Models: foundry/<your-deployment-name> (uses azure-ai-inference SDK)")
    if os.environ.get("AZURE_AI_API_BASE"):
        print(f"             Endpoint: {os.environ['AZURE_AI_API_BASE']}")
    print()

    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    auth_mode = codex_auth_mode()
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    if auth_mode:
        print(f"             Auth mode: {auth_mode}")
    print("             Example models: codex/gpt-5.6-sol, codex/gpt-5.6-terra, codex/gpt-5.5")
    print("             Note: ChatGPT-account auth serves only the ChatGPT lineup (gpt-5.6-sol/terra/luna,")
    print("                   gpt-5.5). gpt-5.3-codex and gpt-5.5-pro need API-key auth or OPENAI_API_KEY.")
    print()

    agy_status = "[installed]" if ANTIGRAVITY_AVAILABLE else "[not installed]"
    print(f"  {'Antigravity':12} {'(Google account)':24} {agy_status}")
    print("             Example models: antigravity/gemini-3.6-flash-high, antigravity/gemini-3.1-pro-high,")
    print("             antigravity/claude-sonnet-4-6, antigravity/gpt-oss-120b-medium (`agy models` lists all)")
    print("             Install: curl -fsSL https://antigravity.google/cli/install.sh | bash")
    print("             Auth: run `agy` once interactively (Google sign-in), then headless works")
    print()

    gemini_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(RETIRED 2026-06-18)':24} {gemini_status}")
    print("             Consumer service ended; enterprise licenses only. Use antigravity/ or gemini/ instead.")
    print()
