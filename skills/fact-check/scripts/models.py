"""Model calling, cost tracking, and parallel execution for fact-checker."""

from __future__ import annotations

import concurrent.futures
import errno
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from providers import (
    ANTIGRAVITY_AVAILABLE,
    ANTIGRAVITY_PATH,
    CODEX_AVAILABLE,
    CODEX_PATH,
    DEFAULT_CODEX_REASONING,
    GEMINI_CLI_AVAILABLE,
    GEMINI_CLI_PATH,
    get_model_cost,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Error substrings that retrying cannot fix: bad model id, wrong auth mode,
# rejected/revoked credentials. These are deterministic 4xx-class failures —
# retrying just burns time and spams warnings.
NON_RETRYABLE_PATTERNS = (
    "not supported when using codex with a chatgpt account",
    "invalid_request_error",
    "model_not_found",
    "does not exist or you do not have access",
    "authenticationerror",
    "invalid api key",
    "incorrect api key",
    "notfounderror",
    # Azure OpenAI: model routed to a proxy that has no such deployment
    "deployment for this resource does not exist",
    # Antigravity CLI deterministic failures
    "is not authenticated",
    "invalid model selection",
    "exceeds the os argument-size limit",
)

CODEX_CHATGPT_HINT = (
    "Codex is authenticated with a ChatGPT account, which only serves: "
    "gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5 "
    "(gpt-5.4/-mini retire 2026-08-31; gpt-5.3-codex-spark needs ChatGPT Pro). "
    "For other models authenticate Codex with an API key or use the "
    "OPENAI_API_KEY litellm route (e.g. --models gpt-5.5-pro)."
)


def is_non_retryable_error(error_msg: str) -> bool:
    """Whether an error is deterministic (4xx-class) and not worth retrying."""
    lower = error_msg.lower()
    return any(p in lower for p in NON_RETRYABLE_PATTERNS)


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o-series, gpt-5).

    Reasoning models differ from standard models:
    - They ignore the temperature parameter (fixed internally)
    - They use max_completion_tokens instead of max_tokens
    """
    model_lower = model.lower()
    if model_lower.startswith(("o1", "o3", "o4")) or any(
        f"/{p}" in model_lower for p in ("o1", "o3", "o4")
    ):
        return True
    if "gpt-5" in model_lower:
        return True
    # xAI reasoning models: grok-*-reasoning but NOT *-non-reasoning
    if "xai/" in model_lower and model_lower.endswith("-reasoning") and not model_lower.endswith("-non-reasoning"):
        return True
    # Moonshot Kimi reasoning models (kimi-k2.5 and later reject temperature,
    # only allow 1). Anchor on the version segment after "kimi-k" so arbitrary
    # "k3" substrings elsewhere in a model id don't match.
    if "moonshot/" in model_lower:
        m = re.search(r"kimi-k(\d+(?:\.\d+)?)", model_lower)
        if m and float(m.group(1)) >= 2.5:
            return True
    return False


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model uses max_completion_tokens instead of max_tokens.

    Most reasoning models use max_completion_tokens, but some providers
    still use max_tokens (litellm doesn't support max_completion_tokens for them).
    """
    if not is_reasoning_model(model):
        return False
    # xAI and Moonshot use max_tokens even for reasoning models
    if model.lower().startswith(("xai/", "moonshot/")):
        return False
    return True


@dataclass
class TriageResponse:
    """Response from a triage model."""

    model: str
    response: str
    verdicts: dict[str, dict]  # {claim_id: {"verdict": ..., "reason": ...}}
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track token usage and costs across model calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add usage for a model call and return the cost."""
        costs = get_model_cost(model)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (
            output_tokens / 1_000_000 * costs["output"]
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if model not in self.by_model:
            self.by_model[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        self.by_model[model]["input_tokens"] += input_tokens
        self.by_model[model]["output_tokens"] += output_tokens
        self.by_model[model]["cost"] += cost

        return cost

    def summary(self) -> str:
        """Generate cost summary string."""
        lines = ["", "=== Cost Summary ==="]
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
        )
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(
                    f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)"
                )
        return "\n".join(lines)

    def breakdown_str(self) -> str:
        """Short cost breakdown for reports."""
        parts = []
        for model, data in self.by_model.items():
            short_name = model.split("/")[-1] if "/" in model else model
            parts.append(f"{short_name}: ${data['cost']:.4f}")
        return ", ".join(parts) if parts else "N/A"


# Global cost tracker
cost_tracker = CostTracker()


def parse_triage_response(response_text: str) -> dict[str, dict]:
    """Parse [TRIAGE] blocks from model response.

    Returns {claim_id: {"verdict": ..., "reason": ...}}
    """
    import re

    verdicts = {}
    blocks = re.findall(
        r"\[TRIAGE\](.*?)\[/TRIAGE\]", response_text, re.DOTALL
    )

    for block in blocks:
        claim_id = None
        verdict = None
        reason = ""

        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("id:"):
                claim_id = line[3:].strip()
            elif line.startswith("verdict:"):
                verdict = line[8:].strip().upper()
            elif line.startswith("reason:"):
                reason = line[7:].strip()

        if claim_id and verdict:
            verdicts[claim_id] = {"verdict": verdict, "reason": reason}

    return verdicts


def call_foundry_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Azure AI Foundry v2 using the azure-ai-inference SDK.

    Returns (response_text, input_tokens, output_tokens).
    """
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    api_key = os.environ.get("AZURE_AI_API_KEY")
    api_base = os.environ.get("AZURE_AI_API_BASE", "")

    if not api_key:
        raise ValueError("AZURE_AI_API_KEY environment variable not set")

    # Derive the /models endpoint from the base URL
    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/models"):
        parts = endpoint.split(".services.ai.azure.com")
        if len(parts) == 2:
            endpoint = parts[0] + ".services.ai.azure.com/models"
        else:
            endpoint = endpoint + "/models"

    deployment_name = model.split("/", 1)[1] if "/" in model else model

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_message),
        ],
        model=deployment_name,
    )

    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return content, input_tokens, output_tokens


def call_codex_model(
    system_prompt: str, user_message: str, model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Codex CLI. Returns (response_text, input_tokens, output_tokens)."""
    if not CODEX_AVAILABLE:
        raise RuntimeError("Codex CLI not found. Install: npm install -g @openai/codex")

    actual_model = model.split("/", 1)[1] if "/" in model else model
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    cmd = [
        CODEX_PATH, "exec", "--json", "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "--model", actual_model,
        "-c", f'model_reasoning_effort="{reasoning_effort}"',
        full_prompt,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )

    response_text = ""
    input_tokens = 0
    output_tokens = 0
    structured_error = None

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                response_text = item.get("text", "")
        elif etype == "turn.completed":
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        elif etype in ("error", "turn.failed"):
            msg = event.get("message") or event.get("error", {}).get("message")
            if msg:
                structured_error = msg

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex CLI failed: {structured_error or result.stderr.strip() or f'exited with code {result.returncode}'}"
        )

    if not response_text:
        raise RuntimeError(
            f"No agent message in Codex output: {structured_error}" if structured_error
            else "No agent message in Codex output"
        )
    return response_text, input_tokens, output_tokens


def call_gemini_cli_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Gemini CLI. Returns (response_text, input_tokens, output_tokens)."""
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError(
            "Gemini CLI not found. Note: Gemini CLI was retired for consumer "
            "accounts on 2026-06-18 — use antigravity/<model> (agy CLI) or "
            "gemini/<model> (GEMINI_API_KEY) instead."
        )

    print(
        "Warning: Gemini CLI consumer service was retired 2026-06-18 in favor of "
        "Antigravity CLI. If this call fails, switch to antigravity/<model> "
        "(agy CLI) or gemini/<model> (GEMINI_API_KEY).",
        file=sys.stderr,
    )

    actual_model = model.split("/", 1)[1] if "/" in model else model
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    cmd = [GEMINI_CLI_PATH, "-m", actual_model, "-y"]
    result = subprocess.run(cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(f"Gemini CLI failed: {result.stderr.strip()}")

    response_text = result.stdout.strip()
    skip_prefixes = ("Loaded cached", "Server ", "Loading extension")
    lines = [l for l in response_text.split("\n") if not any(l.startswith(p) for p in skip_prefixes)]
    response_text = "\n".join(lines).strip()

    if not response_text:
        raise RuntimeError("No response from Gemini CLI")

    input_tokens = len(full_prompt) // 4
    output_tokens = len(response_text) // 4
    return response_text, input_tokens, output_tokens


def resolve_antigravity_model(model: str) -> Optional[str]:
    """Extract the agy model slug from an antigravity/<slug> model string.

    `agy --model` accepts slugs exactly as listed by `agy models`
    (e.g. gemini-3.1-pro-high, claude-sonnet-4-6, gpt-oss-120b-medium).
    Returns None for a bare "antigravity" (use agy's default model).
    """
    slug = model.split("/", 1)[1] if "/" in model else ""
    return slug or None


def call_antigravity_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Antigravity CLI (agy) in headless print mode using Google account auth.

    Sign in once interactively (`agy`) before headless use — print mode reuses
    cached credentials and cannot complete the OAuth flow itself.
    Token counts come from agy JSON metadata when present, else estimated.
    """
    if not ANTIGRAVITY_AVAILABLE:
        raise RuntimeError(
            "Antigravity CLI not found. Install with: "
            "curl -fsSL https://antigravity.google/cli/install.sh | bash "
            "— then run `agy` once to sign in."
        )

    slug = resolve_antigravity_model(model)
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    cmd = [
        ANTIGRAVITY_PATH,
        "-p",
        full_prompt,
        "--output-format",
        "json",
        "--print-timeout",
        f"{timeout}s",
    ]
    if slug:
        cmd.extend(["--model", slug])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Antigravity CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Antigravity CLI not found in PATH")
    except OSError as e:
        # agy takes the prompt as an argv element (it has no stdin/file prompt
        # channel), so a very large document blows the per-argument OS limit
        # (~128 KiB on Linux) with E2BIG before agy even starts.
        if e.errno == errno.E2BIG:
            raise RuntimeError(
                "Antigravity CLI prompt exceeds the OS argument-size limit "
                f"({len(full_prompt):,} chars). agy cannot read prompts from "
                "stdin — use an API-key provider (e.g. gemini/<model>) for "
                "documents this large."
            )
        raise

    stdout = result.stdout.strip()

    # agy prints an interactive OAuth prompt when credentials are missing —
    # detect its fixed prompt strings, not URL fragments (which could appear
    # in legitimate model output).
    if (
        "Waiting for authentication" in result.stdout
        or "paste the authorization code" in result.stdout
    ):
        raise RuntimeError(
            "Antigravity CLI is not authenticated. Run `agy` interactively once "
            "to complete Google sign-in, then retry."
        )

    if result.returncode != 0:
        error_msg = (
            result.stderr.strip()
            or stdout
            or f"Antigravity CLI exited with code {result.returncode}"
        )
        raise RuntimeError(f"Antigravity CLI failed: {error_msg}")

    response_text = ""
    input_tokens = 0
    output_tokens = 0

    # JSON output is a single object; schema may evolve, so probe common keys.
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        status = payload.get("status", "")
        if status and status != "SUCCESS":
            raise RuntimeError(
                f"Antigravity CLI returned status {status}: "
                f"{payload.get('error') or payload.get('response') or stdout[:200]}"
            )
        for key in ("response", "result", "text", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                response_text = value.strip()
                break
        usage = payload.get("usage") or payload.get("metadata") or {}
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)

    if not response_text and payload is None:
        # Fall back to raw stdout only when it wasn't JSON at all
        # (e.g. --output-format ignored by an older agy)
        response_text = stdout

    if not response_text:
        raise RuntimeError(
            "No response text in Antigravity CLI output: " + stdout[:200]
        )

    if not input_tokens:
        input_tokens = len(full_prompt) // 4
    if not output_tokens:
        output_tokens = len(response_text) // 4

    return response_text, input_tokens, output_tokens


def _call_cli_provider_with_retries(model: str, call_fn) -> TriageResponse:
    """Shared retry loop for CLI/SDK providers (codex, gemini-cli, antigravity,
    foundry).

    Retries transient failures with exponential backoff; deterministic errors
    (unknown model, wrong auth mode, bad credentials) fail fast. Codex
    ChatGPT-account model rejections get an actionable hint appended.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            content, in_tok, out_tok = call_fn()
            verdicts = parse_triage_response(content)
            cost = cost_tracker.add(model, in_tok, out_tok)
            return TriageResponse(
                model=model, response=content, verdicts=verdicts,
                input_tokens=in_tok, output_tokens=out_tok, cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            if "not supported when using codex with a chatgpt account" in last_error.lower():
                last_error = f"{last_error}\n  Hint: {CODEX_CHATGPT_HINT}"
            if is_non_retryable_error(last_error):
                print(
                    f"Error: {model} failed (non-retryable): {last_error}",
                    file=sys.stderr,
                )
                break
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )
    return TriageResponse(model=model, response="", verdicts={}, error=last_error)


def call_single_model_triage(
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: int = 600,
) -> TriageResponse:
    """Send claims to a single model for triage assessment."""
    display_model = model

    # Codex CLI path
    if model.startswith("codex/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_codex_model(system_prompt, user_message, model, timeout=timeout),
        )

    # Antigravity CLI path
    if model == "antigravity" or model.startswith("antigravity/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_antigravity_model(system_prompt, user_message, model, timeout=timeout),
        )

    # Gemini CLI path (retired for consumer accounts 2026-06-18; kept for
    # enterprise-license users, warns and points at antigravity/)
    if model.startswith("gemini-cli/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_gemini_cli_model(system_prompt, user_message, model, timeout=timeout),
        )

    # Azure AI Foundry path
    if model.startswith("foundry/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_foundry_model(system_prompt, user_message, model, timeout=timeout),
        )

    # Standard LiteLLM path
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "timeout": timeout,
            }
            if uses_max_completion_tokens(model):
                kwargs["max_completion_tokens"] = 8000
            else:
                kwargs["max_tokens"] = 8000
            if not is_reasoning_model(model):
                kwargs["temperature"] = 0.3  # Lower temp for factual assessment

            response = completion(**kwargs)
            content = response.choices[0].message.content or ""
            in_tok = response.usage.prompt_tokens if response.usage else 0
            out_tok = response.usage.completion_tokens if response.usage else 0

            verdicts = parse_triage_response(content)
            cost = cost_tracker.add(display_model, in_tok, out_tok)

            return TriageResponse(
                model=display_model, response=content, verdicts=verdicts,
                input_tokens=in_tok, output_tokens=out_tok, cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            if is_non_retryable_error(last_error):
                print(
                    f"Error: {display_model} failed (non-retryable): {last_error}",
                    file=sys.stderr,
                )
                break
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"Warning: {display_model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {display_model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )

    return TriageResponse(
        model=display_model, response="", verdicts={}, error=last_error
    )


def triage_claims_parallel(
    models: list[str],
    system_prompt: str,
    user_message: str,
    timeout: int = 600,
) -> list[TriageResponse]:
    """Call multiple models in parallel for claim triage."""
    if not models:
        return []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(
                call_single_model_triage, model, system_prompt, user_message, timeout,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    return results
