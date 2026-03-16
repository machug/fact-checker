"""Model calling, cost tracking, and parallel execution for fact-checker."""

from __future__ import annotations

import concurrent.futures
import json
import os
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
    CODEX_AVAILABLE,
    DEFAULT_CODEX_REASONING,
    DEFAULT_COST,
    GEMINI_CLI_AVAILABLE,
    MODEL_COSTS,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


def is_fixed_temperature_model(model: str) -> bool:
    """Check if a model requires default temperature."""
    model_lower = model.lower()
    if model_lower.startswith(("o1", "o3", "o4")) or any(
        f"/{p}" in model_lower for p in ("o1", "o3", "o4")
    ):
        return True
    if "gpt-5" in model_lower:
        return True
    return False


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
        costs = MODEL_COSTS.get(model, DEFAULT_COST)
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
        "codex", "exec", "--json", "--full-auto", "--skip-git-repo-check",
        "--model", actual_model,
        "-c", f'model_reasoning_effort="{reasoning_effort}"',
        full_prompt,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI failed: {result.stderr.strip()}")

    response_text = ""
    input_tokens = 0
    output_tokens = 0

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response_text = item.get("text", "")
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
        except json.JSONDecodeError:
            continue

    if not response_text:
        raise RuntimeError("No agent message in Codex output")
    return response_text, input_tokens, output_tokens


def call_gemini_cli_model(
    system_prompt: str, user_message: str, model: str, timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Gemini CLI. Returns (response_text, input_tokens, output_tokens)."""
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError("Gemini CLI not found. Install: npm install -g @google/gemini-cli")

    actual_model = model.split("/", 1)[1] if "/" in model else model
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"

    cmd = ["gemini", "-m", actual_model, "-y"]
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
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, in_tok, out_tok = call_codex_model(
                    system_prompt, user_message, model, timeout=timeout,
                )
                verdicts = parse_triage_response(content)
                cost = cost_tracker.add(model, in_tok, out_tok)
                return TriageResponse(
                    model=model, response=content, verdicts=verdicts,
                    input_tokens=in_tok, output_tokens=out_tok, cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        return TriageResponse(model=model, response="", verdicts={}, error=last_error)

    # Gemini CLI path
    if model.startswith("gemini-cli/"):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, in_tok, out_tok = call_gemini_cli_model(
                    system_prompt, user_message, model, timeout=timeout,
                )
                verdicts = parse_triage_response(content)
                cost = cost_tracker.add(model, in_tok, out_tok)
                return TriageResponse(
                    model=model, response=content, verdicts=verdicts,
                    input_tokens=in_tok, output_tokens=out_tok, cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        return TriageResponse(model=model, response="", verdicts={}, error=last_error)

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
            if "gpt-5" not in model.lower():
                kwargs["max_tokens"] = 8000
            if not is_fixed_temperature_model(model):
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
