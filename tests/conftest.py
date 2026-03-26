"""Shared fixtures for fact-checker tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

# Add scripts dir to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).parent.parent / "skills" / "fact-check" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_claims():
    """Minimal list of claims for aggregate/source tests."""
    return [
        {"id": "1", "text": "Azure Sentinel costs $2.46/GB", "category": "pricing", "section": "Pricing"},
        {"id": "2", "text": "Entra ID supports FIDO2 keys", "category": "capability", "section": "Auth"},
        {"id": "3", "text": "S3 supports object lock", "category": "capability", "section": "Storage"},
    ]


@pytest.fixture
def sample_claims_output():
    """Raw text with [CLAIM] blocks as produced by Claude."""
    return (
        "Here are the extracted claims:\n\n"
        "[CLAIM]\n"
        "id: 1\n"
        "text: Azure Sentinel costs $2.46/GB\n"
        "category: pricing\n"
        "section: Pricing\n"
        "[/CLAIM]\n\n"
        "[CLAIM]\n"
        "id: 2\n"
        "text: Entra ID supports FIDO2 keys\n"
        "category: capability\n"
        "section: Auth\n"
        "[/CLAIM]\n"
    )


@pytest.fixture
def sample_verify_output():
    """Raw text with [VERIFY] blocks."""
    return (
        "[VERIFY]\n"
        "id: 1\n"
        "verdict: nuanced\n"
        "source: https://learn.microsoft.com/pricing\n"
        "quote: Pay-as-you-go is $2.46/GB for ingestion\n"
        "explanation: The pricing is correct for pay-as-you-go but the claim\n"
        "omits commitment tier discounts which can reduce cost by up to 50%.\n"
        "suggested_fix: Azure Sentinel costs $2.46/GB (pay-as-you-go;\n"
        "commitment tiers available at lower rates)\n"
        "[/VERIFY]\n\n"
        "[VERIFY]\n"
        "id: 2\n"
        "verdict: confirmed\n"
        "source: https://learn.microsoft.com/entra/identity\n"
        "quote: FIDO2 security keys are supported\n"
        "explanation: Entra ID fully supports FIDO2 security keys for passwordless auth.\n"
        "suggested_fix: N/A\n"
        "[/VERIFY]\n"
    )


@pytest.fixture
def sample_triage_output():
    """Raw text with [TRIAGE] blocks."""
    return (
        "[TRIAGE]\n"
        "id: 1\n"
        "verdict: UNCERTAIN\n"
        "reason: Pricing changes frequently, need to verify current rates\n"
        "[/TRIAGE]\n"
        "[TRIAGE]\n"
        "id: 2\n"
        "verdict: CONFIDENT\n"
        "reason: FIDO2 support in Entra ID is well-established\n"
        "[/TRIAGE]\n"
    )


# ---------------------------------------------------------------------------
# TriageResponse helper — avoids importing models.py (which requires litellm)
# ---------------------------------------------------------------------------


@dataclass
class MockTriageResponse:
    """Standalone TriageResponse for tests that don't need litellm."""

    model: str
    response: str = ""
    verdicts: dict = field(default_factory=dict)
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@pytest.fixture
def make_triage_response():
    """Factory fixture: build a MockTriageResponse per model."""

    def _make(model, verdicts, error=None):
        return MockTriageResponse(model=model, verdicts=verdicts, error=error)

    return _make
