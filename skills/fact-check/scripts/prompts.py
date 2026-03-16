"""Prompt templates for claim extraction, triage, and verification."""

from __future__ import annotations

# --- Claim Extraction ---

EXTRACT_CLAIMS_SYSTEM = """You are a meticulous fact-checker. Your job is to extract every verifiable factual claim from a document.

A verifiable factual claim is a statement that can be confirmed or denied against an authoritative source. Examples:
- Pricing: "$99/user/month", "included in E5 licensing"
- Dates: "GA May 1, 2026", "announced March 9"
- Capabilities: "supports DLP policies for AI prompts", "blocks pasting credit cards into ChatGPT"
- Status: "in preview", "generally available", "deprecated"
- Numbers: "320+ templates", "supports 30+ providers"
- Architecture: "uses Entra Agent ID for identity", "routes through Defender"
- Compliance: "FedRAMP High authorized", "ISO 42001 certified"

Do NOT extract:
- Opinions or recommendations ("we should enable...", "this is the best approach")
- Internal strategy ("position E7 as the governance enablement path")
- Organisational process statements ("training and awareness required")
- Vague qualitative claims ("comprehensive", "robust", "extensive")"""

EXTRACT_CLAIMS_USER = """Extract all verifiable factual claims from this document.

For each claim, output in this exact format:

[CLAIM]
id: <sequential number>
text: <the exact claim as stated in the document>
category: <pricing|capability|date|licensing|compliance|architecture|status|number>
section: <which section of the document this appears in>
[/CLAIM]

Be thorough. Extract every claim that can be verified against an external source.

Document:

{document}"""

# --- Triage ---

TRIAGE_SYSTEM = """You are a fact-checker assessing whether claims about technology products and services are accurate.

For each claim, assess based on your knowledge:
- CONFIDENT: You are certain this is accurate and current
- UNCERTAIN: You're not sure, your knowledge may be outdated, or the claim involves specific details (exact pricing, exact dates, specific feature names) that could easily be wrong
- SUSPECT: You believe this is incorrect, outdated, or misleading

Be especially cautious with:
- Pricing (changes frequently)
- GA dates (often shift)
- Preview vs GA status (evolves rapidly)
- Exact feature names and admin paths (UI changes often)
- Bundling details (licensing is complex and changes)
- Specific numbers (template counts, provider counts)

When in doubt, say UNCERTAIN rather than CONFIDENT. It's better to verify an accurate claim than to miss an error."""

TRIAGE_USER = """Assess the accuracy of each claim below. For each, respond with exactly:

[TRIAGE]
id: <claim id>
verdict: <CONFIDENT|UNCERTAIN|SUSPECT>
reason: <one sentence explaining your assessment, especially for UNCERTAIN/SUSPECT>
[/TRIAGE]

Claims to assess:

{claims}

Context (the full document these claims are from):

{document}"""

# --- Source Verification ---

VERIFY_SYSTEM = """You are a rigorous fact-checker verifying claims against authoritative source material.

Your job is to compare a specific claim against provided source material and render a verdict:

- CONFIRMED: The source material directly supports the claim as stated
- NUANCED: The claim is substantively correct but imprecise, overstated, or missing important qualifications
- INCORRECT: The source material contradicts the claim
- OUTDATED: The claim was once true but the source shows it has changed
- UNCONFIRMED: The source material neither confirms nor denies the claim

For NUANCED, INCORRECT, and OUTDATED verdicts, you MUST provide:
1. What the source actually says (with a direct quote if possible)
2. A suggested fix — the exact replacement text for the document

Be precise. "320+ templates" vs "360+ templates" matters. "All prompts captured" vs "prompts captured when collection policies are configured" matters. These precision issues erode credibility."""

VERIFY_USER = """Verify this claim against the source material provided.

CLAIM: {claim_text}
CATEGORY: {category}
DOCUMENT SECTION: {section}

SOURCE MATERIAL:
{source_material}

Respond in this exact format:

[VERIFY]
id: {claim_id}
verdict: <CONFIRMED|NUANCED|INCORRECT|OUTDATED|UNCONFIRMED>
source: <URL or source identifier>
quote: <relevant quote from source material, or "N/A" if unconfirmed>
explanation: <one paragraph explaining the verdict>
suggested_fix: <exact replacement text if verdict is NUANCED/INCORRECT/OUTDATED, or "N/A" if CONFIRMED/UNCONFIRMED>
[/VERIFY]"""

# --- Report Generation ---

REPORT_TEMPLATE = """# Fact-Check Report: {title}

**Date:** {date}
**Source document:** {source_path}
**Models used (triage):** {triage_models}
**Sources consulted:** {sources_used}

## Summary

- **{total_claims}** claims extracted
- **{confirmed}** confirmed | **{nuanced}** nuanced | **{incorrect}** incorrect | **{outdated}** outdated | **{unconfirmed}** unconfirmed
- Triage: {model_verified} passed model-check, {deep_verified} sent to deep verification
- Cost: ${total_cost:.4f} ({cost_breakdown})

{high_priority_section}

{medium_priority_section}

{low_priority_section}

{verified_section}

{mcp_recommendations}
"""
