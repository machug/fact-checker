"""Main orchestrator and CLI for fact-checker skill."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import CostTracker, cost_tracker, triage_claims_parallel
from prompts import (
    EXTRACT_CLAIMS_SYSTEM,
    EXTRACT_CLAIMS_USER,
    REPORT_TEMPLATE,
    TRIAGE_SYSTEM,
    TRIAGE_USER,
)
from providers import (
    get_available_providers,
    list_profiles,
    list_providers,
    load_profile,
    save_profile,
    validate_model_credentials,
)
from sources import (
    build_source_plan,
    detect_document_domains,
    find_relevant_mcps,
    list_registry,
    load_registry,
    save_registry_addition,
)


def parse_claims_output(text: str) -> list[dict]:
    """Parse [CLAIM] blocks from extraction output."""
    claims = []
    blocks = re.findall(r"\[CLAIM\](.*?)\[/CLAIM\]", text, re.DOTALL)

    for block in blocks:
        claim: dict[str, str] = {}
        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("id:"):
                claim["id"] = line[3:].strip()
            elif line.startswith("text:"):
                claim["text"] = line[5:].strip()
            elif line.startswith("category:"):
                claim["category"] = line[9:].strip()
            elif line.startswith("section:"):
                claim["section"] = line[8:].strip()

        if claim.get("id") and claim.get("text"):
            claims.append(claim)

    return claims


def parse_verify_output(text: str) -> list[dict]:
    """Parse [VERIFY] blocks from verification output."""
    results = []
    blocks = re.findall(r"\[VERIFY\](.*?)\[/VERIFY\]", text, re.DOTALL)

    for block in blocks:
        result: dict[str, str] = {}
        current_key = None

        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("id:"):
                result["id"] = line[3:].strip()
                current_key = "id"
            elif line.startswith("verdict:"):
                result["verdict"] = line[8:].strip().upper()
                current_key = "verdict"
            elif line.startswith("source:"):
                result["source"] = line[7:].strip()
                current_key = "source"
            elif line.startswith("quote:"):
                result["quote"] = line[6:].strip()
                current_key = "quote"
            elif line.startswith("explanation:"):
                result["explanation"] = line[12:].strip()
                current_key = "explanation"
            elif line.startswith("suggested_fix:"):
                result["suggested_fix"] = line[14:].strip()
                current_key = "suggested_fix"
            elif current_key and current_key in ("explanation", "suggested_fix", "quote"):
                result[current_key] = result.get(current_key, "") + " " + line

        if result.get("id") and result.get("verdict"):
            results.append(result)

    return results


def aggregate_triage(
    triage_responses: list,
    claims: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Aggregate triage verdicts across models.

    Returns (flagged_claims, model_verified_claims).
    """
    flagged = []
    verified = []

    for claim in claims:
        claim_id = str(claim["id"])
        all_confident = True
        reasons = []

        for response in triage_responses:
            if response.error:
                continue
            verdict_data = response.verdicts.get(claim_id, {})
            verdict = verdict_data.get("verdict", "UNCERTAIN")

            if verdict != "CONFIDENT":
                all_confident = False
                reasons.append(
                    f"{response.model}: {verdict} — {verdict_data.get('reason', 'no reason given')}"
                )

        if all_confident and any(r.verdicts.get(claim_id) for r in triage_responses if not r.error):
            verified.append(claim)
        else:
            claim_with_reasons = dict(claim)
            claim_with_reasons["triage_reasons"] = reasons
            flagged.append(claim_with_reasons)

    return flagged, verified


def generate_report(
    title: str,
    source_path: str,
    triage_models: list[str],
    sources_used: list[str],
    claims: list[dict],
    verified_claims: list[dict],
    verification_results: list[dict],
    tracker: CostTracker,
) -> str:
    """Generate the fact-check report markdown."""
    # Count verdicts
    confirmed = sum(1 for r in verification_results if r.get("verdict") == "CONFIRMED")
    nuanced = sum(1 for r in verification_results if r.get("verdict") == "NUANCED")
    incorrect = sum(1 for r in verification_results if r.get("verdict") == "INCORRECT")
    outdated = sum(1 for r in verification_results if r.get("verdict") == "OUTDATED")
    unconfirmed = sum(1 for r in verification_results if r.get("verdict") == "UNCONFIRMED")

    # Add model-verified to confirmed count for summary
    total_confirmed = confirmed + len(verified_claims)

    # Build HIGH priority section (INCORRECT / OUTDATED)
    high_items = [r for r in verification_results if r.get("verdict") in ("INCORRECT", "OUTDATED")]
    if high_items:
        high_lines = ["## HIGH PRIORITY (Incorrect / Outdated)\n"]
        high_lines.append("| # | Claim | Verdict | Source | Suggested Fix |")
        high_lines.append("|---|-------|---------|--------|---------------|")
        for r in high_items:
            claim = next((c for c in claims if str(c["id"]) == str(r["id"])), {})
            high_lines.append(
                f"| {r['id']} | {claim.get('text', 'N/A')[:80]} | **{r['verdict']}** | {r.get('source', 'N/A')} | {r.get('suggested_fix', 'N/A')} |"
            )
        high_section = "\n".join(high_lines)
    else:
        high_section = "## HIGH PRIORITY\n\nNo incorrect or outdated claims found."

    # Build MEDIUM priority section (NUANCED)
    med_items = [r for r in verification_results if r.get("verdict") == "NUANCED"]
    if med_items:
        med_lines = ["## MEDIUM PRIORITY (Nuanced)\n"]
        med_lines.append("| # | Claim | Verdict | Source | Suggested Fix |")
        med_lines.append("|---|-------|---------|--------|---------------|")
        for r in med_items:
            claim = next((c for c in claims if str(c["id"]) == str(r["id"])), {})
            med_lines.append(
                f"| {r['id']} | {claim.get('text', 'N/A')[:80]} | **{r['verdict']}** | {r.get('source', 'N/A')} | {r.get('suggested_fix', 'N/A')} |"
            )
        med_section = "\n".join(med_lines)
    else:
        med_section = "## MEDIUM PRIORITY\n\nNo nuanced claims found."

    # Build LOW priority section (UNCONFIRMED)
    low_items = [r for r in verification_results if r.get("verdict") == "UNCONFIRMED"]
    if low_items:
        low_lines = ["## LOW PRIORITY (Unconfirmed)\n"]
        low_lines.append("| # | Claim | Verdict | Notes |")
        low_lines.append("|---|-------|---------|-------|")
        for r in low_items:
            claim = next((c for c in claims if str(c["id"]) == str(r["id"])), {})
            low_lines.append(
                f"| {r['id']} | {claim.get('text', 'N/A')[:80]} | {r['verdict']} | {r.get('explanation', 'N/A')[:100]} |"
            )
        low_section = "\n".join(low_lines)
    else:
        low_section = "## LOW PRIORITY\n\nNo unconfirmed claims."

    # Build verified section
    confirmed_items = [r for r in verification_results if r.get("verdict") == "CONFIRMED"]
    ver_lines = ["## VERIFIED (No Action Required)\n"]
    if verified_claims:
        ver_lines.append(f"### Model-verified ({len(verified_claims)} claims)\n")
        for c in verified_claims:
            ver_lines.append(f"- [{c.get('category', 'N/A')}] {c.get('text', 'N/A')[:100]}")
    if confirmed_items:
        ver_lines.append(f"\n### Source-verified ({len(confirmed_items)} claims)\n")
        for r in confirmed_items:
            claim = next((c for c in claims if str(c["id"]) == str(r["id"])), {})
            ver_lines.append(
                f"- [{claim.get('category', 'N/A')}] {claim.get('text', 'N/A')[:100]} (Source: {r.get('source', 'N/A')})"
            )
    verified_section = "\n".join(ver_lines)

    # MCP recommendations
    mcp_section = ""

    report = REPORT_TEMPLATE.format(
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_path=source_path,
        triage_models=", ".join(triage_models),
        sources_used=", ".join(sources_used) if sources_used else "N/A",
        total_claims=len(claims) + len(verified_claims),
        confirmed=total_confirmed,
        nuanced=nuanced,
        incorrect=incorrect,
        outdated=outdated,
        unconfirmed=unconfirmed,
        model_verified=len(verified_claims),
        deep_verified=len(verification_results),
        total_cost=tracker.total_cost,
        cost_breakdown=tracker.breakdown_str(),
        high_priority_section=high_section,
        medium_priority_section=med_section,
        low_priority_section=low_section,
        verified_section=verified_section,
        mcp_recommendations=mcp_section,
    )

    return report


def cmd_extract(args):
    """Extract claims from a document (extraction phase only)."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text()
    prompt = EXTRACT_CLAIMS_USER.format(document=content)

    print(f"Document: {file_path}")
    print(f"Length: {len(content):,} characters")
    print()
    print("=== EXTRACTION PROMPT ===")
    print(f"System prompt ({len(EXTRACT_CLAIMS_SYSTEM)} chars)")
    print(f"User prompt ({len(prompt)} chars)")
    print()
    print("Send this to Claude to extract claims. The output will contain [CLAIM]...[/CLAIM] blocks.")
    print()
    print("--- System Prompt ---")
    print(EXTRACT_CLAIMS_SYSTEM)
    print()
    print("--- User Prompt ---")
    print(prompt)


def cmd_triage(args):
    """Run triage on extracted claims."""
    models = args.models.split(",")
    valid, invalid = validate_model_credentials(models)

    if invalid:
        print(f"Error: Missing credentials for: {', '.join(invalid)}", file=sys.stderr)
        print("Run: python3 verify.py providers", file=sys.stderr)
        sys.exit(1)

    # Read claims from JSON file
    claims_path = Path(args.claims)
    if not claims_path.exists():
        print(f"Error: Claims file not found: {claims_path}", file=sys.stderr)
        sys.exit(1)

    claims = json.loads(claims_path.read_text())
    document = Path(args.document).read_text() if args.document else ""

    # Format claims for prompt
    claims_text = "\n".join(
        f"[{c['id']}] ({c.get('category', 'N/A')}) {c['text']}"
        for c in claims
    )

    user_message = TRIAGE_USER.format(claims=claims_text, document=document)

    print(f"Triaging {len(claims)} claims with {len(valid)} models: {', '.join(valid)}")
    print()

    responses = triage_claims_parallel(valid, TRIAGE_SYSTEM, user_message)

    # Print results
    for resp in responses:
        if resp.error:
            print(f"\n{resp.model}: ERROR — {resp.error}")
            continue

        print(f"\n{resp.model}: {len(resp.verdicts)} verdicts")
        for cid, v in sorted(resp.verdicts.items()):
            icon = {"CONFIDENT": "✓", "UNCERTAIN": "?", "SUSPECT": "✗"}.get(v["verdict"], "?")
            print(f"  {icon} [{cid}] {v['verdict']}: {v['reason']}")

    print(cost_tracker.summary())

    # Aggregate and output
    flagged, verified = aggregate_triage(responses, claims)

    print(f"\n=== Triage Summary ===")
    print(f"Model-verified (all CONFIDENT): {len(verified)}")
    print(f"Flagged for deep verification: {len(flagged)}")

    # Save flagged claims for next phase
    output_path = claims_path.with_suffix(".flagged.json")
    output_path.write_text(json.dumps(flagged, indent=2))
    print(f"\nFlagged claims saved to: {output_path}")


def cmd_providers(args):
    """List available providers."""
    list_providers()


def cmd_sources(args):
    """List MCP source registry."""
    list_registry()


def cmd_profiles(args):
    """List or manage profiles."""
    if args.subcommand == "list":
        list_profiles()
    elif args.subcommand == "save":
        save_profile(args.name, {"models": args.models})


def cmd_registry(args):
    """Manage MCP registry."""
    if args.subcommand == "list" or not args.subcommand:
        list_registry()
    elif args.subcommand == "add":
        config = {
            "domains": args.domains.split(",") if args.domains else [],
            "tools": json.loads(args.tools) if args.tools else {},
            "description": args.description or "",
        }
        save_registry_addition(args.name, config)


def cmd_check(args):
    """Full pipeline — outputs instructions for Claude to execute."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text()
    domains = detect_document_domains(content)

    print(f"Document: {file_path}")
    print(f"Length: {len(content):,} characters")
    print(f"Detected domains: {', '.join(domains)}")

    # Check available providers
    available = get_available_providers()
    if not available:
        print("\nWarning: No LLM providers configured for triage.", file=sys.stderr)
        print("Run: python3 verify.py providers", file=sys.stderr)

    print(f"\nAvailable providers: {', '.join(p[0] for p in available)}")
    print(f"\nReady for fact-check pipeline. Use the /fact-check skill in Claude Code.")


def main():
    parser = argparse.ArgumentParser(
        description="Fact-checker: verify claims in documents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check
    p_check = subparsers.add_parser("check", help="Full pipeline analysis")
    p_check.add_argument("file", help="Document to fact-check")
    p_check.set_defaults(func=cmd_check)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract claims only")
    p_extract.add_argument("file", help="Document to extract claims from")
    p_extract.set_defaults(func=cmd_extract)

    # triage
    p_triage = subparsers.add_parser("triage", help="Triage extracted claims")
    p_triage.add_argument("claims", help="JSON file of extracted claims")
    p_triage.add_argument("--models", required=True, help="Comma-separated model list")
    p_triage.add_argument("--document", help="Original document for context")
    p_triage.set_defaults(func=cmd_triage)

    # providers
    p_providers = subparsers.add_parser("providers", help="List providers")
    p_providers.set_defaults(func=cmd_providers)

    # sources
    p_sources = subparsers.add_parser("sources", help="List MCP registry")
    p_sources.set_defaults(func=cmd_sources)

    # profiles
    p_profiles = subparsers.add_parser("profiles", help="Manage profiles")
    p_profiles.add_argument("subcommand", choices=["list", "save"])
    p_profiles.add_argument("--name", help="Profile name")
    p_profiles.add_argument("--models", help="Model list")
    p_profiles.set_defaults(func=cmd_profiles)

    # registry
    p_registry = subparsers.add_parser("registry", help="Manage MCP registry")
    p_registry.add_argument("subcommand", nargs="?", default="list", choices=["list", "add"])
    p_registry.add_argument("--name", help="MCP name")
    p_registry.add_argument("--domains", help="Comma-separated domains")
    p_registry.add_argument("--tools", help="JSON tools config")
    p_registry.add_argument("--description", help="MCP description")
    p_registry.set_defaults(func=cmd_registry)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
