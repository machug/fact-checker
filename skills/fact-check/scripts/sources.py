"""MCP-aware source resolution and registry for fact-checker."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path.home() / ".claude" / "fact-checker" / "mcp-registry.json"

# Default MCP registry — maps known MCPs to domains and tool names
DEFAULT_REGISTRY: dict[str, dict] = {
    "microsoft_docs_mcp": {
        "domains": [
            "microsoft", "azure", "m365", "entra", "purview", "defender",
            "copilot", "intune", "teams", "sharepoint", "power platform",
            "dynamics", "office 365", "sentinel", "fabric",
        ],
        "tools": {
            "search": "mcp__microsoft_docs_mcp__microsoft_docs_search",
            "fetch": "mcp__microsoft_docs_mcp__microsoft_docs_fetch",
            "code": "mcp__microsoft_docs_mcp__microsoft_code_sample_search",
        },
        "description": "Official Microsoft Learn documentation",
        "search_instruction": "Use microsoft_docs_search for broad queries, microsoft_docs_fetch for specific URLs",
    },
    "context7": {
        "domains": [
            "library", "framework", "sdk", "package", "npm", "pypi",
            "api reference", "documentation",
        ],
        "tools": {
            "resolve": "mcp__plugin_context7_context7__resolve-library-id",
            "query": "mcp__plugin_context7_context7__query-docs",
        },
        "description": "Library and framework documentation (up-to-date)",
        "search_instruction": "First resolve-library-id, then query-docs with the resolved ID",
    },
    "deepwiki": {
        "domains": [
            "github", "open source", "repository", "open-source",
        ],
        "tools": {
            "read": "mcp__deepwiki__read_wiki_structure",
            "query": "mcp__deepwiki__ask_question",
        },
        "description": "GitHub repository documentation and wikis",
        "search_instruction": "Use read_wiki_structure first, then ask_question for specifics",
    },
    "atlassian": {
        "domains": [
            "jira", "confluence", "atlassian", "bitbucket",
        ],
        "tools": {
            "search": "mcp__atlassian__search",
            "fetch": "mcp__atlassian__fetch",
        },
        "description": "Atlassian Jira and Confluence",
        "search_instruction": "Use search for broad queries, fetch for specific URLs",
    },
}

# Domain keyword detection for document classification
DOMAIN_KEYWORDS = {
    "microsoft": [
        "microsoft", "azure", "m365", "entra", "purview", "defender",
        "copilot", "intune", "teams", "sharepoint", "power platform",
        "sentinel", "fabric", "office 365", "windows", "graph api",
        "conditional access", "dlp", "sensitivity label",
    ],
    "aws": [
        "aws", "amazon", "ec2", "s3", "lambda", "cloudwatch",
        "cloudformation", "iam", "sqs", "sns", "bedrock",
    ],
    "gcp": [
        "google cloud", "gcp", "bigquery", "cloud run", "vertex ai",
        "cloud functions", "gke",
    ],
    "general_tech": [
        "api", "rest", "graphql", "oauth", "saml", "kubernetes",
        "docker", "terraform", "ci/cd",
    ],
}


def load_registry() -> dict[str, dict]:
    """Load MCP registry, merging defaults with user customizations."""
    registry = dict(DEFAULT_REGISTRY)

    if REGISTRY_PATH.exists():
        try:
            custom = json.loads(REGISTRY_PATH.read_text())
            registry.update(custom)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in MCP registry: {e}", file=sys.stderr)

    return registry


def save_registry_addition(name: str, config: dict):
    """Add a custom MCP to the registry."""
    custom = {}
    if REGISTRY_PATH.exists():
        try:
            custom = json.loads(REGISTRY_PATH.read_text())
        except json.JSONDecodeError:
            pass

    custom[name] = config
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(custom, indent=2))
    print(f"Added {name} to MCP registry at {REGISTRY_PATH}")


def detect_document_domains(content: str) -> list[str]:
    """Detect which domains a document covers based on keyword analysis.

    Returns sorted list of domain names by relevance (keyword count).
    """
    content_lower = content.lower()
    domain_scores: dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            domain_scores[domain] = score

    return sorted(domain_scores, key=lambda d: domain_scores[d], reverse=True)


def find_relevant_mcps(
    domains: list[str],
    connected_mcps: list[str],
    registry: Optional[dict] = None,
) -> tuple[list[dict], list[dict]]:
    """Find MCPs relevant to the document's domains.

    Args:
        domains: Document domains from detect_document_domains()
        connected_mcps: List of MCP server names currently connected
        registry: MCP registry (loaded if not provided)

    Returns:
        Tuple of (available_mcps, recommended_mcps) where:
        - available_mcps: relevant MCPs that are connected
        - recommended_mcps: relevant MCPs that are NOT connected (recommendations)
    """
    if registry is None:
        registry = load_registry()

    available = []
    recommended = []

    for mcp_name, mcp_config in registry.items():
        mcp_domains = mcp_config.get("domains", [])
        # Check if any document domain overlaps with MCP domains
        overlap = any(
            any(kw in d for kw in mcp_domains)
            for d in domains
        )
        if not overlap:
            # Also check if any MCP domain keyword appears as a document domain
            overlap = any(d in mcp_domains for d in domains)

        if overlap:
            entry = {
                "name": mcp_name,
                "description": mcp_config.get("description", ""),
                "tools": mcp_config.get("tools", {}),
                "search_instruction": mcp_config.get("search_instruction", ""),
            }

            # Check if this MCP is connected (partial match on name)
            is_connected = any(
                mcp_name.lower() in cm.lower() or cm.lower() in mcp_name.lower()
                for cm in connected_mcps
            )

            if is_connected:
                available.append(entry)
            else:
                recommended.append(entry)

    return available, recommended


def build_source_plan(
    claims: list[dict],
    available_mcps: list[dict],
    domains: list[str],
) -> dict[str, list[str]]:
    """Build a verification source plan mapping claim IDs to ordered source list.

    Returns dict of {claim_id: [source1, source2, ...]} where sources are
    MCP tool names or "web_search" or "model_knowledge".
    """
    plan: dict[str, list[str]] = {}

    for claim in claims:
        claim_id = str(claim.get("id", ""))
        claim_text = claim.get("text", "").lower()
        sources: list[str] = []

        # Match claim to MCPs by checking if claim text relates to MCP domains
        for mcp in available_mcps:
            mcp_name = mcp["name"]
            # Check against registry domains
            registry = load_registry()
            mcp_config = registry.get(mcp_name, {})
            mcp_domains = mcp_config.get("domains", [])

            if any(kw in claim_text for kw in mcp_domains):
                search_tool = mcp.get("tools", {}).get("search")
                if search_tool:
                    sources.append(search_tool)

        # Always add web search as fallback
        sources.append("web_search")
        # Model knowledge as last resort
        sources.append("model_knowledge")

        plan[claim_id] = sources

    return plan


def list_registry():
    """Print the current MCP registry."""
    registry = load_registry()

    print("MCP Source Registry:\n")
    for name, config in sorted(registry.items()):
        domains = ", ".join(config.get("domains", [])[:5])
        if len(config.get("domains", [])) > 5:
            domains += ", ..."
        tools = list(config.get("tools", {}).keys())

        print(f"  {name}")
        print(f"    Description: {config.get('description', 'N/A')}")
        print(f"    Domains: {domains}")
        print(f"    Tools: {', '.join(tools)}")
        print()
