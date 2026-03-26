"""Tests for domain detection and MCP matching in sources.py."""

from sources import (
    build_source_plan,
    detect_document_domains,
    find_relevant_mcps,
    load_registry,
    DEFAULT_REGISTRY,
    DOMAIN_KEYWORDS,
)


class TestDetectDocumentDomains:
    def test_azure_document(self):
        content = "Microsoft Azure Sentinel provides SIEM capabilities with Entra ID integration."
        domains = detect_document_domains(content)
        assert "microsoft" in domains

    def test_aws_document(self):
        content = "AWS Lambda functions triggered by S3 events via SQS queues."
        domains = detect_document_domains(content)
        assert "aws" in domains

    def test_gcp_document(self):
        content = "Google Cloud BigQuery integrates with Vertex AI for ML pipelines."
        domains = detect_document_domains(content)
        assert "gcp" in domains

    def test_mixed_domains_ordered_by_score(self):
        content = (
            "Azure Azure Azure Entra Purview "  # 5 microsoft keywords
            "AWS S3 "  # 2 aws keywords
        )
        domains = detect_document_domains(content)
        assert domains[0] == "microsoft"
        assert "aws" in domains

    def test_no_domains_detected(self):
        content = "This is a cooking recipe for banana bread."
        assert detect_document_domains(content) == []

    def test_case_insensitive(self):
        content = "AZURE SENTINEL and entra id"
        domains = detect_document_domains(content)
        assert "microsoft" in domains

    def test_general_tech_detected(self):
        content = "The REST API uses OAuth2 and SAML for authentication."
        domains = detect_document_domains(content)
        assert "general_tech" in domains


class TestFindRelevantMcps:
    def test_microsoft_domain_finds_ms_docs(self):
        available, recommended = find_relevant_mcps(
            domains=["microsoft"],
            connected_mcps=["microsoft_docs_mcp"],
            registry=DEFAULT_REGISTRY,
        )
        assert any(m["name"] == "microsoft_docs_mcp" for m in available)
        assert not any(m["name"] == "microsoft_docs_mcp" for m in recommended)

    def test_disconnected_mcp_is_recommended(self):
        available, recommended = find_relevant_mcps(
            domains=["microsoft"],
            connected_mcps=[],  # nothing connected
            registry=DEFAULT_REGISTRY,
        )
        assert any(m["name"] == "microsoft_docs_mcp" for m in recommended)
        assert len(available) == 0

    def test_irrelevant_domain_finds_nothing(self):
        available, recommended = find_relevant_mcps(
            domains=["cooking"],
            connected_mcps=["microsoft_docs_mcp"],
            registry=DEFAULT_REGISTRY,
        )
        assert len(available) == 0
        assert len(recommended) == 0

    def test_library_domain_finds_context7(self):
        available, recommended = find_relevant_mcps(
            domains=["library"],
            connected_mcps=["context7"],
            registry=DEFAULT_REGISTRY,
        )
        assert any(m["name"] == "context7" for m in available)

    def test_partial_name_match(self):
        """Connected MCP name can be a partial match."""
        available, _ = find_relevant_mcps(
            domains=["microsoft"],
            connected_mcps=["plugin_microsoft_docs_mcp_server"],
            registry=DEFAULT_REGISTRY,
        )
        assert any(m["name"] == "microsoft_docs_mcp" for m in available)


class TestBuildSourcePlan:
    def test_microsoft_claim_gets_mcp_source(self):
        claims = [{"id": "1", "text": "Azure Sentinel costs $2.46/GB"}]
        available_mcps = [{
            "name": "microsoft_docs_mcp",
            "tools": {"search": "mcp__microsoft_docs_mcp__microsoft_docs_search"},
            "description": "MS docs",
            "search_instruction": "",
        }]
        plan = build_source_plan(claims, available_mcps, ["microsoft"])
        sources = plan["1"]
        assert "mcp__microsoft_docs_mcp__microsoft_docs_search" in sources
        assert "web_search" in sources
        assert "model_knowledge" in sources

    def test_unrelated_claim_gets_only_fallbacks(self):
        claims = [{"id": "1", "text": "Banana bread takes 60 minutes to bake"}]
        available_mcps = [{
            "name": "microsoft_docs_mcp",
            "tools": {"search": "mcp__microsoft_docs_mcp__microsoft_docs_search"},
            "description": "MS docs",
            "search_instruction": "",
        }]
        plan = build_source_plan(claims, available_mcps, ["microsoft"])
        sources = plan["1"]
        assert sources == ["web_search", "model_knowledge"]

    def test_source_order_mcp_before_web(self):
        claims = [{"id": "1", "text": "Azure functions support Python 3.12"}]
        available_mcps = [{
            "name": "microsoft_docs_mcp",
            "tools": {"search": "mcp__ms_search"},
            "description": "MS docs",
            "search_instruction": "",
        }]
        plan = build_source_plan(claims, available_mcps, ["microsoft"])
        sources = plan["1"]
        mcp_idx = sources.index("mcp__ms_search")
        web_idx = sources.index("web_search")
        assert mcp_idx < web_idx
