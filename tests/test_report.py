"""Tests for generate_report() in verify.py."""

from models import CostTracker
from verify import generate_report


class TestGenerateReport:
    def _make_report(self, claims=None, verified=None, results=None, tracker=None):
        return generate_report(
            title="Test Report",
            source_path="test.md",
            triage_models=["model-a", "model-b"],
            sources_used=["web_search"],
            claims=claims or [],
            verified_claims=verified or [],
            verification_results=results or [],
            tracker=tracker or CostTracker(),
        )

    def test_basic_structure(self):
        report = self._make_report()
        assert "# Fact-Check Report: Test Report" in report
        assert "test.md" in report
        assert "model-a, model-b" in report

    def test_verdict_counts(self):
        results = [
            {"id": "1", "verdict": "CONFIRMED"},
            {"id": "2", "verdict": "INCORRECT"},
            {"id": "3", "verdict": "NUANCED"},
        ]
        claims = [
            {"id": "1", "text": "Claim 1", "category": "pricing"},
            {"id": "2", "text": "Claim 2", "category": "pricing"},
            {"id": "3", "text": "Claim 3", "category": "pricing"},
        ]
        report = self._make_report(claims=claims, results=results)
        assert "**1** incorrect" in report
        assert "**1** nuanced" in report

    def test_high_priority_section_with_incorrect(self):
        results = [{"id": "1", "verdict": "INCORRECT", "source": "https://x", "suggested_fix": "fix it"}]
        claims = [{"id": "1", "text": "Wrong claim", "category": "pricing"}]
        report = self._make_report(claims=claims, results=results)
        assert "HIGH PRIORITY" in report
        assert "INCORRECT" in report

    def test_no_high_priority_shows_clean(self):
        report = self._make_report()
        assert "No incorrect or outdated claims found" in report

    def test_model_verified_section(self):
        verified = [{"id": "1", "text": "Verified claim", "category": "capability"}]
        report = self._make_report(verified=verified)
        assert "Model-verified" in report
        assert "Verified claim" in report

    def test_cost_in_report(self):
        tracker = CostTracker()
        tracker.add("codex/latest", 10000, 5000)
        report = self._make_report(tracker=tracker)
        assert "$0.0000" in report

    def test_sources_used_displayed(self):
        report = self._make_report()
        assert "web_search" in report

    def test_no_sources_shows_na(self):
        report = generate_report(
            title="T",
            source_path="t.md",
            triage_models=[],
            sources_used=[],
            claims=[],
            verified_claims=[],
            verification_results=[],
            tracker=CostTracker(),
        )
        assert "N/A" in report
