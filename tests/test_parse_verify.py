"""Tests for parse_verify_output() in verify.py."""

from verify import parse_verify_output


class TestParseVerifyOutput:
    def test_basic_extraction(self, sample_verify_output):
        results = parse_verify_output(sample_verify_output)
        assert len(results) == 2

    def test_verdict_uppercased(self, sample_verify_output):
        results = parse_verify_output(sample_verify_output)
        assert results[0]["verdict"] == "NUANCED"
        assert results[1]["verdict"] == "CONFIRMED"

    def test_multiline_explanation(self, sample_verify_output):
        results = parse_verify_output(sample_verify_output)
        explanation = results[0]["explanation"]
        assert "pay-as-you-go" in explanation
        assert "commitment tier" in explanation

    def test_multiline_suggested_fix(self, sample_verify_output):
        results = parse_verify_output(sample_verify_output)
        fix = results[0]["suggested_fix"]
        assert "pay-as-you-go" in fix
        assert "commitment tiers" in fix

    def test_empty_input(self):
        assert parse_verify_output("") == []

    def test_missing_verdict_skipped(self):
        text = "[VERIFY]\nid: 1\nsource: http://x\n[/VERIFY]"
        assert parse_verify_output(text) == []

    def test_missing_id_skipped(self):
        text = "[VERIFY]\nverdict: CONFIRMED\nsource: http://x\n[/VERIFY]"
        assert parse_verify_output(text) == []

    def test_minimal_valid(self):
        text = "[VERIFY]\nid: 5\nverdict: incorrect\n[/VERIFY]"
        results = parse_verify_output(text)
        assert len(results) == 1
        assert results[0]["verdict"] == "INCORRECT"

    def test_multiline_quote(self):
        text = (
            "[VERIFY]\n"
            "id: 1\n"
            "verdict: confirmed\n"
            "source: https://docs.example.com\n"
            "quote: The feature supports both\n"
            "FIDO2 and passkey authentication.\n"
            "explanation: Confirmed.\n"
            "suggested_fix: N/A\n"
            "[/VERIFY]"
        )
        results = parse_verify_output(text)
        assert "FIDO2 and passkey" in results[0]["quote"]

    def test_source_url_preserved(self, sample_verify_output):
        results = parse_verify_output(sample_verify_output)
        assert results[0]["source"] == "https://learn.microsoft.com/pricing"
