"""Tests for parse_triage_response() in models.py."""

from models import parse_triage_response


class TestParseTriageResponse:
    def test_basic_parse(self, sample_triage_output):
        verdicts = parse_triage_response(sample_triage_output)
        assert len(verdicts) == 2
        assert verdicts["1"]["verdict"] == "UNCERTAIN"
        assert verdicts["2"]["verdict"] == "CONFIDENT"

    def test_reasons_captured(self, sample_triage_output):
        verdicts = parse_triage_response(sample_triage_output)
        assert "frequently" in verdicts["1"]["reason"]
        assert "well-established" in verdicts["2"]["reason"]

    def test_empty_input(self):
        assert parse_triage_response("") == {}

    def test_no_blocks(self):
        assert parse_triage_response("Just some random text.") == {}

    def test_missing_id_skipped(self):
        text = "[TRIAGE]\nverdict: CONFIDENT\nreason: sure\n[/TRIAGE]"
        assert parse_triage_response(text) == {}

    def test_missing_verdict_skipped(self):
        text = "[TRIAGE]\nid: 1\nreason: hmm\n[/TRIAGE]"
        assert parse_triage_response(text) == {}

    def test_verdict_uppercased(self):
        text = "[TRIAGE]\nid: 1\nverdict: suspect\nreason: wrong\n[/TRIAGE]"
        verdicts = parse_triage_response(text)
        assert verdicts["1"]["verdict"] == "SUSPECT"

    def test_empty_reason(self):
        text = "[TRIAGE]\nid: 1\nverdict: CONFIDENT\nreason:\n[/TRIAGE]"
        verdicts = parse_triage_response(text)
        assert verdicts["1"]["reason"] == ""
