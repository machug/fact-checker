"""Tests for parse_claims_output() in verify.py."""

from verify import parse_claims_output


class TestParseClaimsOutput:
    def test_basic_extraction(self, sample_claims_output):
        claims = parse_claims_output(sample_claims_output)
        assert len(claims) == 2
        assert claims[0]["id"] == "1"
        assert claims[0]["text"] == "Azure Sentinel costs $2.46/GB"
        assert claims[0]["category"] == "pricing"
        assert claims[0]["section"] == "Pricing"

    def test_empty_input(self):
        assert parse_claims_output("") == []

    def test_no_claim_blocks(self):
        assert parse_claims_output("Some text with no claims at all.") == []

    def test_missing_id_skipped(self):
        text = "[CLAIM]\ntext: A claim without an ID\ncategory: pricing\n[/CLAIM]"
        assert parse_claims_output(text) == []

    def test_missing_text_skipped(self):
        text = "[CLAIM]\nid: 1\ncategory: pricing\n[/CLAIM]"
        assert parse_claims_output(text) == []

    def test_minimal_valid_claim(self):
        text = "[CLAIM]\nid: 99\ntext: Something factual\n[/CLAIM]"
        claims = parse_claims_output(text)
        assert len(claims) == 1
        assert claims[0]["id"] == "99"
        assert claims[0]["text"] == "Something factual"
        assert "category" not in claims[0]
        assert "section" not in claims[0]

    def test_extra_whitespace_and_blank_lines(self):
        text = (
            "[CLAIM]\n"
            "  id:   42  \n"
            "  text:   Padded claim   \n"
            "  category:  pricing  \n"
            "\n"
            "[/CLAIM]"
        )
        claims = parse_claims_output(text)
        assert len(claims) == 1
        assert claims[0]["id"] == "42"
        assert claims[0]["text"] == "Padded claim"

    def test_multiple_claims_interleaved_with_prose(self):
        text = (
            "Here are the claims:\n\n"
            "[CLAIM]\nid: 1\ntext: First\n[/CLAIM]\n"
            "Some commentary between claims.\n"
            "[CLAIM]\nid: 2\ntext: Second\n[/CLAIM]\n"
            "Final remarks.\n"
        )
        claims = parse_claims_output(text)
        assert len(claims) == 2
        assert claims[0]["text"] == "First"
        assert claims[1]["text"] == "Second"

    def test_claim_with_colon_in_text(self):
        text = "[CLAIM]\nid: 1\ntext: Price: $99/user/month for E5\n[/CLAIM]"
        claims = parse_claims_output(text)
        assert claims[0]["text"] == "Price: $99/user/month for E5"
