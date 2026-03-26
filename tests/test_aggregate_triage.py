"""Tests for aggregate_triage() in verify.py."""

from verify import aggregate_triage


class TestAggregateTriage:
    def test_all_confident_passes(self, sample_claims, make_triage_response):
        """When all models say CONFIDENT, claim is verified."""
        r1 = make_triage_response("model-a", {
            "1": {"verdict": "CONFIDENT", "reason": "ok"},
            "2": {"verdict": "CONFIDENT", "reason": "ok"},
            "3": {"verdict": "CONFIDENT", "reason": "ok"},
        })
        r2 = make_triage_response("model-b", {
            "1": {"verdict": "CONFIDENT", "reason": "ok"},
            "2": {"verdict": "CONFIDENT", "reason": "ok"},
            "3": {"verdict": "CONFIDENT", "reason": "ok"},
        })
        flagged, verified = aggregate_triage([r1, r2], sample_claims)
        assert len(verified) == 3
        assert len(flagged) == 0

    def test_any_uncertain_flags(self, sample_claims, make_triage_response):
        """If any model says UNCERTAIN, the claim gets flagged."""
        r1 = make_triage_response("model-a", {
            "1": {"verdict": "CONFIDENT", "reason": "ok"},
            "2": {"verdict": "CONFIDENT", "reason": "ok"},
            "3": {"verdict": "CONFIDENT", "reason": "ok"},
        })
        r2 = make_triage_response("model-b", {
            "1": {"verdict": "UNCERTAIN", "reason": "pricing changes"},
            "2": {"verdict": "CONFIDENT", "reason": "ok"},
            "3": {"verdict": "CONFIDENT", "reason": "ok"},
        })
        flagged, verified = aggregate_triage([r1, r2], sample_claims)
        assert len(flagged) == 1
        assert flagged[0]["id"] == "1"
        assert len(verified) == 2

    def test_suspect_flags(self, sample_claims, make_triage_response):
        r1 = make_triage_response("model-a", {
            "1": {"verdict": "SUSPECT", "reason": "wrong price"},
        })
        flagged, _ = aggregate_triage([r1], sample_claims)
        flagged_ids = [c["id"] for c in flagged]
        assert "1" in flagged_ids

    def test_flagged_includes_reasons(self, sample_claims, make_triage_response):
        r1 = make_triage_response("model-a", {
            "1": {"verdict": "UNCERTAIN", "reason": "check pricing"},
        })
        flagged, _ = aggregate_triage([r1], sample_claims)
        reasons = flagged[0]["triage_reasons"]
        assert any("model-a" in r for r in reasons)
        assert any("check pricing" in r for r in reasons)

    def test_errored_model_skipped(self, sample_claims, make_triage_response):
        """A model with an error should not affect consensus."""
        r_good = make_triage_response("model-a", {
            "1": {"verdict": "CONFIDENT", "reason": "ok"},
            "2": {"verdict": "CONFIDENT", "reason": "ok"},
            "3": {"verdict": "CONFIDENT", "reason": "ok"},
        })
        r_error = make_triage_response("model-b", {}, error="API timeout")
        flagged, verified = aggregate_triage([r_good, r_error], sample_claims)
        assert len(verified) == 3
        assert len(flagged) == 0

    def test_all_models_errored_flags_everything(self, sample_claims, make_triage_response):
        """If all models error out, every claim should be flagged."""
        r1 = make_triage_response("model-a", {}, error="timeout")
        r2 = make_triage_response("model-b", {}, error="rate limit")
        flagged, verified = aggregate_triage([r1, r2], sample_claims)
        assert len(flagged) == 3
        assert len(verified) == 0

    def test_missing_claim_in_verdicts_flags_it(self, sample_claims, make_triage_response):
        """If a model doesn't return a verdict for a claim, it defaults to UNCERTAIN."""
        r1 = make_triage_response("model-a", {
            "1": {"verdict": "CONFIDENT", "reason": "ok"},
            # claims 2, 3 missing
        })
        flagged, verified = aggregate_triage([r1], sample_claims)
        flagged_ids = [c["id"] for c in flagged]
        assert "2" in flagged_ids
        assert "3" in flagged_ids

    def test_empty_responses_list(self, sample_claims):
        """No triage responses = all claims flagged."""
        flagged, verified = aggregate_triage([], sample_claims)
        assert len(flagged) == 3
        assert len(verified) == 0
