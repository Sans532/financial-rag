import src.agents.verifier as verifier_module
from src.agents.graph import route_after_verification
from src.agents.state import initial_state
from src.agents.verifier import finalize_node, verifier_node


def test_route_after_verification_finalizes_when_passed():
    state = initial_state("q", ["AAPL"])
    state["verification_passed"] = True
    state["retry_count"] = 0
    assert route_after_verification(state) == "finalize"


def test_route_after_verification_retries_when_failed_and_budget_remains():
    state = initial_state("q", ["AAPL"])
    state["verification_passed"] = False
    state["retry_count"] = 0
    state["max_retries"] = 2
    assert route_after_verification(state) == "retry"


def test_route_after_verification_finalizes_when_retries_exhausted():
    state = initial_state("q", ["AAPL"])
    state["verification_passed"] = False
    state["retry_count"] = 2
    state["max_retries"] = 2
    assert route_after_verification(state) == "finalize"


def test_finalize_node_passthrough_when_no_flags():
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = "Apple's revenue was $90B."
    state["verification_results"] = [
        {
            "claim": {"text": "revenue was $90B", "metric": "revenue", "value": 9e10,
                       "unit": "USD", "company": "AAPL", "period": "Q1", "source_span": ""},
            "xbrl_value": 9e10,
            "xbrl_tag": "Revenues",
            "status": "confirmed",
            "delta_pct": 0.0,
        }
    ]
    result = finalize_node(state)
    assert result["final_answer"] == "Apple's revenue was $90B."
    assert "Verification notes" not in result["final_answer"]


def test_finalize_node_appends_footer_for_contradicted_claims():
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = "Apple's revenue was $50B."
    state["verification_results"] = [
        {
            "claim": {"text": "revenue was $50B", "metric": "revenue", "value": 5e10,
                       "unit": "USD", "company": "AAPL", "period": "Q1", "source_span": ""},
            "xbrl_value": 9e10,
            "xbrl_tag": "Revenues",
            "status": "contradicted",
            "delta_pct": 44.0,
        }
    ]
    result = finalize_node(state)
    assert "Verification notes" in result["final_answer"]
    assert "conflicts with SEC XBRL data" in result["final_answer"]


def test_finalize_node_appends_footer_for_unverifiable_claims():
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = "Apple's revenue grew."
    state["verification_results"] = [
        {
            "claim": {"text": "revenue grew 5%", "metric": "revenue", "value": 5,
                       "unit": "%", "company": "AAPL", "period": "Q1", "source_span": ""},
            "xbrl_value": None,
            "xbrl_tag": None,
            "status": "unverifiable",
            "delta_pct": None,
        }
    ]
    result = finalize_node(state)
    assert "could not be independently verified" in result["final_answer"]


def test_finalize_node_handles_missing_draft():
    state = initial_state("q", ["AAPL"])
    result = finalize_node(state)
    assert "unable to generate" in result["final_answer"]


def test_verifier_node_no_draft_short_circuits(monkeypatch):
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = ""
    result = verifier_node(state)
    assert result["verification_passed"] is False
    assert result["verification_results"] == []


def test_verifier_node_passes_when_all_claims_confirmed(monkeypatch):
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = "Revenue was $90B."

    fake_claim = {
        "text": "Revenue was $90B", "metric": "revenue", "value": 9e10,
        "unit": "USD", "company": "AAPL", "period": "Q1", "source_span": "",
    }
    fake_result = {
        "claim": fake_claim, "xbrl_value": 9e10, "xbrl_tag": "Revenues",
        "status": "confirmed", "delta_pct": 0.0,
    }

    monkeypatch.setattr(verifier_module, "extract_numeric_claims", lambda text: [fake_claim])
    monkeypatch.setattr(verifier_module, "verify_claims", lambda claims, client: [fake_result])
    monkeypatch.setattr(verifier_module.XbrlClient, "__enter__", lambda self: self)
    monkeypatch.setattr(verifier_module.XbrlClient, "__exit__", lambda self, *a: None)

    result = verifier_node(state)
    assert result["verification_passed"] is True
    assert result["retry_hints"] == []


def test_verifier_node_fails_and_produces_hints_when_contradicted(monkeypatch):
    state = initial_state("q", ["AAPL"])
    state["draft_answer"] = "Revenue was $50B."

    fake_claim = {
        "text": "Revenue was $50B", "metric": "revenue", "value": 5e10,
        "unit": "USD", "company": "AAPL", "period": "Q1", "source_span": "",
    }
    fake_result = {
        "claim": fake_claim, "xbrl_value": 9e10, "xbrl_tag": "Revenues",
        "status": "contradicted", "delta_pct": 44.0,
    }

    monkeypatch.setattr(verifier_module, "extract_numeric_claims", lambda text: [fake_claim])
    monkeypatch.setattr(verifier_module, "verify_claims", lambda claims, client: [fake_result])
    monkeypatch.setattr(verifier_module.XbrlClient, "__enter__", lambda self: self)
    monkeypatch.setattr(verifier_module.XbrlClient, "__exit__", lambda self, *a: None)

    result = verifier_node(state)
    assert result["verification_passed"] is False
    assert len(result["retry_hints"]) == 1
    assert "does not match SEC XBRL data" in result["retry_hints"][0]
