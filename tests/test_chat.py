import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pandas as pd

from app.main import app
from app.services.analytics import return_rate_by_category
from app.services.chat_intent import classify_question, match_analytics_capability
from app.services.chat_session import (
    apply_active_filters_to_plan,
    merge_active_filters,
)
from app.services.plan_schema import Filter, QueryPlan

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

CSV_FILES = ["products.csv", "customers.csv", "stores.csv", "orders.csv", "inventory.csv"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_clean_run(client: TestClient) -> str:
    """Upload samples, profile and clean them; returns run_id."""
    open_files = []
    upload_payload = []
    try:
        for filename in CSV_FILES:
            file_obj = open(SAMPLE_DIR / filename, "rb")
            open_files.append(file_obj)
            upload_payload.append(("files", (filename, file_obj, "text/csv")))
        upload_res = client.post("/runs/upload", files=upload_payload)
    finally:
        for f in open_files:
            f.close()
    assert upload_res.status_code == 200, upload_res.text
    run_id = upload_res.json()["run_id"]

    profile_res = client.post(f"/runs/{run_id}/profile")
    assert profile_res.status_code == 200, profile_res.text

    clean_res = client.post(f"/runs/{run_id}/clean")
    assert clean_res.status_code == 200, clean_res.text
    return run_id


def create_session(client: TestClient, run_id: str) -> str:
    res = client.post(f"/runs/{run_id}/chat/sessions")
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def _clean_path(run_id: str, name: str) -> str:
    """Path to a run's clean CSV (mirror of the router's BASE_DATA_DIR layout)."""
    return str(SAMPLE_DIR.parent / "runs" / run_id / "clean" / f"{name}.csv")


def ask(client: TestClient, run_id: str, session_id: str, question: str):
    return client.post(
        f"/runs/{run_id}/chat/sessions/{session_id}/turns",
        json={"question": question},
    )


# ---------------------------------------------------------------------------
# Unit tests: intent classifier and filter helpers
# ---------------------------------------------------------------------------

def test_classify_unanswerable_keywords():
    for q in [
        "What will our competitor's prices be next year?",
        "Will it rain more next quarter?",
        "Forecast revenue for 2027",
        "How is the stock market doing?",
    ]:
        assert classify_question(q, ["orders"]) == "unanswerable", q


def test_classify_ambiguous_cases():
    assert classify_question("compare", ["orders"]) == "ambiguous"          # too short
    assert classify_question("it", ["orders"]) == "ambiguous"               # too short
    assert classify_question("How many were there for those?", ["orders"]) == "ambiguous"  # unresolved pronoun
    # Pronoun IS resolvable when context exists -> answerable
    assert (
        classify_question("How many were there for those?", ["orders"], {"region": "West"})
        == "answerable"
    )
    # Bare comparison with nothing to compare
    assert classify_question("Compare with last year for us", ["orders"]) == "ambiguous"


def test_classify_answerable():
    assert classify_question("Show me orders in the West region", ["orders"]) == "answerable"
    assert classify_question("Which categories had the highest return rate among those?", ["orders"], {"region": "West"}) == "answerable"


def test_merge_active_filters_new_overrides_old():
    merged = merge_active_filters(
        {"region": "West", "store_id": "S01"},
        [Filter(field="region", op="eq", value="East")],
    )
    assert merged == {"region": "East", "store_id": "S01"}


def test_apply_active_filters_skips_existing_fields():
    plan = QueryPlan(
        intent="filter",
        dataset="orders",
        filters=[Filter(field="region", op="eq", value="East")],
    )
    updated = apply_active_filters_to_plan(plan, {"region": "West"})
    # Planner's explicit East filter wins; no duplicate West added
    assert len(updated.filters) == 1
    assert updated.filters[0].value == "East"


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

def test_unanswerable_question_refused():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res = ask(client, run_id, session_id, "What will our competitor's prices be next year?")
        assert res.status_code == 200, res.text

        turn = res.json()
        assert turn["turn_index"] == 1
        assert turn["status"] == "refused"
        assert turn["plan"] is None
        assert turn["evidence"] is None
        assert turn["answer_text"]  # explanation of what's missing


def test_ambiguous_question_needs_clarification():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res = ask(client, run_id, session_id, "compare")
        assert res.status_code == 200, res.text

        turn = res.json()
        assert turn["turn_index"] == 1
        assert turn["status"] == "clarification_needed"
        assert turn["plan"] is None
        assert turn["answer_text"]  # targeted clarifying question


def test_multi_turn_region_carries_forward():
    """Generic-path carry-forward: region=West set in turn 1 must carry into a
    turn 2 that does NOT match an analytics capability (those route to
    analytics.py directly with their own region pre-filter)."""
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        # Turn 1: establish context
        res1 = ask(client, run_id, session_id, "Show me orders in the West region")
        assert res1.status_code == 200, res1.text
        turn1 = res1.json()
        assert turn1["status"] == "ok"
        assert turn1["turn_index"] == 1

        # Turn 1's plan/evidence must show the region filter the question named
        t1_filters = turn1["evidence"]["filters_applied"]
        assert any(
            f["field"] == "region" and f["value"] == "West" and f["op"] == "eq"
            for f in t1_filters
        ), turn1["evidence"]

        # Turn 2: generic-path follow-up (top-products matches no capability),
        # no region mention — West must still be carried into the plan
        res2 = ask(client, run_id, session_id, "Show me the top products by revenue")
        assert res2.status_code == 200, res2.text
        turn2 = res2.json()
        assert turn2["status"] == "ok"
        assert turn2["turn_index"] == 2

        # West must be carried forward even though turn 2 never mentioned it
        t2_filters = turn2["evidence"]["filters_applied"]
        assert any(
            f["field"] == "region" and f["value"] == "West" and f["op"] == "eq"
            for f in t2_filters
        ), turn2["evidence"]

        t2_plan_fields = [f["field"] for f in turn2["plan"]["filters"]]
        assert "region" in t2_plan_fields
        t2_plan_values = [f["value"] for f in turn2["plan"]["filters"]]
        assert "West" in t2_plan_values


def test_plan_rejected_returns_200_not_500(monkeypatch):
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        def bad_planner(question, available_datasets=None, allow_fallback=True):
            bad_plan = QueryPlan(
                intent="filter",
                dataset="orders",
                filters=[Filter(field="ssn", op="eq", value="123-45-6789")],
            )
            return bad_plan, "mock"

        monkeypatch.setattr("app.routers.chat.generate_plan", bad_planner)

        res = ask(client, run_id, session_id, "Show me all customer records")
        assert res.status_code == 200, res.text  # never a 500

        turn = res.json()
        assert turn["status"] == "plan_rejected"
        assert turn["plan"] is not None  # the rejected plan is stored
        assert turn["plan"]["filters"][0]["field"] == "ssn"
        assert turn["evidence"] is None
        assert "unknown_field" in turn["answer_text"] or "rejected" in turn["answer_text"]


def test_get_session_returns_turns_in_order():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        ask(client, run_id, session_id, "Show me orders in the West region")
        ask(client, run_id, session_id, "compare")
        ask(client, run_id, session_id, "What will our competitor's prices be next year?")

        res = client.get(f"/runs/{run_id}/chat/sessions/{session_id}")
        assert res.status_code == 200, res.text

        data = res.json()
        assert data["session_id"] == session_id
        assert data["run_id"] == run_id
        assert data["active_filters"] == {"region": "West"}

        turns = data["turns"]
        assert [t["turn_index"] for t in turns] == [1, 2, 3]
        assert [t["status"] for t in turns] == ["ok", "clarification_needed", "refused"]

        expected_keys = {"turn_index", "status", "question", "plan", "plan_source", "answer_text", "evidence", "created_at"}
        for t in turns:
            assert expected_keys.issubset(set(t.keys())), t.keys()
        # The "ok" turn has full provenance; refusals do not
        assert turns[0]["plan"] is not None
        assert turns[0]["evidence"] is not None
        assert turns[2]["plan"] is None


# ---------------------------------------------------------------------------
# Analytics capability routing
# ---------------------------------------------------------------------------

def test_match_analytics_capability_patterns():
    assert match_analytics_capability("Which category has the highest return rate?") == "return_rate_by_category"
    assert match_analytics_capability("Show me revenue by category") == "revenue_by_category"
    assert match_analytics_capability("Compare store performance") == "store_performance"
    assert match_analytics_capability("What are our stockout items?") == "stockout_and_ageing"
    assert match_analytics_capability("Which stores had low sales?") == "store_performance"
    # No capability -> falls through to generic pipeline
    assert match_analytics_capability("Show me orders in the West region") is None
    assert match_analytics_capability("Forecast revenue for 2027") is None


def test_capability_return_rate_no_context_is_grounded():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res = ask(client, run_id, session_id, "Which category has the highest return rate?")
        assert res.status_code == 200, res.text
        turn = res.json()
        assert turn["status"] == "ok"
        assert turn["plan"]["capability"] == "return_rate_by_category"
        assert turn["plan_source"] == "analytics_capability"

        evidence = turn["evidence"]
        metrics = evidence["metrics"]
        assert metrics, "expected at least one category in metrics"

        # The named category must exist in the computed metrics with the
        # exact return_rate value quoted in the text — not invented text.
        import re as _re
        match = _re.search(r"'([^']+)' has the highest return rate", turn["answer_text"])
        assert match, turn["answer_text"]
        named = match.group(1)
        assert named in metrics
        rate_str = f"{metrics[named]['return_rate']:.1%}"
        assert rate_str in turn["answer_text"]
        # It really is the max
        max_rate = max(m["return_rate"] for m in metrics.values())
        assert metrics[named]["return_rate"] == max_rate


def test_capability_two_turn_west_scoped_return_rate():
    """The corrected brief scenario: turn 2 must use return-rate business
    logic scoped to the carried-over West region, with real numbers."""
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res1 = ask(client, run_id, session_id, "Show me orders in the West region")
        assert res1.status_code == 200, res1.text
        assert res1.json()["status"] == "ok"

        res2 = ask(client, run_id, session_id, "Which categories had the highest return rate among those?")
        assert res2.status_code == 200, res2.text
        turn2 = res2.json()
        assert turn2["status"] == "ok"
        assert turn2["plan"]["capability"] == "return_rate_by_category"
        assert turn2["plan"]["region_filter_applied"] is True

        evidence = turn2["evidence"]
        assert evidence["computed_from"]["dataset_versions"]["orders"] < 300  # pre-filtered

        # Cross-check: recompute return rate directly on the West-filtered
        # orders and compare with what the API returned.
        orders_df = pd.read_csv(_clean_path(run_id, "orders"))
        products_df = pd.read_csv(_clean_path(run_id, "products"))
        expected = return_rate_by_category(
            orders_df[orders_df["region"] == "West"].copy(), products_df
        )
        assert evidence["metrics"] == expected["metrics"]

        # The named category matches the true max under the West scope
        named = next(
            c for c, m in evidence["metrics"].items()
            if m["return_rate"] == max(mm["return_rate"] for mm in evidence["metrics"].values())
        )
        assert named in turn2["answer_text"]


def test_capability_stockout_items():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res = ask(client, run_id, session_id, "What are our stockout items?")
        assert res.status_code == 200, res.text
        turn = res.json()
        assert turn["status"] == "ok"
        assert turn["plan"]["capability"] == "stockout_and_ageing"
        # Inventory has no region column — no region filter to apply
        assert turn["plan"]["region_filter_applied"] is False
        assert "stockout_items" in turn["evidence"]["metrics"]


def test_session_run_id_mismatch_404():
    with TestClient(app) as client:
        run_id = setup_clean_run(client)
        other_run_id = setup_clean_run(client)
        session_id = create_session(client, run_id)

        res = client.get(f"/runs/{other_run_id}/chat/sessions/{session_id}")
        assert res.status_code == 404

        res = ask(client, other_run_id, session_id, "Show me orders")
        assert res.status_code == 404
