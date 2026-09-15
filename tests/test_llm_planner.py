"""
Tests for llm_planner.py covering:
 - mock path (USE_MOCK_LLM=True)
 - invalid JSON from LLM → ValueError propagates when allow_fallback=False
 - network/API failure → mock_fallback when allow_fallback=True (default)
 - SECURITY: LLM never receives raw row data
"""
import json
import unittest.mock as mock

import pytest

from app.services.llm_planner import _mock_plan, generate_plan
from app.services.plan_schema import QueryPlan


# ---------------------------------------------------------------------------
# Mock planner correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected_intent,expected_dataset", [
    ("What is the revenue by category?",        "aggregate", "orders"),
    ("Show me the top 5 products by revenue",   "top_n",     "orders"),
    ("Which stores have the highest sales?",    "aggregate", "orders"),
    ("Show me products with low stock",         "filter",    "inventory"),
    ("How many customers are in each region?",  "aggregate", "customers"),
    ("What is the average order value?",        "aggregate", "orders"),
    ("Just describe what's in the orders",      "describe",  "orders"),   # fallback
])
def test_mock_plan_keywords(question, expected_intent, expected_dataset):
    plan = _mock_plan(question, ["orders", "products", "customers", "stores", "inventory"])
    assert plan.intent == expected_intent, f"Q={question!r}: expected intent={expected_intent!r}, got {plan.intent!r}"
    assert plan.dataset == expected_dataset, f"Q={question!r}: expected dataset={expected_dataset!r}, got {plan.dataset!r}"
    assert isinstance(plan, QueryPlan)


def test_mock_plan_always_returns_valid_queryplan():
    """Even with an empty/nonsense question, mock returns a valid QueryPlan."""
    plan = _mock_plan("zzzzz", [])
    assert isinstance(plan, QueryPlan)
    assert plan.intent in {"filter", "aggregate", "compare", "top_n", "describe"}


# ---------------------------------------------------------------------------
# generate_plan with USE_MOCK_LLM=True (no API call ever made)
# ---------------------------------------------------------------------------

def test_generate_plan_mock_mode_returns_source_mock(monkeypatch):
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", True)
    plan, source = generate_plan("revenue by category")
    assert source == "mock"
    assert isinstance(plan, QueryPlan)


def test_generate_plan_mock_mode_never_imports_google(monkeypatch):
    """Ensure no google.generativeai import is attempted in mock mode."""
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", True)
    import sys
    # Remove the module if already imported, to force fresh import check
    sys.modules.pop("google.generativeai", None)
    with mock.patch.dict(sys.modules, {"google.generativeai": None}):
        # If llm_planner tried to import it, this would raise ImportError
        plan, source = generate_plan("revenue by category")
    assert source == "mock"


# ---------------------------------------------------------------------------
# Error-handling: invalid JSON response
# ---------------------------------------------------------------------------

def test_invalid_json_raises_valueerror_when_no_fallback(monkeypatch):
    """
    If Gemini returns garbled non-JSON, _gemini_plan raises ValueError.
    With allow_fallback=False, generate_plan re-raises it.
    """
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", False)
    monkeypatch.setattr("app.services.llm_planner.settings.GEMINI_API_KEY", "fake-key")

    def _bad_gemini(question, available_datasets):
        raise ValueError("Gemini returned non-JSON response: <html>Error 503</html>")

    monkeypatch.setattr("app.services.llm_planner._gemini_plan", _bad_gemini)

    with pytest.raises(ValueError, match="non-JSON"):
        generate_plan("revenue by category", allow_fallback=False)


def test_invalid_json_falls_back_to_mock_by_default(monkeypatch):
    """
    With allow_fallback=True (default), any LLM error produces source='mock_fallback'.
    The returned plan is still a valid QueryPlan — the endpoint does NOT crash.
    """
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", False)
    monkeypatch.setattr("app.services.llm_planner.settings.GEMINI_API_KEY", "fake-key")

    def _bad_gemini(question, available_datasets):
        raise ValueError("Gemini returned non-JSON response: <html>Error 503</html>")

    monkeypatch.setattr("app.services.llm_planner._gemini_plan", _bad_gemini)

    plan, source = generate_plan("revenue by category")   # allow_fallback=True by default
    assert source == "mock_fallback"
    assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# Error-handling: network / API call failure
# ---------------------------------------------------------------------------

def test_api_timeout_falls_back_to_mock(monkeypatch):
    """
    A network timeout (simulated with TimeoutError) triggers mock_fallback,
    not a 500 crash.
    """
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", False)
    monkeypatch.setattr("app.services.llm_planner.settings.GEMINI_API_KEY", "fake-key")

    def _timeout_gemini(question, available_datasets):
        raise ValueError("Gemini API call failed: timed out after 15 seconds")

    monkeypatch.setattr("app.services.llm_planner._gemini_plan", _timeout_gemini)

    plan, source = generate_plan("top stores by revenue")
    assert source == "mock_fallback"
    assert isinstance(plan, QueryPlan)


def test_missing_api_key_falls_back_to_mock(monkeypatch):
    """
    A missing GEMINI_API_KEY raises ValueError inside _gemini_plan;
    generate_plan catches it and returns source='mock_fallback'.
    """
    monkeypatch.setattr("app.services.llm_planner.settings.USE_MOCK_LLM", False)
    monkeypatch.setattr("app.services.llm_planner.settings.GEMINI_API_KEY", "")

    plan, source = generate_plan("revenue by category")
    assert source == "mock_fallback"
    assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# SECURITY: LLM never sees raw row data
# ---------------------------------------------------------------------------

def test_llm_receives_only_schema_not_row_data():
    """
    Asserts that the LLM receives ONLY schema descriptions + the question —
    NOT any row-level CSV data.

    Strategy: inspect the module-level _SYSTEM_PROMPT (sent verbatim) and
    the source of _gemini_plan (to verify the user_message construction)
    without making a real network call.
    """
    import re
    import inspect
    import app.services.llm_planner as planner_module

    prompt = planner_module._SYSTEM_PROMPT

    # 1. Schema field names must be present
    assert "order_id" in prompt
    assert "product_name" in prompt
    assert "stock_quantity" in prompt
    assert "signup_date" in prompt

    # 2. No raw numeric CSV data in the prompt
    assert not re.search(r"\d+\.\d+,\s*\d+", prompt), \
        "System prompt must not contain raw numeric CSV-style data"

    # 3. No dataframe operations embedded in the prompt string
    assert "pd.DataFrame" not in prompt
    assert "read_csv" not in prompt

    # 4. Inspect _gemini_plan source: user_message is built from
    #    `available_datasets` (list of names) + `question` (string) — no row data
    src = inspect.getsource(planner_module._gemini_plan)
    assert "available_datasets" in src
    assert "question" in src
    # None of these row-data operations should appear in _gemini_plan
    for forbidden in ("read_csv", ".values", "to_csv", "iterrows", "itertuples", "to_dict"):
        assert forbidden not in src, \
            f"_gemini_plan must not use '{forbidden}' — that would send row data to the LLM"

