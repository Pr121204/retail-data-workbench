"""
LLM Planner: converts a natural language question into a QueryPlan.

When USE_MOCK_LLM=True (default), returns a deterministic mock plan based on
simple keyword matching — no external calls, fully reproducible.

When USE_MOCK_LLM=False, calls the Gemini API with a structured prompt and
parses the JSON response into a QueryPlan.

The function NEVER validates the plan — validation is the caller's responsibility
(use plan_validator.validate_plan).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from app.config import settings
from app.services.plan_schema import Filter, Metric, QueryPlan, Sort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a retail data analyst assistant. Given a natural-language question and a
description of the available clean datasets, you must output ONLY a single JSON
object (no markdown, no explanation) that represents a QueryPlan to answer the
question.

Available datasets and their fields:
  - orders:    order_id, customer_id, product_id, store_id, quantity, revenue, order_date, region
  - products:  product_id, product_name, category, brand, sku, price
  - customers: customer_id, name, email, region, signup_date
  - stores:    store_id, store_name, region, city
  - inventory: product_id, store_id, stock_quantity, last_restock_date

Allowed joins (left → right):
  - orders → products    (on product_id)
  - orders → customers   (on customer_id)
  - orders → stores      (on store_id)
  - inventory → products (on product_id)
  - inventory → stores   (on store_id)

QueryPlan JSON schema:
{
  "intent":   "filter" | "aggregate" | "compare" | "top_n" | "describe",
  "dataset":  "<one of the 5 dataset names>",
  "join":     {"with": "<dataset>"} | null,
  "filters":  [{"field": "...", "op": "eq|neq|gt|gte|lt|lte|in|contains", "value": ...}],
  "group_by": ["<field>", ...],
  "metrics":  [{"agg": "sum|avg|count|min|max|count_distinct", "field": "<field>|*", "as": "<alias>"}],
  "sort":     [{"field": "<field_or_alias>", "dir": "asc|desc"}],
  "limit":    <int 1-200>
}

Rules:
- Use only the field names listed above.  Use "*" as the field only for count.
- If the question asks for a top-N result, set intent="top_n" and include a sort + limit.
- If the question asks about a single dataset with no aggregation, use intent="filter".
- If the question needs a join, always set the join key.
- Do NOT add any field that doesn't exist in the schema above.
- Output raw JSON only, with no surrounding text.
"""


# ---------------------------------------------------------------------------
# Mock planner — used when USE_MOCK_LLM=True
# ---------------------------------------------------------------------------

def _mock_plan(question: str, available_datasets: list[str]) -> QueryPlan:
    """
    Returns a deterministic mock QueryPlan based on keyword matching in the question.
    The intent is to give realistic-looking plans for demo purposes without any API call.
    """
    q = question.lower()

    # Revenue by category (most common demo query)
    if "revenue" in q and "category" in q:
        return QueryPlan(
            intent="aggregate",
            dataset="orders",
            join={"with": "products"},
            filters=[],
            group_by=["category"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
                Metric(**{"agg": "avg", "field": "revenue", "as": "aov"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
            limit=20,
        )

    # Top products by revenue
    if ("top" in q or "best" in q) and "product" in q and "revenue" in q:
        return QueryPlan(
            intent="top_n",
            dataset="orders",
            join={"with": "products"},
            filters=[],
            group_by=["product_name"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
            limit=10,
        )

    # Bottom products by revenue
    if ("bottom" in q or "worst" in q or "lowest" in q) and "product" in q and "revenue" in q:
        return QueryPlan(
            intent="top_n",
            dataset="orders",
            join={"with": "products"},
            filters=[],
            group_by=["product_name"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
            ],
            sort=[Sort(field="total_revenue", dir="asc")],
            limit=10,
        )

    # Distinct values are useful for exploratory questions and do not require
    # an LLM or unrestricted expression evaluation.
    if "distinct" in q or "unique" in q:
        for field, dataset in (
            ("category", "products"),
            ("brand", "products"),
            ("region", "orders"),
            ("city", "stores"),
        ):
            if field in q and dataset in available_datasets:
                return QueryPlan(
                    intent="aggregate",
                    dataset=dataset,
                    filters=[],
                    group_by=[field],
                    metrics=[Metric(**{"agg": "count", "field": "*", "as": "count"})],
                    sort=[Sort(field=field, dir="asc")],
                    limit=200,
                )

    # Descriptive statistics over a numeric retail measure.
    if "statistic" in q or "statistics" in q or "describe" in q:
        field = "revenue" if "revenue" in q or "sales" in q else "quantity"
        if "orders" in available_datasets:
            return QueryPlan(
                intent="describe",
                dataset="orders",
                filters=[],
                group_by=[],
                metrics=[
                    Metric(**{"agg": "count", "field": field, "as": "count"}),
                    Metric(**{"agg": "avg", "field": field, "as": "average"}),
                    Metric(**{"agg": "min", "field": field, "as": "minimum"}),
                    Metric(**{"agg": "max", "field": field, "as": "maximum"}),
                ],
                sort=[],
                limit=1,
            )

    # Store performance
    if "store" in q and ("revenue" in q or "performance" in q or "sales" in q):
        return QueryPlan(
            intent="aggregate",
            dataset="orders",
            join={"with": "stores"},
            filters=[],
            group_by=["store_name"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
            limit=20,
        )

    # Low stock / inventory
    if "stock" in q or "inventory" in q or "low" in q:
        return QueryPlan(
            intent="filter",
            dataset="inventory",
            join={"with": "products"},
            filters=[Filter(field="stock_quantity", op="lt", value=10)],
            group_by=[],
            metrics=[],
            sort=[Sort(field="stock_quantity", dir="asc")],
            limit=20,
        )

    # Customer count by region
    if "customer" in q and "region" in q:
        return QueryPlan(
            intent="aggregate",
            dataset="customers",
            join=None,
            filters=[],
            group_by=["region"],
            metrics=[
                Metric(**{"agg": "count", "field": "*", "as": "customer_count"}),
            ],
            sort=[Sort(field="customer_count", dir="desc")],
            limit=20,
        )

    # Orders in a specific region
    if "region" in q and "order" in q:
        return QueryPlan(
            intent="aggregate",
            dataset="orders",
            join=None,
            filters=[],
            group_by=["region"],
            metrics=[
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
            limit=20,
        )

    # Average order value
    if "average" in q and ("order" in q or "revenue" in q):
        return QueryPlan(
            intent="aggregate",
            dataset="orders",
            join=None,
            filters=[],
            group_by=[],
            metrics=[
                Metric(**{"agg": "avg", "field": "revenue", "as": "avg_order_value"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
            ],
            sort=[],
            limit=1,
        )

    # Fallback: describe orders
    primary_dataset = "orders" if "orders" in available_datasets else (available_datasets[0] if available_datasets else "orders")
    for ds in ["products", "customers", "stores", "inventory"]:
        if ds in q and ds in available_datasets:
            primary_dataset = ds
            break

    return QueryPlan(
        intent="describe",
        dataset=primary_dataset,
        join=None,
        filters=[],
        group_by=[],
        metrics=[],
        sort=[],
        limit=20,
    )


# ---------------------------------------------------------------------------
# Gemini API planner — used when USE_MOCK_LLM=False
# ---------------------------------------------------------------------------

def _gemini_plan(question: str, available_datasets: list[str]) -> QueryPlan:
    """
    Calls the Gemini API to generate a QueryPlan from a natural language question.
    Parses the raw JSON response and constructs a QueryPlan object.
    Raises ValueError if the response cannot be parsed.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Either set it in .env or enable USE_MOCK_LLM=True."
        )

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    user_message = (
        f"Available datasets in this run: {', '.join(available_datasets)}\n\n"
        f"Question: {question}"
    )

    logger.info("Calling Gemini API for question: %r", question)
    try:
        response = model.generate_content(
            [_SYSTEM_PROMPT, user_message],
            generation_config={"temperature": 0, "max_output_tokens": 1024},
            request_options={"timeout": 15},  # 15-second hard timeout
        )
    except Exception as api_exc:
        raise ValueError(f"Gemini API call failed: {api_exc}") from api_exc

    raw_text = response.text.strip()

    # Strip markdown code fences if the model wrapped the JSON
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove first and last fence lines
        raw_text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    logger.debug("Gemini raw response: %s", raw_text)

    try:
        plan_dict = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON response: {raw_text[:200]}"
        ) from exc

    # Remap "as" alias for Metric (pydantic uses alias "as_")
    metrics_raw = plan_dict.get("metrics", [])
    metrics = []
    for m in metrics_raw:
        metrics.append(
            Metric(**{"agg": m["agg"], "field": m["field"], "as": m.get("as", m.get("as_", "value"))})
        )

    filters = [Filter(**f) for f in plan_dict.get("filters", [])]
    sort = [Sort(**s) for s in plan_dict.get("sort", [])]

    return QueryPlan(
        intent=plan_dict["intent"],
        dataset=plan_dict["dataset"],
        join=plan_dict.get("join"),
        filters=filters,
        group_by=plan_dict.get("group_by", []),
        metrics=metrics,
        sort=sort,
        limit=int(plan_dict.get("limit", 20)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_plan(
    question: str,
    available_datasets: Optional[list[str]] = None,
    allow_fallback: bool = True,
) -> tuple[QueryPlan, str]:
    """
    Generates a QueryPlan from a natural language question.

    Args:
        question: The user's natural language question.
        available_datasets: List of dataset names available in the run.
                            Defaults to all 5 known datasets.
        allow_fallback: If True (default), a failed LLM call silently degrades
                        to the mock planner and returns source='mock_fallback'.
                        Set False to let exceptions propagate (useful in tests).

    Returns:
        A tuple of (QueryPlan, source) where source is:
          - "mock"          — USE_MOCK_LLM=True, normal mock path
          - "llm"           — Gemini API returned a valid plan
          - "mock_fallback" — Gemini call failed; fell back to mock planner

    Raises:
        ValueError: Only if allow_fallback=False and the LLM fails.
    """
    if available_datasets is None:
        available_datasets = ["orders", "products", "customers", "stores", "inventory"]

    if settings.USE_MOCK_LLM:
        plan = _mock_plan(question, available_datasets)
        return plan, "mock"

    try:
        plan = _gemini_plan(question, available_datasets)
        return plan, "llm"
    except Exception as exc:
        if not allow_fallback:
            raise
        logger.warning(
            "LLM plan generation failed (%s: %s) — falling back to mock planner.",
            type(exc).__name__,
            exc,
        )
        plan = _mock_plan(question, available_datasets)
        return plan, "mock_fallback"
