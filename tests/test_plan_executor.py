"""
Tests for app/services/plan_executor.py

Covers:
  - Aggregate with group_by
  - Filters (eq, gt, contains, in)
  - Join (valid registry pair)
  - Sort + limit
  - Scalar metrics (no group_by)
  - Missing dataset -> execution_error, not an unhandled exception
  - Filter on missing field -> warning, not crash
  - JSON-safe output (no NaN, no numpy types)
  - result_row_count vs len(result_rows) with limit
  - evidence keys are always present
"""

import math

import pandas as pd
import pytest

from app.services.plan_executor import execute_plan
from app.services.plan_schema import Filter, Metric, QueryPlan, Sort


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def orders_df():
    return pd.DataFrame({
        "order_id":   ["O1", "O2", "O3", "O4", "O5"],
        "product_id": ["P1", "P2", "P1", "P3", "P2"],
        "store_id":   ["S1", "S1", "S2", "S2", "S1"],
        "customer_id":["C1", "C2", "C1", "C3", "C2"],
        "revenue":    [100.0, 200.0, 150.0, 300.0, 250.0],
        "quantity":   [1, 2, 1, 3, 2],
        "region":     ["North", "South", "North", "East", "South"],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    })


@pytest.fixture()
def products_df():
    return pd.DataFrame({
        "product_id":   ["P1", "P2", "P3"],
        "product_name": ["Widget", "Gadget", "Doohickey"],
        "category":     ["Electronics", "Apparel", "Electronics"],
        "brand":        ["BrandA", "BrandB", "BrandA"],
        "sku":          ["SKU-1", "SKU-2", "SKU-3"],
        "price":        [99.99, 199.99, 299.99],
    })


@pytest.fixture()
def clean_datasets(orders_df, products_df):
    return {"orders": orders_df, "products": products_df}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_plan(**kwargs) -> QueryPlan:
    defaults = dict(
        intent="aggregate",
        dataset="orders",
        join=None,
        filters=[],
        group_by=[],
        metrics=[],
        sort=[],
        limit=20,
    )
    defaults.update(kwargs)
    return QueryPlan(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecutePlanAggregate:

    def test_aggregate_group_by_region(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            group_by=["region"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
                Metric(**{"agg": "count", "field": "*", "as": "order_count"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        rows = result["result_rows"]
        assert len(rows) == 3  # North, South, East
        # South = 200+250=450, North = 100+150=250, East = 300
        regions = [r["region"] for r in rows]
        assert regions[0] == "South"
        assert "total_revenue" in rows[0]
        assert "order_count" in rows[0]

    def test_grouped_count_star_counts_rows_and_preserves_null_groups(self, clean_datasets):
        clean_datasets["orders"] = clean_datasets["orders"].copy()
        clean_datasets["orders"].loc[0, "region"] = None
        plan = make_plan(
            intent="aggregate",
            group_by=["region"],
            metrics=[Metric(**{"agg": "count", "field": "*", "as": "order_count"})],
        )

        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        counts = {row["region"]: row["order_count"] for row in result["result_rows"]}
        assert counts[None] == 1
        assert counts["South"] == 2

    def test_scalar_metrics_no_group_by(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            group_by=[],
            metrics=[
                Metric(**{"agg": "sum",   "field": "revenue", "as": "total"}),
                Metric(**{"agg": "avg",   "field": "revenue", "as": "avg_rev"}),
                Metric(**{"agg": "count", "field": "*",       "as": "n"}),
                Metric(**{"agg": "min",   "field": "revenue", "as": "min_rev"}),
                Metric(**{"agg": "max",   "field": "revenue", "as": "max_rev"}),
            ],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_row_count"] == 1
        row = result["result_rows"][0]
        assert row["total"] == 1000.0
        assert row["n"] == 5
        assert row["min_rev"] == 100.0
        assert row["max_rev"] == 300.0
        assert abs(row["avg_rev"] - 200.0) < 0.01

    def test_count_distinct(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            group_by=[],
            metrics=[
                Metric(**{"agg": "count_distinct", "field": "product_id", "as": "unique_products"}),
            ],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_rows"][0]["unique_products"] == 3


class TestExecutePlanFilter:

    def test_filter_eq(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="region", op="eq", value="North")],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        rows = result["result_rows"]
        assert all(r["region"] == "North" for r in rows)
        assert result["result_row_count"] == 2

    def test_filter_gt(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="revenue", op="gt", value=200.0)],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_row_count"] == 2
        assert all(r["revenue"] > 200 for r in result["result_rows"])

    def test_filter_in(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="region", op="in", value=["North", "East"])],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_row_count"] == 3

    def test_filter_contains(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="order_id", op="contains", value="O1")],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_row_count"] == 1

    def test_filter_on_missing_field_returns_structured_execution_error(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="nonexistent_col", op="eq", value="X")],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is not None
        assert result["result_rows"] == []
        assert "nonexistent_col" in result["execution_error"]

    def test_multiple_filters_are_cumulative(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[
                Filter(field="region", op="eq", value="South"),
                Filter(field="revenue", op="gte", value=250.0),
            ],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["result_row_count"] == 1
        assert result["result_rows"][0]["order_id"] == "O5"


class TestExecutePlanJoin:

    def test_join_orders_to_products(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            dataset="orders",
            join={"with": "products"},
            group_by=["category"],
            metrics=[
                Metric(**{"agg": "sum", "field": "revenue", "as": "total_revenue"}),
            ],
            sort=[Sort(field="total_revenue", dir="desc")],
        )
        result = execute_plan(plan, clean_datasets)

        assert result.get("execution_error") is None
        assert result["evidence"]["join_report"] is not None
        assert "product_id" in result["evidence"]["columns_used"]
        rows = result["result_rows"]
        categories = {r["category"]: r["total_revenue"] for r in rows}
        assert categories["Electronics"] == 550.0
        assert categories["Apparel"] == 450.0

    def test_join_report_keys_are_present(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            dataset="orders",
            join={"with": "products"},
            group_by=["category"],
            metrics=[Metric(**{"agg": "count", "field": "*", "as": "n"})],
        )
        result = execute_plan(plan, clean_datasets)

        jr = result["evidence"]["join_report"]
        for key in ("left_rows", "right_rows", "joined_rows", "left_key", "right_key",
                    "fan_out_detected", "unmatched_left_count"):
            assert key in jr, f"join_report missing key: {key}"

    def test_invalid_join_dataset_returns_execution_error(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            dataset="orders",
            join={"with": "stores"},
            group_by=["store_id"],
            metrics=[Metric(**{"agg": "count", "field": "*", "as": "n"})],
        )
        result = execute_plan(plan, clean_datasets)

        assert "execution_error" in result
        assert result["execution_error"] is not None


class TestExecutePlanSortAndLimit:

    def test_sort_asc(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            sort=[Sort(field="revenue", dir="asc")],
        )
        result = execute_plan(plan, clean_datasets)

        revenues = [r["revenue"] for r in result["result_rows"]]
        assert revenues == sorted(revenues)

    def test_limit_restricts_result_rows_but_not_result_row_count(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            sort=[Sort(field="revenue", dir="desc")],
            limit=2,
        )
        result = execute_plan(plan, clean_datasets)

        assert result["result_row_count"] == 5
        assert len(result["result_rows"]) == 2

    def test_limit_one_returns_single_row(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            group_by=[],
            metrics=[Metric(**{"agg": "sum", "field": "revenue", "as": "total"})],
            limit=1,
        )
        result = execute_plan(plan, clean_datasets)

        assert len(result["result_rows"]) == 1


class TestExecutePlanEvidence:

    def test_evidence_keys_always_present(self, clean_datasets):
        plan = make_plan()
        result = execute_plan(plan, clean_datasets)

        ev = result["evidence"]
        for key in ("dataset", "join", "join_report", "filters_applied",
                    "filters_warnings", "group_by", "metrics",
                    "row_count_before_filter", "row_count_after_filter", "limit_applied"):
            assert key in ev, f"evidence missing key: {key}"

    def test_row_counts_in_evidence_are_ints(self, clean_datasets):
        plan = make_plan(
            intent="filter",
            filters=[Filter(field="region", op="eq", value="North")],
        )
        result = execute_plan(plan, clean_datasets)

        ev = result["evidence"]
        assert isinstance(ev["row_count_before_filter"], int)
        assert isinstance(ev["row_count_after_filter"], int)
        assert ev["row_count_before_filter"] == 5
        assert ev["row_count_after_filter"] == 2

    def test_evidence_contains_lineage_operations_and_bounded_preview(self, clean_datasets):
        plan = make_plan(
            intent="aggregate",
            group_by=["region"],
            metrics=[Metric(**{"agg": "count", "field": "*", "as": "orders"})],
        )
        result = execute_plan(
            plan,
            clean_datasets,
            evidence_context={
                "run_id": "run-123",
                "dataset_stage": "clean",
                "dataset_version": "run-123:clean",
            },
        )

        evidence = result["evidence"]
        assert evidence["run_id"] == "run-123"
        assert evidence["dataset_stage"] == "clean"
        assert evidence["dataset_version"] == "run-123:clean"
        assert evidence["columns_used"] == ["region"]
        assert "group_by" in evidence["operations"]
        assert "count" in evidence["operations"]
        assert len(evidence["result_preview"]) <= 20


class TestExecutePlanMissingDataset:

    def test_missing_dataset_returns_execution_error(self, clean_datasets):
        plan = make_plan(dataset="inventory")
        result = execute_plan(plan, clean_datasets)

        assert "execution_error" in result
        assert result["execution_error"] is not None
        assert result["result_rows"] == []
        assert result["result_row_count"] == 0

    def test_evidence_present_even_on_error(self, clean_datasets):
        plan = make_plan(dataset="inventory")
        result = execute_plan(plan, clean_datasets)

        ev = result["evidence"]
        for key in ("dataset", "join", "join_report", "filters_applied",
                    "filters_warnings", "group_by", "metrics",
                    "row_count_before_filter", "row_count_after_filter", "limit_applied"):
            assert key in ev, f"evidence missing key '{key}' on error path"


class TestExecutePlanJsonSafe:

    def test_no_nan_in_output(self):
        df = pd.DataFrame({
            "product_id": ["P1", "P2"],
            "revenue":    [100.0, float("nan")],
            "region":     ["North", "South"],
        })
        plan = make_plan(dataset="orders", intent="filter")
        result = execute_plan(plan, {"orders": df})

        for row in result["result_rows"]:
            for val in row.values():
                if isinstance(val, float):
                    assert not math.isnan(val), "NaN must be converted to None in output"

    def test_no_numpy_types_in_output(self, clean_datasets):
        import numpy as np

        plan = make_plan(
            intent="aggregate",
            group_by=["region"],
            metrics=[
                Metric(**{"agg": "sum",   "field": "revenue", "as": "total"}),
                Metric(**{"agg": "count", "field": "*",       "as": "n"}),
            ],
        )
        result = execute_plan(plan, clean_datasets)

        for row in result["result_rows"]:
            for val in row.values():
                assert not isinstance(val, (np.integer, np.floating, np.bool_)), \
                    f"numpy type {type(val)} found — must be native Python"
