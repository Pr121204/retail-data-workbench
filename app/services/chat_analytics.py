"""
Bridges chat questions to the retail analytics functions in analytics.py.

Questions matched by chat_intent.match_analytics_capability() need business
logic the generic QueryPlan/execute_plan pipeline cannot express (e.g. a
"return" is an order with revenue < 0 — there is no returns column). This
module loads the required clean DataFrames, applies the session's carried-over
region filter as a PRE-filter, calls the matching analytics function, and
renders answer_text purely by reading values out of the computed result —
numbers are never invented here, only formatted from the real dict.
"""
from __future__ import annotations

import pandas as pd

from app.services.analytics import (
    return_rate_by_category,
    revenue_by_category,
    stockout_and_ageing,
    store_performance,
)


def _apply_region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Pre-filter a DataFrame to a single region; returns a copy."""
    if "region" in df.columns:
        return df[df["region"] == region].copy()
    return df.copy()


# ---------------------------------------------------------------------------
# Grounded answer summarisers — every number comes from the computed result
# ---------------------------------------------------------------------------

def _summarise_return_rate(result: dict) -> str:
    metrics = result.get("metrics", {})
    if not metrics:
        return "No category data was available to compute return rates."
    name, m = max(metrics.items(), key=lambda kv: kv[1].get("return_rate", 0.0))
    return (
        f"Across {len(metrics)} categories, '{name}' has the highest return rate: "
        f"{m['return_rate']:.1%} ({m['return_count']} of {m['total_order_count']} orders). "
        f"Assumption: {result.get('assumption', 'orders with revenue < 0 are returns')}."
    )


def _summarise_revenue_by_category(result: dict) -> str:
    metrics = result.get("metrics", {})
    if not metrics:
        return "No category data was available to compute revenue."
    name, m = max(metrics.items(), key=lambda kv: kv[1].get("total_revenue", 0.0))
    combined = round(sum(v.get("total_revenue", 0.0) for v in metrics.values()), 2)
    return (
        f"Across {len(metrics)} categories, '{name}' leads with total revenue "
        f"{m['total_revenue']} ({m['order_count']} orders, AOV {m['aov']}). "
        f"All categories combined: {combined}."
    )


def _summarise_store_performance(result: dict) -> str:
    metrics = result.get("metrics", {})
    if not metrics:
        return "No store data was available."
    sid, m = max(metrics.items(), key=lambda kv: kv[1].get("total_revenue", 0.0))
    return (
        f"Across {len(metrics)} stores, '{m['store_name']}' (store {sid}) leads with "
        f"total revenue {m['total_revenue']} from {m['order_count']} orders and "
        f"{m['distinct_customer_count']} distinct customers."
    )


def _summarise_stockout(result: dict) -> str:
    metrics = result.get("metrics", {})
    stockouts = metrics.get("stockout_items", [])
    low = metrics.get("low_stock_items", [])
    ageing = metrics.get("ageing_unknown_count", 0)
    text = (
        f"Found {len(stockouts)} stockout item(s) and {len(low)} low-stock item(s) "
        f"(threshold {result.get('low_stock_threshold', 5)}); "
        f"{ageing} record(s) have unknown restock age."
    )
    if stockouts:
        sample = ", ".join(f"{i['product_id']} ({i['store_id']})" for i in stockouts[:3])
        text += f" Stockouts include: {sample}."
    return text


_SUMMARISERS = {
    "revenue_by_category": _summarise_revenue_by_category,
    "return_rate_by_category": _summarise_return_rate,
    "store_performance": _summarise_store_performance,
    "stockout_and_ageing": _summarise_stockout,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analytics_capability(
    capability: str,
    clean_datasets: dict,
    active_filters: dict | None = None,
) -> tuple[dict, str, bool]:
    """
    Runs the named analytics capability against the run's clean DataFrames.

    Returns (result, answer_text, region_filter_applied). `result` is the full
    analytics dict (metrics + computed_from) — stored verbatim as evidence.
    `answer_text` is grounded string formatting of that result. The region
    from active_filters is pre-applied to the orders frame for order-based
    capabilities, so carried-over context like {"region": "West"} scopes the
    capability answer too.

    Raises ValueError if the capability is unknown or its datasets are missing.
    """
    active_filters = active_filters or {}
    region = active_filters.get("region")
    orders_df = clean_datasets.get("orders")
    products_df = clean_datasets.get("products")
    summariser = _SUMMARISERS.get(capability)
    if summariser is None:
        raise ValueError(f"Unknown analytics capability '{capability}'")

    if capability in ("revenue_by_category", "return_rate_by_category"):
        if orders_df is None or products_df is None:
            raise ValueError("This question requires the 'orders' and 'products' datasets in the run")
        orders_in = orders_df
        region_applied = False
        if region is not None:
            orders_in = _apply_region_filter(orders_df, region)
            region_applied = "region" in orders_df.columns
        func = return_rate_by_category if capability == "return_rate_by_category" else revenue_by_category
        result = func(orders_in, products_df)
        return result, summariser(result), region_applied

    if capability == "store_performance":
        stores_df = clean_datasets.get("stores")
        if orders_df is None or stores_df is None:
            raise ValueError("This question requires the 'orders' and 'stores' datasets in the run")
        orders_in = orders_df
        region_applied = False
        if region is not None:
            orders_in = _apply_region_filter(orders_df, region)
            region_applied = "region" in orders_df.columns
        result = store_performance(orders_in, stores_df)
        return result, summariser(result), region_applied

    # stockout_and_ageing: inventory has no region column, so a region filter
    # cannot be applied — carried region context simply doesn't scope it.
    inventory_df = clean_datasets.get("inventory")
    if inventory_df is None or products_df is None:
        raise ValueError("This question requires the 'inventory' and 'products' datasets in the run")
    result = stockout_and_ageing(inventory_df, products_df)
    return result, summariser(result), False
