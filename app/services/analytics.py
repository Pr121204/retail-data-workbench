from typing import Any, Dict
import pandas as pd

from app.services.joins import safe_join


def revenue_by_category(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes total revenue, order count, and AOV per category.
    Drops rows with null or negative revenue before aggregating.
    """
    joined_df, join_report = safe_join(orders_df, products_df, "product_id", "product_id", how="left")

    # Exclude null and negative revenue rows
    invalid_mask = joined_df["revenue"].isna() | (joined_df["revenue"] < 0)
    rows_excluded = int(invalid_mask.sum())
    valid_df = joined_df[~invalid_mask].copy()

    metrics: Dict[str, Dict[str, Any]] = {}
    for cat, group in valid_df.groupby("category", dropna=False):
        category_name = str(cat) if pd.notna(cat) else "Unknown"
        tot_rev = round(float(group["revenue"].sum()), 2)
        ord_cnt = int(len(group))
        aov = round(float(tot_rev / ord_cnt), 2) if ord_cnt > 0 else 0.0

        metrics[category_name] = {
            "total_revenue": tot_rev,
            "order_count": ord_cnt,
            "aov": aov,
        }

    return {
        "metrics": metrics,
        "computed_from": {
            "join_report": join_report,
            "rows_excluded_negative_or_null_revenue": rows_excluded,
            "dataset_versions": {
                "orders": int(len(orders_df)),
                "products": int(len(products_df)),
            },
        },
    }


def store_performance(orders_df: pd.DataFrame, stores_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes total revenue, order count, and distinct customer count per store.
    """
    joined_df, join_report = safe_join(orders_df, stores_df, "store_id", "store_id", how="left")

    # Exclude null and negative revenue rows for store revenue calculations
    invalid_mask = joined_df["revenue"].isna() | (joined_df["revenue"] < 0)
    rows_excluded = int(invalid_mask.sum())
    valid_df = joined_df[~invalid_mask].copy()

    metrics: Dict[str, Dict[str, Any]] = {}
    for store_id, group in valid_df.groupby("store_id", dropna=False):
        sid = str(store_id) if pd.notna(store_id) else "Unknown"
        store_name = (
            str(group["store_name"].dropna().iloc[0])
            if "store_name" in group.columns and group["store_name"].notna().any()
            else sid
        )
        tot_rev = round(float(group["revenue"].sum()), 2)
        ord_cnt = int(len(group))
        distinct_custs = (
            int(group["customer_id"].dropna().nunique())
            if "customer_id" in group.columns
            else 0
        )

        metrics[sid] = {
            "store_name": store_name,
            "total_revenue": tot_rev,
            "order_count": ord_cnt,
            "distinct_customer_count": distinct_custs,
        }

    return {
        "metrics": metrics,
        "computed_from": {
            "join_report": join_report,
            "rows_excluded_negative_or_null_revenue": rows_excluded,
            "dataset_versions": {
                "orders": int(len(orders_df)),
                "stores": int(len(stores_df)),
            },
        },
    }


def return_rate_by_category(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes return rate per category based on orders with revenue < 0.
    """
    joined_df, join_report = safe_join(orders_df, products_df, "product_id", "product_id", how="left")

    # Filter to rows where revenue is not null
    valid_orders = joined_df[joined_df["revenue"].notna()].copy()

    metrics: Dict[str, Dict[str, Any]] = {}
    for cat, group in valid_orders.groupby("category", dropna=False):
        category_name = str(cat) if pd.notna(cat) else "Unknown"
        ret_cnt = int((group["revenue"] < 0).sum())
        total_cnt = int(len(group))
        ret_rate = round(float(ret_cnt / total_cnt), 4) if total_cnt > 0 else 0.0

        metrics[category_name] = {
            "return_count": ret_cnt,
            "total_order_count": total_cnt,
            "return_rate": ret_rate,
        }

    return {
        "metrics": metrics,
        "assumption": "orders with revenue < 0 are treated as returns",
        "computed_from": {
            "join_report": join_report,
            "dataset_versions": {
                "orders": int(len(orders_df)),
                "products": int(len(products_df)),
            },
        },
    }


def stockout_and_ageing(inventory_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies stockout items (qty <= 0 or null), low stock items (0 < qty <= 5),
    and counts records with unknown ageing (null last_restock_date).
    """
    joined_df, join_report = safe_join(inventory_df, products_df, "product_id", "product_id", how="left")

    stockout_items = []
    low_stock_items = []

    for _, row in joined_df.iterrows():
        qty = row.get("stock_quantity")
        pid = str(row.get("product_id", ""))
        pname = row.get("product_name")
        pname_str = str(pname) if pd.notna(pname) else None
        sid = str(row.get("store_id", ""))

        if pd.isna(qty) or qty == 0:
            stockout_items.append(
                {
                    "product_id": pid,
                    "product_name": pname_str,
                    "store_id": sid,
                }
            )
        elif 0 < qty <= 5:
            stockout_items_val = int(qty) if isinstance(qty, (int, float)) and float(qty).is_integer() else round(float(qty), 2)
            low_stock_items.append(
                {
                    "product_id": pid,
                    "product_name": pname_str,
                    "store_id": sid,
                    "stock_quantity": stockout_items_val,
                }
            )

    ageing_unknown_count = (
        int(joined_df["last_restock_date"].isna().sum())
        if "last_restock_date" in joined_df.columns
        else 0
    )

    return {
        "metrics": {
            "stockout_items": stockout_items,
            "low_stock_items": low_stock_items,
            "ageing_unknown_count": ageing_unknown_count,
        },
        "low_stock_threshold": 5,
        "computed_from": {
            "join_report": join_report,
            "dataset_versions": {
                "inventory": int(len(inventory_df)),
                "products": int(len(products_df)),
            },
        },
    }


def compute_all_analytics(clean_datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Computes all analytics metrics across loaded clean datasets.
    Defensively catches exceptions per metric function.
    """
    results: Dict[str, Any] = {}

    # 1. revenue_by_category
    try:
        if "orders" in clean_datasets and "products" in clean_datasets:
            results["revenue_by_category"] = revenue_by_category(
                clean_datasets["orders"], clean_datasets["products"]
            )
        else:
            results["revenue_by_category"] = {"error": "Missing required datasets 'orders' or 'products'"}
    except Exception as e:
        results["revenue_by_category"] = {"error": str(e)}

    # 2. store_performance
    try:
        if "orders" in clean_datasets and "stores" in clean_datasets:
            results["store_performance"] = store_performance(
                clean_datasets["orders"], clean_datasets["stores"]
            )
        else:
            results["store_performance"] = {"error": "Missing required datasets 'orders' or 'stores'"}
    except Exception as e:
        results["store_performance"] = {"error": str(e)}

    # 3. return_rate_by_category
    try:
        if "orders" in clean_datasets and "products" in clean_datasets:
            results["return_rate_by_category"] = return_rate_by_category(
                clean_datasets["orders"], clean_datasets["products"]
            )
        else:
            results["return_rate_by_category"] = {"error": "Missing required datasets 'orders' or 'products'"}
    except Exception as e:
        results["return_rate_by_category"] = {"error": str(e)}

    # 4. stockout_and_ageing
    try:
        if "inventory" in clean_datasets and "products" in clean_datasets:
            results["stockout_and_ageing"] = stockout_and_ageing(
                clean_datasets["inventory"], clean_datasets["products"]
            )
        else:
            results["stockout_and_ageing"] = {"error": "Missing required datasets 'inventory' or 'products'"}
    except Exception as e:
        results["stockout_and_ageing"] = {"error": str(e)}

    return results
