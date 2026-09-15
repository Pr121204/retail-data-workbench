from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.analytics import revenue_by_category, stockout_and_ageing

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_revenue_by_category_unit():
    orders_df = pd.DataFrame(
        [
            {"order_id": "O1", "product_id": "P1", "revenue": 100.0},
            {"order_id": "O2", "product_id": "P1", "revenue": 200.0},
            {"order_id": "O3", "product_id": "P2", "revenue": 50.0},
            # Negative revenue to be excluded
            {"order_id": "O4", "product_id": "P2", "revenue": -15.0},
            # Null revenue to be excluded
            {"order_id": "O5", "product_id": "P2", "revenue": None},
        ]
    )

    products_df = pd.DataFrame(
        [
            {"product_id": "P1", "category": "Electronics"},
            {"product_id": "P2", "category": "Apparel"},
        ]
    )

    result = revenue_by_category(orders_df, products_df)

    assert "metrics" in result
    assert "computed_from" in result
    assert result["computed_from"]["rows_excluded_negative_or_null_revenue"] == 2

    metrics = result["metrics"]
    assert "Electronics" in metrics
    assert metrics["Electronics"]["total_revenue"] == 300.0
    assert metrics["Electronics"]["order_count"] == 2
    assert metrics["Electronics"]["aov"] == 150.0

    assert "Apparel" in metrics
    assert metrics["Apparel"]["total_revenue"] == 50.0
    assert metrics["Apparel"]["order_count"] == 1
    assert metrics["Apparel"]["aov"] == 50.0


def test_stockout_and_ageing_unit():
    inventory_df = pd.DataFrame(
        [
            {"product_id": "P1", "store_id": "S1", "stock_quantity": 0, "last_restock_date": "2023-01-01"},
            {"product_id": "P2", "store_id": "S1", "stock_quantity": 4, "last_restock_date": "2023-02-01"},
            {"product_id": "P3", "store_id": "S2", "stock_quantity": 50, "last_restock_date": None},
            {"product_id": "P4", "store_id": "S2", "stock_quantity": None, "last_restock_date": "2023-03-01"},
        ]
    )

    products_df = pd.DataFrame(
        [
            {"product_id": "P1", "product_name": "Widget Zero"},
            {"product_id": "P2", "product_name": "Widget Low"},
            {"product_id": "P3", "product_name": "Widget Good"},
            {"product_id": "P4", "product_name": "Widget Null"},
        ]
    )

    result = stockout_and_ageing(inventory_df, products_df)

    metrics = result["metrics"]
    stockout_pids = {item["product_id"] for item in metrics["stockout_items"]}
    assert stockout_pids == {"P1", "P4"}
    assert len(metrics["stockout_items"]) == 2

    low_stock_pids = {item["product_id"] for item in metrics["low_stock_items"]}
    assert low_stock_pids == {"P2"}
    assert len(metrics["low_stock_items"]) == 1
    assert metrics["low_stock_items"][0]["stock_quantity"] == 4

    assert metrics["ageing_unknown_count"] == 1


def test_analytics_integration():
    csv_files = ["products.csv", "customers.csv", "stores.csv", "orders.csv", "inventory.csv"]

    with TestClient(app) as client:
        # Step 1: Upload
        open_files = []
        upload_payload = []
        try:
            for filename in csv_files:
                file_obj = open(SAMPLE_DIR / filename, "rb")
                open_files.append(file_obj)
                upload_payload.append(("files", (filename, file_obj, "text/csv")))

            upload_res = client.post("/runs/upload", files=upload_payload)
        finally:
            for f in open_files:
                f.close()

        assert upload_res.status_code == 200, upload_res.text
        run_id = upload_res.json()["run_id"]

        # Step 2: Profile & Clean
        profile_res = client.post(f"/runs/{run_id}/profile")
        assert profile_res.status_code == 200, profile_res.text

        clean_res = client.post(f"/runs/{run_id}/clean")
        assert clean_res.status_code == 200, clean_res.text

        # Step 3: GET analytics
        analytics_res = client.get(f"/runs/{run_id}/analytics")
        assert analytics_res.status_code == 200, analytics_res.text
        data = analytics_res.json()

        expected_keys = [
            "revenue_by_category",
            "store_performance",
            "return_rate_by_category",
            "stockout_and_ageing",
        ]
        for key in expected_keys:
            assert key in data, f"Missing metric key '{key}' in analytics response"
            assert "error" not in data[key], f"Metric '{key}' returned error: {data[key].get('error')}"
