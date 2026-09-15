from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.profiling import profile_dataframe

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_profile_dataframe_unit():
    # Hand-built messy DataFrame
    data = {
        "record_id": ["R01", "R02", "R02", "R04"],          # Duplicate key value
        "name": ["Alpha", "Beta", None, "Delta"],            # 1 null value
        "price": ["10.50", "25.00", "not_a_number", "40.00"], # 1 unparseable numeric value
        "status": ["active", "active", "pending", "pending"],
    }
    df = pd.DataFrame(data)

    profile = profile_dataframe(df)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 4

    # Assert null count
    assert profile["columns"]["name"]["null_count"] == 1
    assert profile["columns"]["name"]["null_pct"] == 25.0

    # Assert duplicate key detection for record_id
    assert "record_id" in profile["duplicate_key_columns"]
    assert profile["duplicate_key_columns"]["record_id"] == 1

    # Assert parse failure detection on price
    price_profile = profile["columns"]["price"]
    assert "parse_failures" in price_profile
    assert price_profile["parse_failures"]["attempted_as"] == "numeric"
    assert price_profile["parse_failures"]["failure_count"] == 1


def test_profile_runs_integration():
    csv_files = ["products.csv", "customers.csv", "stores.csv", "orders.csv", "inventory.csv"]

    with TestClient(app) as client:
        # Step 1: Upload the 5 sample CSVs
        open_files = []
        upload_payload = []
        try:
            for filename in csv_files:
                file_obj = open(SAMPLE_DIR / filename, "rb")
                open_files.append(file_obj)
                upload_payload.append(
                    ("files", (filename, file_obj, "text/csv"))
                )

            upload_res = client.post("/runs/upload", files=upload_payload)
        finally:
            for f in open_files:
                f.close()

        assert upload_res.status_code == 200, upload_res.text
        upload_data = upload_res.json()
        run_id = upload_data["run_id"]
        assert upload_data["status"] == "profiling"

        # Step 2: Trigger profiling
        profile_res = client.post(f"/runs/{run_id}/profile")
        assert profile_res.status_code == 200, profile_res.text
        profile_data = profile_res.json()

        assert profile_data["run_id"] == run_id
        assert profile_data["status"] == "profiled"
        assert "profiles" in profile_data

        profiles = profile_data["profiles"]
        assert "products" in profiles
        assert "orders" in profiles

        # Assert products' profile shows null_count > 0 for price column
        products_profile = profiles["products"]
        assert products_profile["columns"]["price"]["null_count"] > 0

        # Assert orders' profile shows duplicate_key_columns has order_id > 0
        orders_profile = profiles["orders"]
        assert "order_id" in orders_profile["duplicate_key_columns"]
        assert orders_profile["duplicate_key_columns"]["order_id"] > 0

        # Step 3: Verify GET /runs/{run_id} reflects profiled state and profiles
        get_res = client.get(f"/runs/{run_id}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["status"] == "profiled"
        for ds in get_data["datasets"]:
            assert ds["profile_json"] is not None
