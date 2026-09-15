from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.joins import get_join_config, safe_join

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_safe_join_unit():
    # Left DataFrame with an unmatched key
    left_df = pd.DataFrame(
        [
            {"order_id": 101, "product_id": "P01"},
            {"order_id": 102, "product_id": "P02"},
            {"order_id": 103, "product_id": "P_UNMATCHED"},
        ]
    )

    # Right DataFrame with a duplicate key (fan-out risk)
    right_df = pd.DataFrame(
        [
            {"product_id": "P01", "name": "Item 1 A"},
            {"product_id": "P01", "name": "Item 1 B"},  # Duplicate P01
            {"product_id": "P02", "name": "Item 2"},
        ]
    )

    joined_df, report = safe_join(
        left_df=left_df,
        right_df=right_df,
        left_key="product_id",
        right_key="product_id",
        how="left",
    )

    assert report["fan_out_detected"] is True
    assert report["fan_out_key_duplicate_count"] == 1
    assert report["unmatched_left_count"] == 1
    assert "P_UNMATCHED" in report["unmatched_left_examples"]
    assert report["left_rows"] == 3
    assert report["right_rows"] == 3
    # Left join with 1 duplicate on P01 yields 4 rows
    assert report["joined_rows"] == 4


def test_get_join_config_invalid_unit():
    with pytest.raises(ValueError) as exc_info:
        get_join_config("customers", "inventory")
    assert "not permitted by JOIN_REGISTRY" in str(exc_info.value)


def test_safe_join_rejects_invalid_mode_and_missing_keys():
    frame = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="Unsupported join mode"):
        safe_join(frame, frame, "id", "id", how="cross")

    with pytest.raises(ValueError, match="Right join key"):
        safe_join(frame, frame, "id", "missing")


def test_join_integration_valid_and_invalid():
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

        # Step 2: Profile
        profile_res = client.post(f"/runs/{run_id}/profile")
        assert profile_res.status_code == 200, profile_res.text

        # Step 3: Clean
        clean_res = client.post(f"/runs/{run_id}/clean")
        assert clean_res.status_code == 200, clean_res.text

        # Step 4: Valid join orders -> products
        join_res = client.post(
            f"/runs/{run_id}/join",
            json={"left": "orders", "right": "products", "how": "left"},
        )
        assert join_res.status_code == 200, join_res.text
        join_data = join_res.json()

        assert "report" in join_data
        assert "preview" in join_data

        report = join_data["report"]
        assert report["fan_out_detected"] is False
        assert report["fan_out_key_duplicate_count"] == 0
        # Check unmatched_left_count is close to the ~15 seeded broken references
        assert 5 <= report["unmatched_left_count"] <= 20
        assert len(report["unmatched_left_examples"]) > 0

        preview = join_data["preview"]
        assert len(preview) <= 20

        # Step 5: Invalid join pair customers -> orders
        invalid_res = client.post(
            f"/runs/{run_id}/join",
            json={"left": "customers", "right": "orders"},
        )
        assert invalid_res.status_code == 400
        assert "not permitted by JOIN_REGISTRY" in invalid_res.json()["detail"]
