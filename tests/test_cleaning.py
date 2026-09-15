from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.cleaning import apply_cleaning_plan, build_cleaning_plan

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_cleaning_unit():
    # Build a small messy DataFrame by hand
    df = pd.DataFrame(
        [
            {"item_name": "  Widget A", "region": "west", "price": "10.5", "status": "active"},
            {"item_name": "Widget B ", "region": "WEST", "price": "unparseable_string", "status": "active"},
            {"item_name": "Widget C", "region": "East", "price": "30.0", "status": "pending"},
            # Fully duplicate row identical to row 0
            {"item_name": "  Widget A", "region": "west", "price": "10.5", "status": "active"},
        ]
    )

    plan = build_cleaning_plan(df, "items")
    df_clean, executed_plan = apply_cleaning_plan(df, plan)

    # 1. Assert exact duplicate row was removed (4 rows -> 3 rows)
    assert len(df_clean) == 3

    # 2. Assert whitespace trimmed on string columns
    for name in df_clean["item_name"]:
        assert name == name.strip()

    # 3. Assert case normalized for region (collapses to title-case)
    assert set(df_clean["region"].unique()).issubset({"West", "East"})
    assert "west" not in df_clean["region"].values
    assert "WEST" not in df_clean["region"].values

    # 4. Assert row count did NOT drop because of numeric coercion
    # The row with 'unparseable_string' is kept with NaN in price
    assert len(df_clean) == 3
    assert pd.isna(df_clean.loc[1, "price"])


def test_cleaning_integration():
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
        clean_data = clean_res.json()

        assert clean_data["run_id"] == run_id
        assert clean_data["status"] == "cleaned"
        assert "results" in clean_data

        # Assert products dataset validation shows row_count_before == row_count_after == 60
        products_val = clean_data["results"]["products"]["validation"]
        assert products_val["row_count_before"] == 60
        assert products_val["row_count_after"] == 60

        # Assert customers cleaning normalized region case variants
        cleaned_cust_path = Path("data") / "runs" / run_id / "clean" / "customers.csv"
        assert cleaned_cust_path.exists()
        df_cleaned_cust = pd.read_csv(cleaned_cust_path)
        unique_regions = set(df_cleaned_cust["region"].dropna().unique())
        assert unique_regions == {"North", "South", "East", "West"}
