from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_upload_sample_datasets():
    csv_files = ["products.csv", "customers.csv", "stores.csv", "orders.csv", "inventory.csv"]
    
    # Ensure sample CSV files exist
    for filename in csv_files:
        assert (SAMPLE_DIR / filename).exists(), f"Missing sample file: {filename}"

    with TestClient(app) as client:
        # Prepare multipart files payload
        open_files = []
        upload_payload = []
        try:
            for filename in csv_files:
                file_obj = open(SAMPLE_DIR / filename, "rb")
                open_files.append(file_obj)
                upload_payload.append(
                    ("files", (filename, file_obj, "text/csv"))
                )

            # POST /runs/upload
            response = client.post("/runs/upload", files=upload_payload)
        finally:
            for f in open_files:
                f.close()

        assert response.status_code == 200, response.text
        data = response.json()

        assert "run_id" in data
        assert data["status"] == "profiling"
        assert len(data["datasets"]) == 5

        # Check row counts match generated datasets
        dataset_rows = {d["name"]: d["row_count"] for d in data["datasets"]}
        assert dataset_rows.get("products") == 60
        assert dataset_rows.get("customers") == 80
        assert dataset_rows.get("stores") == 10
        assert dataset_rows.get("orders") == 300
        assert dataset_rows.get("inventory") == 50

        run_id = data["run_id"]

        # GET /runs/{run_id}
        get_response = client.get(f"/runs/{run_id}")
        assert get_response.status_code == 200, get_response.text
        get_data = get_response.json()

        assert get_data["id"] == run_id
        assert get_data["status"] == "profiling"
        assert len(get_data["datasets"]) == 5

        get_dataset_names = {d["name"] for d in get_data["datasets"]}
        assert get_dataset_names == {"products", "customers", "stores", "orders", "inventory"}
