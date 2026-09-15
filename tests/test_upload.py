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


def test_upload_rejects_non_csv_and_duplicate_dataset_names():
    with TestClient(app) as client:
        non_csv = client.post(
            "/runs/upload",
            files=[("files", ("data.txt", b"not csv", "text/plain"))],
        )
        assert non_csv.status_code == 415

        duplicate = client.post(
            "/runs/upload",
            files=[
                ("files", ("orders.csv", b"id\n1\n", "text/csv")),
                ("files", ("orders.csv", b"id\n2\n", "text/csv")),
            ],
        )
        assert duplicate.status_code == 400
        assert "Duplicate" in duplicate.json()["detail"]


def test_upload_rejects_unreadable_csv_before_creating_run():
    with TestClient(app) as client:
        response = client.post(
            "/runs/upload",
            files=[("files", ("broken.csv", b"id,name\n1,\"unterminated\n", "text/csv"))],
        )

    assert response.status_code == 422
    assert "not a readable CSV" in response.json()["detail"]


def test_upload_enforces_file_count_and_byte_limits(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MAX_UPLOAD_FILES", 1)
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 4)
    with TestClient(app) as client:
        too_many = client.post(
            "/runs/upload",
            files=[
                ("files", ("one.csv", b"id\n1\n", "text/csv")),
                ("files", ("two.csv", b"id\n2\n", "text/csv")),
            ],
        )
        too_large = client.post(
            "/runs/upload",
            files=[("files", ("large.csv", b"id\n1\n", "text/csv"))],
        )

    assert too_many.status_code == 413
    assert too_large.status_code == 413
