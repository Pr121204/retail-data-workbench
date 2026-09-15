"""Background execution mode for long-running pipeline stages.

POST /runs/{id}/profile|clean?wait=false returns 202 with a "running_*" state,
executes with a dedicated DB session, and — on failure — marks the run
"failed" with a persisted error_message instead of leaving it stuck.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.runs import _clean_impl, _profile_impl, _run_background_stage


def _upload_and_profile(client):
    files = [("files", ("products.csv", b"id,category\nP1,Toys\nP2,Books\n", "text/csv"))]
    run_id = client.post("/runs/upload", files=files).json()["run_id"]
    client.post(f"/runs/{run_id}/profile")
    return run_id


def test_clean_background_mode_returns_202_and_completes():
    with TestClient(app) as client:
        run_id = _upload_and_profile(client)

        resp = client.post(f"/runs/{run_id}/clean?wait=false")
        assert resp.status_code == 202
        assert resp.json()["status"] == "running_clean"

        # BackgroundTasks run synchronously in TestClient after the response;
        # by the time post() returns, the task has completed.
        final = client.get(f"/runs/{run_id}").json()
        assert final["status"] == "cleaned"
        clean_ds = [d for d in final["datasets"] if d["stage"] == "clean"]
        assert len(clean_ds) == 1


def test_profile_background_mode_returns_202_and_completes():
    with TestClient(app) as client:
        run_id = client.post(
            "/runs/upload",
            files=[("files", ("products.csv", b"id,category\nP1,Toys\n", "text/csv"))],
        ).json()["run_id"]

        resp = client.post(f"/runs/{run_id}/profile?wait=false")
        assert resp.status_code == 202
        assert resp.json()["status"] == "running_profile"
        assert client.get(f"/runs/{run_id}").json()["status"] == "profiled"


def test_background_failure_marks_run_failed_with_persisted_error():
    with TestClient(app) as client:
        run_id = _upload_and_profile(client)

        # Corrupt impl: _clean_impl raises -> _run_background_stage must mark
        # the run failed instead of leaving it stuck in running_clean.
        def broken(run_id, db):
            raise RuntimeError("boom")

        _run_background_stage("clean", run_id, broken)
        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "failed"
        assert "boom" in run["error_message"]


def test_impl_functions_raise_on_wrong_state():
    from app.db import SessionLocal

    with TestClient(app) as client:
        run_id = _upload_and_profile(client)
        db = SessionLocal()
        try:
            _clean_impl(run_id, db)  # first clean succeeds (status profiled)
            with pytest.raises(ValueError, match="cleanable"):
                _clean_impl(run_id, db)  # second clean -> wrong state
            with pytest.raises(ValueError, match="profileable"):
                _profile_impl(run_id, db)  # status cleaned -> not profileable
            with pytest.raises(ValueError):
                _profile_impl("no-such-run", db)
        finally:
            db.close()
