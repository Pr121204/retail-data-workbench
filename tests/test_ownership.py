"""Ownership isolation (data-isolation boundary).

Runs uploaded with an X-Run-Owner token reject requests without (or with a
different) token — 403. Runs uploaded without a token stay publicly
addressable (single-tenant demo mode, backward compatible).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


def _upload(client, owner_header=None):
    files = [("files", ("products.csv", b"id,category\nP1,Toys\nP2,Books\n", "text/csv"))]
    headers = {"X-Run-Owner": owner_header} if owner_header else None
    return client.post("/runs/upload", files=files, headers=headers)


def test_upload_without_token_stays_public():
    with TestClient(app) as client:
        resp = _upload(client)
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # All run-scoped endpoints reachable without any owner header.
        assert client.get(f"/runs/{run_id}").status_code == 200
        assert client.get(f"/runs/{run_id}/analytics").status_code == 400  # not cleaned yet, NOT 403


def test_owned_run_rejects_missing_and_wrong_token():
    with TestClient(app) as client:
        resp = _upload(client, owner_header="alice")
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        # No token -> 403
        assert client.get(f"/runs/{run_id}").status_code == 403
        # Wrong token -> 403
        assert client.get(f"/runs/{run_id}", headers={"X-Run-Owner": "mallory"}).status_code == 403
        # Correct token -> 200
        assert client.get(f"/runs/{run_id}", headers={"X-Run-Owner": "alice"}).status_code == 200

        # Ownership applies to chat session creation too.
        assert client.post(f"/runs/{run_id}/chat/sessions").status_code == 403
        ok = client.post(f"/runs/{run_id}/chat/sessions", headers={"X-Run-Owner": "alice"})
        assert ok.status_code == 200


def test_owned_run_rejects_token_on_mutation_endpoints():
    with TestClient(app) as client:
        run_id = _upload(client, owner_header="alice").json()["run_id"]

        assert client.post(f"/runs/{run_id}/profile").status_code == 403
        assert (
            client.post(f"/runs/{run_id}/profile", headers={"X-Run-Owner": "alice"}).status_code
            == 200
        )
