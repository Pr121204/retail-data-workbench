import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.evaluate import run_case

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_evaluator_batch_survives_broken_case(tmp_path):
    output_path = tmp_path / "results.json"
    artifacts_dir = tmp_path / "artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.evaluate",
            "--input",
            "data/samples/evaluation_cases.json",
            "--output",
            str(output_path),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # The deliberately broken case must NOT abort the batch
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Evaluation complete:" in result.stdout
    # The intentional demo failure must be called out in the CLI output so a
    # bare "5/6 cases passed" is never mistaken for a defect.
    assert "intentional failure-isolation demo" in result.stdout

    assert output_path.exists(), "results.json was not written"
    data = json.loads(output_path.read_text(encoding="utf-8"))

    meta = data["run_metadata"]
    assert meta["total_cases"] == 9
    assert meta["passed"] == 8
    assert meta["failed"] == 1
    assert meta["intentional_failures"] == ["case_006_broken"]
    assert meta["started_at"] and meta["finished_at"]

    cases = data["cases"]
    assert len(cases) == 9

    broken = [c for c in cases if c["status"] == "error"]
    assert len(broken) == 1
    broken = broken[0]
    assert broken["id"] == "case_006_broken"
    assert broken["intentional_failure"] is True
    assert "does_not_exist" in broken["error"]
    assert "FileNotFoundError" in broken["error"]

    ok_cases = [c for c in cases if c["status"] == "ok"]
    assert len(ok_cases) == 8

    # Per-case artifact dirs were created (including the edge cases)
    for case_id in ["case_001", "case_002", "case_003", "case_006_broken", "case_007_all_null_column", "case_009_large_file"]:
        assert (artifacts_dir / case_id).is_dir(), f"missing artifacts dir for {case_id}"

    # Broken case gets a traceback artifact for debugging
    assert (artifacts_dir / "case_006_broken" / "traceback.txt").exists()

    # The ambiguous chat case exercised the refusal path without crashing
    ambiguous = next(c for c in cases if c["id"] == "case_005_chat_ambiguous")
    assert ambiguous["status"] == "ok"
    assert ambiguous["chat_evaluation"][0]["status"] == "clarification_needed"

    # The revenue chat case passed its expectations
    revenue = next(c for c in cases if c["id"] == "case_002")
    assert revenue["status"] == "ok"
    rev_turn = revenue["chat_evaluation"][0]
    assert rev_turn["status"] == "ok"
    assert rev_turn["result_row_count"] >= 1

    # Edge cases: all-null column survives; large file is fully processed.
    allnull = next(c for c in cases if c["id"] == "case_007_all_null_column")
    assert allnull["status"] == "ok"
    assert allnull["validation"]["allnull"]["row_count_after"] == 20

    large = next(c for c in cases if c["id"] == "case_009_large_file")
    assert large["status"] == "ok"
    large_turn = large["chat_evaluation"][0]
    assert large_turn["evidence"]["row_count_before_filter"] == 100000


def test_evaluator_rejects_unsupported_expectations(tmp_path):
    case = {
        "id": "unsupported",
        "type": "chat",
        "datasets": ["data/samples/orders.csv"],
        "question": "What is the average order value?",
        "expected": {"unknown_contract": True},
    }

    with pytest.raises(ValueError, match="Unsupported chat expectations"):
        run_case(case, tmp_path / "artifacts")


def test_evaluator_enforces_multiturn_filters_and_evidence(tmp_path):
    case = {
        "id": "multiturn-contract",
        "type": "chat",
        "datasets": [
            "data/samples/orders.csv",
            "data/samples/products.csv",
        ],
        "questions": [
            "Show me orders in the West region",
            "Which categories had the highest return rate among those?",
        ],
        "expected_statuses": ["ok", "ok"],
        "expected": {
            "expected_status": "ok",
            "answer_contains": ["highest return rate"],
            "evidence_keys": ["metrics", "assumption", "lineage"],
            "evidence_equals": {"lineage": {"dataset_stage": "clean", "dataset_version": "clean:evaluator"}},
        },
    }

    record = run_case(case, tmp_path / "artifacts")

    assert record["status"] == "ok"
    assert len(record["chat_evaluation"]) == 2
    assert record["chat_evaluation"][1]["plan"]["region_filter_applied"] is True
