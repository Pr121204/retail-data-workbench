import json
import subprocess
import sys
from pathlib import Path

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
    assert meta["total_cases"] == 6
    assert meta["passed"] == 5
    assert meta["failed"] == 1
    assert meta["intentional_failures"] == ["case_006_broken"]
    assert meta["started_at"] and meta["finished_at"]

    cases = data["cases"]
    assert len(cases) == 6

    broken = [c for c in cases if c["status"] == "error"]
    assert len(broken) == 1
    broken = broken[0]
    assert broken["id"] == "case_006_broken"
    assert broken["intentional_failure"] is True
    assert "does_not_exist" in broken["error"]
    assert "FileNotFoundError" in broken["error"]

    ok_cases = [c for c in cases if c["status"] == "ok"]
    assert len(ok_cases) == 5

    # Per-case artifact dirs were created
    for case_id in ["case_001", "case_002", "case_003", "case_006_broken"]:
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
