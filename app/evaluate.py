"""
Headless batch evaluator.

Runs evaluation cases against the SAME service functions the API routes use
(profiling, cleaning, joins, analytics, the chat pipeline) — no server, no
reimplementation. Each case is executed inside its own try/except: a failed
case is recorded with status="error" and the batch continues.

Usage:
    python -m app.evaluate --input cases.json --output results.json \
        --artifacts-dir ./evaluation_output

Case types:
  - "cleaning": upload+profile+clean the listed CSVs; records profile_before,
    cleaning_plan, profile_after, validation.
  - "chat": cleaning pipeline, then a single chat turn in an ephemeral chat
    session. Supported expectations: "min_result_rows", "expected_status".
    Other expectation keys are reported as NOT_YET_SUPPORTED, not ignored.
  - "join": cleaning pipeline, then safe_join on the named datasets; records
    the join report.

Note: the cleaning stage is run through the same functions the API's
/clean endpoint calls (build_cleaning_plan / apply_cleaning_plan /
validate_cleaning / profile_dataframe); datasets are NOT run through the
DB-backed upload endpoint to keep the evaluator stateless and headless.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.services.chat_intent import classify_question, match_analytics_capability
from app.services.chat_session import apply_active_filters_to_plan
from app.services.cleaning import (
    apply_cleaning_plan,
    build_cleaning_plan,
    validate_cleaning,
)
from app.services.joins import get_join_config, safe_join
from app.services.llm_planner import generate_plan
from app.services.plan_executor import execute_plan
from app.services.plan_validator import PlanValidationError, validate_plan
from app.services.profiling import profile_dataframe

SUPPORTED_EXPECTATIONS = {"min_result_rows", "expected_status"}
SUPPORTED_CASE_TYPES = {"cleaning", "chat", "join"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Shared pipeline pieces (same functions the API routes call)
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    dataset_paths: list[dict],
    artifacts_case_dir: Path,
) -> dict:
    """
    Profile + clean the given datasets in memory.

    dataset_paths: [{"name": ..., "path": ...}, ...] — validated by callers.
    Returns {"datasets": {name: df_clean}, "profile_before": {...},
             "cleaning_plan": [...], "profile_after": {...},
             "validation": {...}, "cleaned_files": [...]}
    """
    datasets: dict[str, pd.DataFrame] = {}
    profile_before: dict[str, dict] = {}
    profile_after: dict[str, dict] = {}
    cleaning_plan: dict[str, list] = {}
    validation: dict[str, dict] = {}
    cleaned_files: list[str] = []

    for entry in dataset_paths:
        name = entry["name"]
        df_raw = pd.read_csv(entry["path"])
        datasets[name] = df_raw

        profile_before[name] = profile_dataframe(df_raw)

        plan = build_cleaning_plan(df_raw, name)
        df_clean, updated_plan = apply_cleaning_plan(df_raw, plan)
        profile_after[name] = profile_dataframe(df_clean)
        validation[name] = validate_cleaning(df_raw, df_clean, updated_plan)

        clean_path = artifacts_case_dir / f"{name}_clean.csv"
        df_clean.to_csv(clean_path, index=False)
        cleaned_files.append(str(clean_path))

        cleaning_plan[name] = [step.model_dump() for step in updated_plan]

        # Replace with the clean frame so later stages see cleaned data —
        # same as the API, where chat/query always reads stage="clean".
        datasets[name] = df_clean

    return {
        "datasets": datasets,
        "profile_before": profile_before,
        "cleaning_plan": cleaning_plan,
        "profile_after": profile_after,
        "validation": validation,
        "cleaned_files": cleaned_files,
    }


# ---------------------------------------------------------------------------
# Chat evaluation (same pipeline as the chat router: plan → validate → execute)
# ---------------------------------------------------------------------------

def run_chat_turn_evaluation(
    question: str,
    clean_datasets: dict,
    active_filters: dict | None = None,
) -> dict:
    """
    One chat turn: classify → (capability | plan → validate → execute).
    Mirrors app/routers/chat.py without any DB. Returns a turn record with
    "status", "plan", "answer", "evidence", "result_row_count".
    """
    active_filters = active_filters or {}
    turn: dict = {"question": question}

    capability = match_analytics_capability(question)
    if capability is not None:
        from app.services.chat_analytics import run_analytics_capability

        result, answer_text, region_applied = run_analytics_capability(
            capability, clean_datasets, active_filters
        )
        turn.update(
            {
                "status": "ok",
                "plan": {
                    "capability": capability,
                    "region_filter_applied": region_applied,
                },
                "plan_source": "analytics_capability",
                "answer": answer_text,
                "evidence": result,
                "result_row_count": len(result.get("metrics", {})),
            }
        )
        return turn

    dataset_names = sorted(clean_datasets.keys())
    classification = classify_question(question, dataset_names, active_filters)

    if classification == "unanswerable":
        turn.update({"status": "refused", "answer": "Question is outside the data's scope."})
        return turn
    if classification == "ambiguous":
        turn.update({"status": "clarification_needed", "answer": "Question is too vague to act on."})
        return turn

    plan, plan_source = generate_plan(question=question, available_datasets=dataset_names)
    plan = apply_active_filters_to_plan(plan, active_filters)

    try:
        validated_plan = validate_plan(plan)
    except PlanValidationError as e:
        turn.update(
            {
                "status": "plan_rejected",
                "plan": plan.model_dump(by_alias=True),
                "plan_source": plan_source,
                "answer": f"Plan rejected by validator: {e.message}",
            }
        )
        return turn

    execution_result = execute_plan(validated_plan, clean_datasets)
    turn.update(
        {
            "status": "llm_fallback" if plan_source == "mock_fallback" else "ok",
            "plan": validated_plan.model_dump(by_alias=True),
            "plan_source": plan_source,
            "answer": f"Found {execution_result['result_row_count']} result(s).",
            "evidence": execution_result["evidence"],
            "result_row_count": execution_result["result_row_count"],
        }
    )
    return turn


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------

def _load_dataset_paths(case: dict) -> list[dict]:
    """Validate that every listed CSV exists; raises FileNotFoundError."""
    paths = []
    for raw in case.get("datasets", []):
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {raw}")
        paths.append({"name": p.stem, "path": p})
    return paths


def run_case(case: dict, artifacts_dir: Path) -> dict:
    """
    Runs one evaluation case. Raises on failure — the caller wraps this in
    try/except so one bad case never aborts the batch.
    """
    case_type = case.get("type")
    if case_type not in SUPPORTED_CASE_TYPES:
        raise ValueError(f"Unsupported case type '{case_type}' (supported: {sorted(SUPPORTED_CASE_TYPES)})")

    case_id = case["id"]
    case_dir = artifacts_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "id": case_id,
        "type": case_type,
        "status": "ok",
        "profile_before": None,
        "issues": None,
        "cleaning_plan": None,
        "profile_after": None,
        "validation": None,
        "chat_evaluation": None,
        "join_report": None,
        "error": None,
        "artifacts": {"cleaned_files": []},
    }

    dataset_paths = _load_dataset_paths(case)
    pipeline = run_cleaning_pipeline(dataset_paths, case_dir)
    record["profile_before"] = pipeline["profile_before"]
    record["cleaning_plan"] = pipeline["cleaning_plan"]
    record["profile_after"] = pipeline["profile_after"]
    record["validation"] = pipeline["validation"]

    issues = []
    for name, val in pipeline["validation"].items():
        for issue in val.get("unresolved_issues", []):
            issues.append(f"{name}: {issue}")
    record["issues"] = issues

    record["artifacts"]["cleaned_files"] = pipeline["cleaned_files"]

    if case_type == "chat":
        question = case.get("question")
        if not question:
            raise ValueError("Chat case requires a 'question' field")

        turn = run_chat_turn_evaluation(
            question=question,
            clean_datasets=pipeline["datasets"],
        )
        record["chat_evaluation"] = [turn]

        expected = case.get("expected") or {}
        unsupported = sorted(set(expected.keys()) - SUPPORTED_EXPECTATIONS)
        if unsupported:
            record["unsupported_expectations"] = {
                key: "NOT_YET_SUPPORTED" for key in unsupported
            }

        if "expected_status" in expected and turn["status"] != expected["expected_status"]:
            raise AssertionError(
                f"expected_status={expected['expected_status']} but turn status was '{turn['status']}'"
            )
        if "min_result_rows" in expected:
            observed = turn.get("result_row_count", 0)
            if observed < expected["min_result_rows"]:
                raise AssertionError(
                    f"min_result_rows={expected['min_result_rows']} but only {observed} row(s) returned"
                )

    elif case_type == "join":
        left, right = case.get("left"), case.get("right")
        if not left or not right:
            raise ValueError("Join case requires 'left' and 'right' dataset names")
        cfg = get_join_config(left, right)
        joined_df, report = safe_join(
            left_df=pipeline["datasets"][left],
            right_df=pipeline["datasets"][right],
            left_key=cfg["left_key"],
            right_key=cfg["right_key"],
            how=case.get("how", "left"),
        )
        record["join_report"] = report
        joined_path = case_dir / f"{left}_JOIN_{right}.csv"
        joined_df.head(200).to_csv(joined_path, index=False)
        record["artifacts"]["cleaned_files"].append(str(joined_path))

    return record


def evaluate(input_path: Path, output_path: Path, artifacts_dir: Path) -> dict:
    """Runs all cases; a per-case failure is recorded and never aborts the batch."""
    results: dict = {
        "run_metadata": {
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
        },
        "cases": [],
    }

    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Cannot read input file '{input_path}': {e}")

    cases = raw["cases"] if isinstance(raw, dict) else raw
    results["run_metadata"]["total_cases"] = len(cases)

    for case in cases:
        case_id = case.get("id", f"case_{len(results['cases']) + 1}")
        try:
            record = run_case(case, artifacts_dir)
        except Exception as e:
            record = {
                "id": case_id,
                "type": case.get("type"),
                "status": "error",
                "profile_before": None,
                "issues": None,
                "cleaning_plan": None,
                "profile_after": None,
                "validation": None,
                "chat_evaluation": None,
                "join_report": None,
                "error": f"{type(e).__name__}: {e}",
                "artifacts": {"cleaned_files": []},
            }
            if artifacts_dir is not None:
                tb_dir = artifacts_dir / case_id
                tb_dir.mkdir(parents=True, exist_ok=True)
                (tb_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        results["cases"].append(record)

    results["run_metadata"]["passed"] = sum(
        1 for c in results["cases"] if c["status"] == "ok"
    )
    results["run_metadata"]["failed"] = sum(
        1 for c in results["cases"] if c["status"] == "error"
    )
    results["run_metadata"]["finished_at"] = _utc_now_iso()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluate",
        description="Headless batch evaluator for the retail data workbench.",
    )
    parser.add_argument("--input", required=True, help="Path to the evaluation cases JSON file")
    parser.add_argument("--output", required=True, help="Path to write the results JSON")
    parser.add_argument(
        "--artifacts-dir", required=True, help="Directory for per-case artifacts (cleaned CSVs, tracebacks)"
    )
    args = parser.parse_args(argv)

    started = _utc_now_iso()
    results = evaluate(Path(args.input), Path(args.output), Path(args.artifacts_dir))
    results["run_metadata"]["started_at"] = started

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    meta = results["run_metadata"]
    print(
        f"Evaluation complete: {meta['passed']}/{meta['total_cases']} cases passed. "
        f"Results written to {output_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
