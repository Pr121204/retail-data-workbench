"""
Chat layer: multi-turn Q&A sessions on top of the existing single-shot pipeline
(generate_plan -> validate_plan -> execute_plan). Those three functions are
treated as a black box — this layer only adds intent gating, context
carry-forward, and persistence of turns.
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.chat import ChatSession, ChatTurn
from app.models.run import Dataset, Run
from app.services.chat_intent import classify_question, match_analytics_capability
from app.services.csv_io import read_csv_robust
from app.services.chat_analytics import run_analytics_capability
from app.services.chat_session import (
    apply_active_filters_to_plan,
    extract_filters_from_question,
    merge_active_filters,
)
from app.services.llm_planner import generate_plan
from app.services.plan_executor import execute_plan
from app.services.plan_validator import PlanValidationError, validate_plan
from app.routers.runs import _require_run_owner

router = APIRouter(prefix="/runs", tags=["chat"])


class CreateSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _turn_to_dict(turn: ChatTurn) -> dict:
    """Full serialisation of a ChatTurn row, JSON text columns decoded."""
    return {
        "turn_index": turn.turn_index,
        "status": turn.status,
        "question": turn.question,
        "plan": json.loads(turn.plan_json) if turn.plan_json else None,
        "plan_source": turn.plan_source,
        "answer_text": turn.answer_text,
        "evidence": json.loads(turn.evidence_json) if turn.evidence_json else None,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


def _load_clean_datasets(db: Session, run: Run) -> dict:
    """Load all clean DataFrames for a run (mirrors the /query endpoint)."""
    import pandas as pd

    clean_datasets = {}
    for ds in run.datasets:
        if ds.stage == "clean":
            clean_datasets[ds.name] = read_csv_robust(ds.file_path)
    return clean_datasets


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{run_id}/chat/sessions",
    status_code=status.HTTP_200_OK,
    response_model=CreateSessionResponse,
)
def create_chat_session(
    run_id: str,
    db: Session = Depends(get_db),
    x_run_owner: Optional[str] = Header(None, alias="X-Run-Owner"),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    _require_run_owner(run, x_run_owner)

    session = ChatSession(run_id=run.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    return {"session_id": session.id}


@router.post(
    "/{run_id}/chat/sessions/{session_id}/turns",
    status_code=status.HTTP_200_OK,
)
def create_chat_turn(
    run_id: str,
    session_id: str,
    request: TurnRequest,
    db: Session = Depends(get_db),
    x_run_owner: Optional[str] = Header(None, alias="X-Run-Owner"),
):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'question' must be a non-empty string",
        )

    # Load the session (404 if not found or run_id mismatch)
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.run_id == run_id)
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found for run '{run_id}'",
        )

    # The run must have clean datasets to answer anything
    run = db.query(Run).filter(Run.id == run_id).first()
    if run:
        _require_run_owner(run, x_run_owner)
    clean_datasets = _load_clean_datasets(db, run) if run else {}
    dataset_names = sorted(clean_datasets.keys())

    turn_index = len(session.turns) + 1

    # The turn row is persisted in every branch — refusals and clarifications
    # are part of the conversation history too.
    turn = ChatTurn(
        session_id=session.id,
        turn_index=turn_index,
        question=question,
        status="ok",
    )

    # ---- Gate 0: analytics capability routing ----
    # Domain questions (return rate, revenue by category, store performance,
    # stockouts) need business logic the generic QueryPlan pipeline can't
    # express, so they bypass generate_plan/validate_plan/execute_plan
    # entirely and call analytics.py directly.
    active_filters = json.loads(session.active_filters_json or "{}")
    capability = match_analytics_capability(question)
    if capability is not None:
        try:
            result, answer_text, region_applied = run_analytics_capability(
                capability, clean_datasets, active_filters
            )
        except ValueError as e:
            # Missing datasets etc. — a normal, modelled outcome (200).
            turn.status = "plan_rejected"
            turn.plan_json = json.dumps({"capability": capability, "error": str(e)})
            turn.plan_source = "analytics_capability"
            turn.answer_text = str(e)
            db.add(turn)
            db.commit()
            db.refresh(turn)
            return _turn_to_dict(turn)

        turn.status = "ok"
        turn.plan_json = json.dumps(
            {"capability": capability, "region_filter_applied": region_applied}
        )
        turn.plan_source = "analytics_capability"
        turn.answer_text = answer_text
        result["lineage"] = {
            "run_id": run.id,
            "dataset_stage": "clean",
            "dataset_version": f"run:{run.id}:clean",
        }
        turn.evidence_json = json.dumps(result)
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return _turn_to_dict(turn)

    # ---- Gate 1: lightweight intent classification BEFORE any planning ----
    classification = classify_question(question, dataset_names, active_filters)

    if classification == "unanswerable":
        turn.status = "refused"
        turn.answer_text = (
            "I can't answer that question: it asks about something outside the "
            "uploaded data (competitors, weather, external market data, or a "
            "future forecast). I can only answer questions about the run's "
            "clean datasets: " + (", ".join(dataset_names) if dataset_names else "(none)")
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return _turn_to_dict(turn)

    if classification == "ambiguous":
        turn.status = "clarification_needed"
        turn.answer_text = (
            "Could you clarify the question? Please name the dataset or field "
            "you're interested in (e.g. 'orders in the West region' or "
            "'revenue by category') so I can build a concrete query."
        )
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return _turn_to_dict(turn)

    # ---- Answerable: plan -> validate -> execute ----
    try:
        plan, plan_source = generate_plan(question=question, available_datasets=dataset_names)
    except Exception as e:
        turn.status = "plan_rejected"
        turn.answer_text = f"Could not generate a query plan for this question: {e}"
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return _turn_to_dict(turn)

    # Carry forward context from earlier turns, then merge in any explicit
    # constraint the question itself names (e.g. "in the West region").
    plan = apply_active_filters_to_plan(plan, active_filters)
    for f in extract_filters_from_question(question):
        if f.field not in {pf.field for pf in plan.filters}:
            plan.filters.append(f)

    try:
        validated_plan = validate_plan(plan)
    except PlanValidationError as e:
        # Deliberately 200: a rejected plan is a normal, modelled outcome.
        turn.status = "plan_rejected"
        turn.plan_json = json.dumps(plan.model_dump(by_alias=True))
        turn.plan_source = plan_source
        turn.answer_text = f"Generated plan was rejected by the validator: {e.message}"
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return _turn_to_dict(turn)

    execution_result = execute_plan(
        validated_plan,
        clean_datasets,
        evidence_context={
            "run_id": run.id,
            "dataset_stage": "clean",
            "dataset_version": f"run:{run.id}:clean",
        },
    )

    # Update carried-over context for the next turn
    session.active_filters_json = json.dumps(
        merge_active_filters(active_filters, validated_plan.filters)
    )

    # Grounded answer text: plain string formatting from the computed result.
    result_rows = execution_result["result_rows"]
    row_count = execution_result["result_row_count"]
    execution_error = execution_result.get("execution_error")

    if execution_error:
        answer_text = f"The query could not be executed: {execution_error}"
    elif row_count == 0:
        answer_text = "No results matched the query."
    else:
        top_row = result_rows[0]
        top_summary = ", ".join(f"{k}={v}" for k, v in list(top_row.items())[:5])
        answer_text = f"Found {row_count} result(s). Top result: {top_summary}"

    turn.status = "llm_fallback" if plan_source == "mock_fallback" else "ok"
    turn.plan_json = json.dumps(validated_plan.model_dump(by_alias=True))
    turn.plan_source = plan_source
    turn.answer_text = answer_text
    turn.evidence_json = json.dumps(execution_result["evidence"])

    db.add(turn)
    db.commit()
    db.refresh(turn)
    return _turn_to_dict(turn)


@router.get("/{run_id}/chat/sessions/{session_id}", status_code=status.HTTP_200_OK)
def get_chat_session(
    run_id: str,
    session_id: str,
    db: Session = Depends(get_db),
    x_run_owner: Optional[str] = Header(None, alias="X-Run-Owner"),
):
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.run_id == run_id)
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found for run '{run_id}'",
        )
    run = db.query(Run).filter(Run.id == run_id).first()
    if run:
        _require_run_owner(run, x_run_owner)

    return {
        "session_id": session.id,
        "run_id": session.run_id,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "active_filters": json.loads(session.active_filters_json or "{}"),
        "turns": [_turn_to_dict(t) for t in session.turns],
    }
