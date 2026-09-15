import io
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.config import settings
from app.models.run import Dataset, Run
from app.services.analytics import compute_all_analytics
from app.services.cleaning import apply_cleaning_plan, build_cleaning_plan, validate_cleaning
from app.services.joins import get_join_config, safe_join
from app.services.llm_planner import generate_plan
from app.services.plan_executor import execute_plan
from app.services.plan_validator import PlanValidationError, validate_plan
from app.services.profiling import profile_dataframe

router = APIRouter(prefix="/runs", tags=["runs"])

BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "runs"


class JoinRequest(BaseModel):
    left: str
    right: str
    how: str = "left"


class QueryRequest(BaseModel):
    question: str


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_runs(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one CSV file is required",
        )
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"At most {settings.MAX_UPLOAD_FILES} files may be uploaded per run",
        )

    prepared_files = []
    dataset_names = set()
    total_bytes = 0
    for file in files:
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".csv":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{file.filename or '(unnamed)'}' must have a .csv extension",
            )
        dataset_name = Path(filename).stem
        if not dataset_name or dataset_name in dataset_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Duplicate or empty dataset name for file '{filename}'",
            )
        dataset_names.add(dataset_name)

        content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
        if len(content) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{filename}' exceeds the {settings.MAX_UPLOAD_BYTES} byte upload limit",
            )
        total_bytes += len(content)
        if total_bytes > settings.MAX_UPLOAD_BYTES * settings.MAX_UPLOAD_FILES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Total uploaded content exceeds the configured limit",
            )
        try:
            pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{filename}' is not a readable CSV: {exc}",
            )
        prepared_files.append((filename, dataset_name, content))

    run = Run(status="pending")
    db.add(run)
    db.flush()  # populate run.id

    raw_dir = BASE_DATA_DIR / run.id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []

    for filename, dataset_name, content in prepared_files:
        file_path = raw_dir / filename

        # Write uploaded file content to disk
        with open(file_path, "wb") as f:
            f.write(content)

        # Compute row and column count using pandas
        try:
            df = pd.read_csv(file_path)
            row_count = int(len(df))
            column_count = int(len(df.columns))
        except Exception:
            row_count = None
            column_count = None

        dataset = Dataset(
            run_id=run.id,
            name=dataset_name,
            stage="raw",
            file_path=str(file_path),
            row_count=row_count,
            column_count=column_count,
        )
        db.add(dataset)
        dataset_summaries.append(
            {
                "name": dataset_name,
                "row_count": row_count,
                "column_count": column_count,
            }
        )

    run.status = "profiling"
    db.commit()
    db.refresh(run)

    return {
        "run_id": run.id,
        "status": run.status,
        "datasets": dataset_summaries,
    }


@router.post("/{run_id}/profile", status_code=status.HTTP_200_OK)
def profile_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    if run.status != "profiling":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run status must be 'profiling' to profile, but is '{run.status}'",
        )

    profiles = {}
    for ds in run.datasets:
        if ds.stage == "raw":
            try:
                df = pd.read_csv(ds.file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to read dataset file at '{ds.file_path}': {e}",
                )
            profile = profile_dataframe(df)
            ds.profile_json = json.dumps(profile)
            profiles[ds.name] = profile

    run.status = "profiled"
    db.commit()
    db.refresh(run)

    return {
        "run_id": run.id,
        "status": "profiled",
        "profiles": profiles,
    }


@router.post("/{run_id}/clean", status_code=status.HTTP_200_OK)
def clean_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    if run.status != "profiled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run status must be 'profiled' to clean, but is '{run.status}'",
        )

    clean_dir = BASE_DATA_DIR / run.id / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    raw_datasets = [ds for ds in run.datasets if ds.stage == "raw"]

    for ds in raw_datasets:
        try:
            df_raw = pd.read_csv(ds.file_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read dataset '{ds.name}' at '{ds.file_path}': {e}",
            )

        plan = build_cleaning_plan(df_raw, ds.name)
        df_clean, updated_plan = apply_cleaning_plan(df_raw, plan)

        clean_csv_path = clean_dir / f"{ds.name}.csv"
        df_clean.to_csv(clean_csv_path, index=False)

        clean_profile = profile_dataframe(df_clean)
        val_result = validate_cleaning(df_raw, df_clean, updated_plan)

        plan_dicts = [step.model_dump() for step in updated_plan]

        clean_ds = Dataset(
            run_id=run.id,
            name=ds.name,
            stage="clean",
            file_path=str(clean_csv_path),
            row_count=int(len(df_clean)),
            column_count=int(len(df_clean.columns)),
            profile_json=json.dumps(clean_profile),
            cleaning_plan_json=json.dumps(plan_dicts),
            validation_json=json.dumps(val_result),
        )
        db.add(clean_ds)

        results[ds.name] = {
            "plan": plan_dicts,
            "validation": val_result,
        }

    run.status = "cleaned"
    db.commit()
    db.refresh(run)

    return {
        "run_id": run.id,
        "status": "cleaned",
        "results": results,
    }


@router.get("/{run_id}", status_code=status.HTTP_200_OK)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run with id '{run_id}' not found",
        )

    return {
        "id": run.id,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "error_message": run.error_message,
        "datasets": [
            {
                "id": ds.id,
                "run_id": ds.run_id,
                "name": ds.name,
                "stage": ds.stage,
                "file_path": ds.file_path,
                "row_count": ds.row_count,
                "column_count": ds.column_count,
                "profile_json": json.loads(ds.profile_json) if ds.profile_json else None,
                "cleaning_plan_json": json.loads(ds.cleaning_plan_json) if ds.cleaning_plan_json else None,
                "validation_json": json.loads(ds.validation_json) if ds.validation_json else None,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
            }
            for ds in run.datasets
        ],
    }


@router.post("/{run_id}/join", status_code=status.HTTP_200_OK)
def join_datasets(run_id: str, request: JoinRequest, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )

    # Check status is "cleaned" or later
    if run.status not in ["cleaned", "validated"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run status must be 'cleaned' or later to perform joins, but is '{run.status}'",
        )
    if request.how not in {"left", "inner"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Join mode must be 'left' or 'inner'",
        )

    # Validate join configuration via safety registry
    try:
        join_cfg = get_join_config(request.left, request.right)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Look up clean datasets
    left_ds = next((ds for ds in run.datasets if ds.stage == "clean" and ds.name == request.left), None)
    right_ds = next((ds for ds in run.datasets if ds.stage == "clean" and ds.name == request.right), None)

    if not left_ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clean dataset '{request.left}' not found in run '{run_id}'",
        )
    if not right_ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clean dataset '{request.right}' not found in run '{run_id}'",
        )

    try:
        left_df = pd.read_csv(left_ds.file_path)
        right_df = pd.read_csv(right_ds.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset file(s): {e}",
        )

    try:
        joined_df, report = safe_join(
            left_df=left_df,
            right_df=right_df,
            left_key=join_cfg["left_key"],
            right_key=join_cfg["right_key"],
            how=request.how,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    # Preview: first 20 rows, JSON-safe (NaN converted to null/None)
    preview_df = joined_df.head(20).copy()
    preview = [
        {col: (None if pd.isna(val) else val) for col, val in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    return {
        "report": report,
        "preview": preview,
    }


@router.get("/{run_id}/analytics", status_code=status.HTTP_200_OK)
def get_run_analytics(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )

    if run.status not in ["cleaned", "validated"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run status must be 'cleaned' or later to compute analytics, but is '{run.status}'",
        )

    clean_datasets = {}
    for ds in run.datasets:
        if ds.stage == "clean":
            try:
                clean_datasets[ds.name] = pd.read_csv(ds.file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to read clean dataset '{ds.name}' at '{ds.file_path}': {e}",
                )

    analytics_results = compute_all_analytics(clean_datasets)
    return analytics_results


@router.post("/{run_id}/query", status_code=status.HTTP_200_OK)
def query_run(
    run_id: str,
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Accepts a natural language question, generates a QueryPlan via the LLM
    (or mock), validates it, executes it against the run's clean datasets,
    and returns results with a full evidence trail.
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )

    if run.status not in ["cleaned", "validated"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Run status must be 'cleaned' or 'validated' to query, "
                f"but is '{run.status}'"
            ),
        )

    # Load all clean datasets for this run
    clean_datasets: dict[str, pd.DataFrame] = {}
    for ds in run.datasets:
        if ds.stage == "clean":
            try:
                clean_datasets[ds.name] = pd.read_csv(ds.file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to read clean dataset '{ds.name}': {e}",
                )

    if not clean_datasets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No clean datasets found for this run. Run /clean first.",
        )

    available_datasets = list(clean_datasets.keys())

    # Step 1: Generate plan (LLM or mock)
    try:
        plan, plan_source = generate_plan(
            question=request.question,
            available_datasets=available_datasets,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Plan generation failed: {e}",
        )

    # Step 2: Validate plan against the allow-list
    try:
        validated_plan = validate_plan(plan)
    except PlanValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Plan validation failed",
                "error_code": e.code,
                "error_message": e.message,
                "generated_plan": plan.model_dump(by_alias=True),
                "plan_source": plan_source,
            },
        )

    # Step 3: Execute the validated plan
    execution_result = execute_plan(
        validated_plan,
        clean_datasets,
        evidence_context={
            "run_id": run.id,
            "dataset_stage": "clean",
            "dataset_version": f"run:{run.id}:clean",
        },
    )

    return {
        "question": request.question,
        "plan_source": plan_source,
        "plan": validated_plan.model_dump(by_alias=True),
        "result_rows": execution_result["result_rows"],
        "result_row_count": execution_result["result_row_count"],
        "evidence": execution_result["evidence"],
        "execution_error": execution_result.get("execution_error"),
    }
