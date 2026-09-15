from typing import Any, Dict, List, Tuple
import pandas as pd
from pydantic import BaseModel


class CleaningStep(BaseModel):
    step_id: str
    reason: str
    affected_fields: List[str]
    risk: str  # "low", "medium", or "high"
    source: str = "code"
    status: str = "proposed"  # "proposed", "applied", "skipped", "failed"
    rows_affected: int = 0


def is_text_column(series: pd.Series) -> bool:
    """Returns True if series is string or object dtype."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def build_cleaning_plan(df: pd.DataFrame, dataset_name: str) -> List[CleaningStep]:
    """
    Inspects the dataframe and proposes deterministic cleaning steps.
    """
    steps: List[CleaningStep] = []

    # 1. Check for leading/trailing whitespace in string columns
    ws_cols = []
    for col in df.columns:
        if is_text_column(df[col]):
            non_null_str = df[col].dropna().astype(str)
            if len(non_null_str) > 0:
                if bool((non_null_str.str.strip() != non_null_str).any()):
                    ws_cols.append(str(col))

    if ws_cols:
        steps.append(
            CleaningStep(
                step_id="trim_whitespace",
                reason=f"Trim leading/trailing whitespace in string columns: {', '.join(ws_cols)}",
                affected_fields=ws_cols,
                risk="low",
                source="code",
                status="proposed",
            )
        )

    # 2. Check for case-inconsistent categorical columns
    for col in df.columns:
        if is_text_column(df[col]):
            non_null = df[col].dropna().astype(str)
            if len(non_null) > 0:
                raw_unique_count = non_null.nunique()
                cleaned_unique_count = non_null.str.strip().str.lower().nunique()
                if cleaned_unique_count < raw_unique_count:
                    col_lower = str(col).lower()
                    is_categorical = (
                        any(kw in col_lower for kw in ["category", "region", "brand", "status", "type", "city"])
                        or (cleaned_unique_count / max(len(df), 1) <= 0.5)
                        or cleaned_unique_count <= 50
                    )
                    if is_categorical:
                        steps.append(
                            CleaningStep(
                                step_id=f"normalize_case_{col}",
                                reason=f"Normalize inconsistent casing and whitespace in categorical column '{col}'",
                                affected_fields=[str(col)],
                                risk="low",
                                source="code",
                                status="proposed",
                            )
                        )

    # 3. Numeric columns stored as object/string
    for col in df.columns:
        col_lower = str(col).lower()
        if "date" not in col_lower and any(kw in col_lower for kw in ["price", "revenue", "quantity", "stock"]):
            if is_text_column(df[col]):
                steps.append(
                    CleaningStep(
                        step_id=f"coerce_numeric_{col}",
                        reason=f"Coerce values in numeric column '{col}' to numbers, setting unparseable strings to NaN",
                        affected_fields=[str(col)],
                        risk="medium",
                        source="code",
                        status="proposed",
                    )
                )

    # 4. Date columns
    for col in df.columns:
        if "date" in str(col).lower():
            steps.append(
                CleaningStep(
                    step_id=f"parse_dates_{col}",
                    reason=f"Parse dates in '{col}' and normalize to ISO YYYY-MM-DD",
                    affected_fields=[str(col)],
                    risk="medium",
                    source="code",
                    status="proposed",
                )
            )

    # 5. Fully duplicate rows
    if bool(df.duplicated().any()):
        dup_count = int(df.duplicated().sum())
        steps.append(
            CleaningStep(
                step_id="dedupe_exact_rows",
                reason=f"Remove {dup_count} exact duplicate row(s)",
                affected_fields=[str(c) for c in df.columns],
                risk="low",
                source="code",
                status="proposed",
            )
        )

    # 6. Duplicate IDs that look like data-entry errors (uniqueness ratio >= 80% or dataset primary key)
    for col in df.columns:
        col_str = str(col).lower()
        if col_str.endswith("_id") or col_str == "sku":
            dups = int(df[col].dropna().duplicated().sum())
            if dups > 0:
                cardinality_ratio = df[col].nunique() / max(len(df), 1)
                is_primary_id = col_str.startswith(dataset_name.rstrip("s").lower())
                if is_primary_id or cardinality_ratio >= 0.8:
                    steps.append(
                        CleaningStep(
                            step_id=f"dedupe_duplicate_ids_{col}",
                            reason=f"Flag {dups} duplicate ID value(s) in '{col}' for manual review",
                            affected_fields=[str(col)],
                            risk="high",
                            source="code",
                            status="proposed",
                        )
                    )

    return steps


def apply_cleaning_plan(df: pd.DataFrame, plan: List[CleaningStep]) -> Tuple[pd.DataFrame, List[CleaningStep]]:
    """
    Executes each cleaning step in order defensively.
    """
    df_clean = df.copy()

    for step in plan:
        try:
            if step.step_id == "trim_whitespace":
                rows_affected = 0
                for col in step.affected_fields:
                    if col in df_clean.columns:
                        s = df_clean[col].dropna().astype(str)
                        has_ws = df_clean[col].notna() & (s.str.strip() != s)
                        rows_affected += int(has_ws.sum())
                        df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                step.rows_affected = rows_affected
                step.status = "applied"

            elif step.step_id.startswith("normalize_case_"):
                rows_affected = 0
                for col in step.affected_fields:
                    if col in df_clean.columns:
                        def norm_case(val):
                            if isinstance(val, str):
                                return val.strip().lower().title()
                            return val

                        s = df_clean[col].dropna().astype(str)
                        changed = df_clean[col].notna() & (s.apply(norm_case) != s)
                        rows_affected += int(changed.sum())
                        df_clean[col] = df_clean[col].apply(norm_case)
                step.rows_affected = rows_affected
                step.status = "applied"

            elif step.step_id.startswith("coerce_numeric_"):
                rows_affected = 0
                for col in step.affected_fields:
                    if col in df_clean.columns:
                        orig_notna = df_clean[col].notna()
                        coerced = pd.to_numeric(df_clean[col], errors="coerce")
                        became_nan = int((orig_notna & coerced.isna()).sum())
                        df_clean[col] = coerced
                        rows_affected += became_nan
                step.rows_affected = rows_affected
                step.status = "applied"

            elif step.step_id.startswith("parse_dates_"):
                rows_affected = 0
                for col in step.affected_fields:
                    if col in df_clean.columns:
                        parsed = pd.to_datetime(df_clean[col], errors="coerce", format="mixed")
                        mask_success = parsed.notna()
                        formatted_dates = parsed[mask_success].dt.strftime("%Y-%m-%d")

                        s = df_clean[col].copy()
                        s.loc[mask_success] = formatted_dates
                        df_clean[col] = s
                        rows_affected += int(mask_success.sum())
                step.rows_affected = rows_affected
                step.status = "applied"

            elif step.step_id == "dedupe_exact_rows":
                dups_count = int(df_clean.duplicated().sum())
                df_clean = df_clean.drop_duplicates(keep="first").reset_index(drop=True)
                step.rows_affected = dups_count
                step.status = "applied"

            elif step.step_id.startswith("dedupe_duplicate_ids_"):
                # Conservative: do NOT drop rows, only count and skip
                total_dups = 0
                for col in step.affected_fields:
                    if col in df_clean.columns:
                        total_dups += int(df_clean[col].dropna().duplicated().sum())
                step.rows_affected = total_dups
                step.status = "skipped"
                step.reason = f"{step.reason} (Manual review required - skipped to avoid silent data loss)"

            else:
                step.status = "skipped"
                step.rows_affected = 0

        except Exception as e:
            step.status = "failed"
            step.rows_affected = 0
            step.reason = f"{step.reason} (Failed with error: {e})"

    return df_clean, plan


def validate_cleaning(df_before: pd.DataFrame, df_after: pd.DataFrame, plan: List[CleaningStep]) -> Dict[str, Any]:
    """
    Compares dataframes before and after cleaning and summarizes unresolved issues.
    """
    steps_applied = sum(1 for s in plan if s.status == "applied")
    steps_skipped = sum(1 for s in plan if s.status == "skipped")
    steps_failed = sum(1 for s in plan if s.status == "failed")

    unresolved_issues: List[str] = []

    # 1. Check for remaining nulls
    for col in df_after.columns:
        null_count = int(df_after[col].isna().sum())
        if null_count > 0:
            unresolved_issues.append(f"{col}: {null_count} rows still null")

    # 2. Check for remaining duplicate IDs in key columns
    for col in df_after.columns:
        col_str = str(col).lower()
        if col_str.endswith("_id") or col_str == "sku":
            dup_count = int(df_after[col].dropna().duplicated().sum())
            if dup_count > 0:
                cardinality_ratio = df_after[col].nunique() / max(len(df_after), 1)
                if cardinality_ratio >= 0.8:
                    unresolved_issues.append(f"{col}: {dup_count} duplicate ids remain (not dropped)")

    # 3. Check for skipped or failed steps
    for s in plan:
        if s.status == "skipped" and s.step_id.startswith("dedupe_duplicate_ids_") and s.rows_affected > 0:
            for f in s.affected_fields:
                msg = f"{f}: {s.rows_affected} duplicate ids remain (not dropped)"
                if msg not in unresolved_issues:
                    unresolved_issues.append(msg)
        elif s.status == "failed":
            unresolved_issues.append(f"{s.step_id}: execution failed ({s.reason})")

    return {
        "row_count_before": int(len(df_before)),
        "row_count_after": int(len(df_after)),
        "column_count_before": int(len(df_before.columns)),
        "column_count_after": int(len(df_after.columns)),
        "steps_applied": steps_applied,
        "steps_skipped": steps_skipped,
        "steps_failed": steps_failed,
        "unresolved_issues": unresolved_issues,
    }
