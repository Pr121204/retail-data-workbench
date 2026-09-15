from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.services.joins import JOIN_REGISTRY, safe_join
from app.services.plan_schema import QueryPlan


def to_json_safe(val: Any) -> Any:
    """Converts numpy/pandas types to native Python JSON-serializable types."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return round(float(val), 2)
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def execute_plan(
    plan: QueryPlan,
    clean_datasets: Dict[str, pd.DataFrame],
    evidence_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes an already-validated QueryPlan against loaded clean DataFrames.
    Top-level exception handler ensures it never raises an unhandled error.
    """
    try:
        # a) Start with base_df = clean_datasets[plan.dataset].copy()
        if plan.dataset not in clean_datasets:
            raise ValueError(f"Dataset '{plan.dataset}' not found in loaded clean datasets")

        base_df = clean_datasets[plan.dataset].copy()
        row_count_before_filter = int(len(base_df))

        # b) If plan.join is set: use get_join_config + safe_join
        join_report: Optional[Dict[str, Any]] = None
        if plan.join is not None and isinstance(plan.join, dict) and "with" in plan.join:
            with_dataset = plan.join["with"]
            if with_dataset not in clean_datasets:
                raise ValueError(f"Joined dataset '{with_dataset}' not found in clean datasets")

            right_df = clean_datasets[with_dataset].copy()

            if (plan.dataset, with_dataset) in JOIN_REGISTRY:
                cfg = JOIN_REGISTRY[(plan.dataset, with_dataset)]
                base_df, join_report = safe_join(
                    left_df=base_df,
                    right_df=right_df,
                    left_key=cfg["left_key"],
                    right_key=cfg["right_key"],
                    how="left",
                )
            elif (with_dataset, plan.dataset) in JOIN_REGISTRY:
                cfg = JOIN_REGISTRY[(with_dataset, plan.dataset)]
                base_df, join_report = safe_join(
                    left_df=base_df,
                    right_df=right_df,
                    left_key=cfg["right_key"],
                    right_key=cfg["left_key"],
                    how="left",
                )
            else:
                raise ValueError(f"Unsafe join between '{plan.dataset}' and '{with_dataset}'")

            # Update row count before filters in case join changed row count
            row_count_before_filter = int(len(base_df))

        # c) Apply filters using boolean masks
        filters_applied: List[Dict[str, Any]] = []
        filters_warnings: List[Dict[str, Any]] = []

        for f in plan.filters:
            try:
                if f.field not in base_df.columns:
                    raise KeyError(f"Field '{f.field}' not found in DataFrame")

                col_series = base_df[f.field]

                if f.op == "eq":
                    mask = col_series == f.value
                elif f.op == "neq":
                    mask = col_series != f.value
                elif f.op == "gt":
                    mask = col_series > f.value
                elif f.op == "gte":
                    mask = col_series >= f.value
                elif f.op == "lt":
                    mask = col_series < f.value
                elif f.op == "lte":
                    mask = col_series <= f.value
                elif f.op == "in":
                    values_set = f.value if isinstance(f.value, (list, tuple, set)) else [f.value]
                    mask = col_series.isin(values_set)
                elif f.op == "contains":
                    mask = col_series.astype(str).str.contains(str(f.value), case=False, na=False)
                else:
                    raise ValueError(f"Unsupported operation '{f.op}'")

                base_df = base_df[mask.fillna(False)].copy()
                filters_applied.append(f.model_dump(by_alias=True))
            except Exception as ex:
                filters_warnings.append(
                    {
                        "filter": f.model_dump(by_alias=True),
                        "warning": str(ex),
                    }
                )

        if filters_warnings:
            raise ValueError(
                "One or more filters could not be applied: "
                + "; ".join(item["warning"] for item in filters_warnings)
            )

        row_count_after_filter = int(len(base_df))

        # d) Aggregation / Metrics / Group By
        agg_map = {
            "sum": "sum",
            "avg": "mean",
            "count": "count",
            "min": "min",
            "max": "max",
            "count_distinct": "nunique",
        }

        if plan.group_by and plan.metrics:
            grouped = base_df.groupby(plan.group_by, dropna=False, sort=False)
            result_df = None
            for metric in plan.metrics:
                if metric.agg == "count" and metric.field == "*":
                    metric_df = grouped.size().reset_index(name=metric.as_)
                else:
                    metric_df = (
                        grouped[metric.field]
                        .agg(agg_map.get(metric.agg, metric.agg))
                        .reset_index(name=metric.as_)
                    )
                if result_df is None:
                    result_df = metric_df
                else:
                    result_df = result_df.merge(metric_df, on=plan.group_by, how="outer")

        elif plan.metrics and not plan.group_by:
            row: Dict[str, Any] = {}
            for m in plan.metrics:
                func = agg_map.get(m.agg, m.agg)
                if m.field == "*":
                    val = len(base_df)
                else:
                    series = base_df[m.field].dropna()
                    if func == "sum":
                        val = series.sum()
                    elif func == "mean":
                        val = series.mean()
                    elif func == "count":
                        val = series.count()
                    elif func == "min":
                        val = series.min()
                    elif func == "max":
                        val = series.max()
                    elif func == "nunique":
                        val = series.nunique()
                    else:
                        val = None
                row[m.as_] = val
            result_df = pd.DataFrame([row])

        else:
            result_df = base_df.copy()

        # e) Sorting
        if plan.sort:
            sort_cols = []
            asc_list = []
            for s in plan.sort:
                if s.field in result_df.columns:
                    sort_cols.append(s.field)
                    asc_list.append(s.dir == "asc")
            if sort_cols:
                result_df = result_df.sort_values(by=sort_cols, ascending=asc_list)

        # f) Limit & count
        result_row_count = int(len(result_df))
        limited_df = result_df.head(plan.limit)

        # g) JSON-safe conversion
        result_rows = [
            {col: to_json_safe(val) for col, val in r.items()}
            for r in limited_df.to_dict(orient="records")
        ]

        result_preview = [
            {col: to_json_safe(val) for col, val in r.items()}
            for r in limited_df.head(20).to_dict(orient="records")
        ]
        context = dict(evidence_context or {})
        columns_used = sorted(
            set(plan.group_by)
            | {metric.field for metric in plan.metrics if metric.field != "*"}
            | {filter_.field for filter_ in plan.filters}
            | {sort_.field for sort_ in plan.sort}
        )
        if join_report:
            columns_used = sorted(
                set(columns_used)
                | {join_report["left_key"], join_report["right_key"]}
            )
        operations = []
        if plan.filters:
            operations.append("filter")
        if plan.group_by:
            operations.append("group_by")
        operations.extend(metric.agg for metric in plan.metrics)
        if plan.sort:
            operations.append("sort")
        if plan.limit:
            operations.append("limit")

        return {
            "result_rows": result_rows,
            "result_row_count": result_row_count,
            "evidence": {
                "run_id": context.get("run_id"),
                "dataset_stage": context.get("dataset_stage", "clean"),
                "dataset_version": context.get("dataset_version", "clean"),
                "dataset": plan.dataset,
                "columns_used": columns_used,
                "join": plan.join,
                "join_report": join_report,
                "filters_applied": filters_applied,
                "filters_warnings": filters_warnings,
                "group_by": plan.group_by,
                "metrics": [m.model_dump(by_alias=True) for m in plan.metrics],
                "operations": operations,
                "row_count_before_filter": row_count_before_filter,
                "row_count_after_filter": row_count_after_filter,
                "limit_applied": plan.limit,
                "result_preview": result_preview,
            },
        }

    except Exception as e:
        return {
            "result_rows": [],
            "result_row_count": 0,
            "evidence": {
                "run_id": (evidence_context or {}).get("run_id"),
                "dataset_stage": (evidence_context or {}).get("dataset_stage", "clean"),
                "dataset_version": (evidence_context or {}).get("dataset_version", "clean"),
                "dataset": getattr(plan, "dataset", None),
                "columns_used": [],
                "join": getattr(plan, "join", None),
                "join_report": None,
                "filters_applied": [],
                "filters_warnings": [],
                "group_by": getattr(plan, "group_by", []),
                "metrics": [m.model_dump(by_alias=True) for m in getattr(plan, "metrics", [])],
                "operations": [],
                "row_count_before_filter": 0,
                "row_count_after_filter": 0,
                "limit_applied": getattr(plan, "limit", 20),
                "result_preview": [],
            },
            "execution_error": str(e),
        }
