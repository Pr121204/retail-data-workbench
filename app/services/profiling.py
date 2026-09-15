from typing import Any, Dict
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a plain JSON-serializable profiling summary of a DataFrame.
    Pure function without any I/O or database interactions.
    """
    row_count = int(len(df))
    column_count = int(len(df.columns))

    columns_profile: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(float((null_count / row_count) * 100), 2) if row_count > 0 else 0.0

        non_null_series = series.dropna()
        distinct_count = int(non_null_series.nunique())
        cardinality_ratio = round(float(distinct_count / row_count), 4) if row_count > 0 else 0.0

        sample_values = [str(v) for v in non_null_series.unique()[:5]]

        col_dict: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "distinct_count": distinct_count,
            "cardinality_ratio": cardinality_ratio,
            "sample_values": sample_values,
        }

        # Parse failure detection heuristic
        col_lower = str(col).lower()
        parse_info = None

        if "date" in col_lower:
            try:
                if len(non_null_series) > 0:
                    parsed = pd.to_datetime(non_null_series, errors="coerce", format="mixed")
                    failures = int(parsed.isna().sum())
                else:
                    failures = 0
                parse_info = {
                    "attempted_as": "date",
                    "failure_count": failures,
                }
            except Exception:
                parse_info = {
                    "attempted_as": "date",
                    "failure_count": 0,
                }
        elif any(kw in col_lower for kw in ["price", "revenue", "quantity", "stock"]):
            try:
                if len(non_null_series) > 0:
                    parsed = pd.to_numeric(non_null_series, errors="coerce")
                    failures = int(parsed.isna().sum())
                else:
                    failures = 0
                parse_info = {
                    "attempted_as": "numeric",
                    "failure_count": failures,
                }
            except Exception:
                parse_info = {
                    "attempted_as": "numeric",
                    "failure_count": 0,
                }
        elif "id" in col_lower:
            try:
                if len(non_null_series) > 0:
                    if pd.api.types.is_numeric_dtype(series.dtype):
                        parsed = pd.to_numeric(non_null_series, errors="coerce")
                        failures = int(parsed.isna().sum())
                        parse_info = {
                            "attempted_as": "numeric",
                            "failure_count": failures,
                        }
                    else:
                        # Check for numeric expectation in object/string ID
                        parsed = pd.to_numeric(non_null_series, errors="coerce")
                        valid_num_count = int(parsed.notna().sum())
                        invalid_num_count = int(parsed.isna().sum())
                        if valid_num_count > 0 and (valid_num_count >= invalid_num_count or valid_num_count >= len(non_null_series) * 0.3):
                            parse_info = {
                                "attempted_as": "numeric",
                                "failure_count": invalid_num_count,
                            }
            except Exception:
                pass

        if parse_info is not None:
            col_dict["parse_failures"] = parse_info

        columns_profile[str(col)] = col_dict

    duplicate_row_count = int(df.duplicated().sum())

    duplicate_key_columns: Dict[str, int] = {}
    for col in df.columns:
        if str(col).lower().endswith("_id"):
            duplicate_key_columns[str(col)] = int(df[col].dropna().duplicated().sum())

    return {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns_profile,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_key_columns": duplicate_key_columns,
    }
