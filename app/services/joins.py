from typing import Any, Dict, List, Tuple
import pandas as pd

JOIN_REGISTRY: Dict[Tuple[str, str], Dict[str, str]] = {
    ("orders", "products"): {"left_key": "product_id", "right_key": "product_id"},
    ("orders", "customers"): {"left_key": "customer_id", "right_key": "customer_id"},
    ("orders", "stores"): {"left_key": "store_id", "right_key": "store_id"},
    ("inventory", "products"): {"left_key": "product_id", "right_key": "product_id"},
    ("inventory", "stores"): {"left_key": "store_id", "right_key": "store_id"},
}


def get_join_config(left_name: str, right_name: str) -> Dict[str, str]:
    """
    Looks up JOIN_REGISTRY safety allow-list. Raises ValueError if the pair is not permitted.
    """
    pair = (left_name, right_name)
    if pair not in JOIN_REGISTRY:
        allowed = ", ".join(f"({l}->{r})" for l, r in JOIN_REGISTRY.keys())
        raise ValueError(
            f"Join between '{left_name}' and '{right_name}' is not permitted by JOIN_REGISTRY. Allowed joins: {allowed}"
        )
    return JOIN_REGISTRY[pair]


def safe_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "left",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs a validated cross-dataset join with fan-out check and unmatched row tracking.
    """
    if how not in {"left", "inner", "right", "outer"}:
        raise ValueError(f"Unsupported join mode '{how}'")
    if left_key not in left_df.columns:
        raise ValueError(f"Left join key '{left_key}' is missing")
    if right_key not in right_df.columns:
        raise ValueError(f"Right join key '{right_key}' is missing")

    # 1. Check right_key uniqueness in right_df BEFORE joining
    right_key_series = right_df[right_key].dropna()
    fan_out_duplicates = int(right_key_series.duplicated().sum())
    fan_out_detected = fan_out_duplicates > 0

    # 2. Count rows in left_df whose left_key value does not appear in right_df's right_key
    right_unique_keys = set(right_key_series.unique())
    unmatched_mask = ~left_df[left_key].isin(right_unique_keys)
    unmatched_left_count = int(unmatched_mask.sum())
    unmatched_left_examples = [
        str(k) for k in left_df.loc[unmatched_mask, left_key].dropna().unique()[:5]
    ]

    # 3. Perform the merge
    if left_key == right_key:
        joined_df = pd.merge(left_df, right_df, on=left_key, how=how)
    else:
        joined_df = pd.merge(left_df, right_df, left_on=left_key, right_on=right_key, how=how)

    report = {
        "left_rows": int(len(left_df)),
        "right_rows": int(len(right_df)),
        "joined_rows": int(len(joined_df)),
        "left_key": left_key,
        "right_key": right_key,
        "fan_out_detected": fan_out_detected,
        "fan_out_key_duplicate_count": fan_out_duplicates,
        "unmatched_left_count": unmatched_left_count,
        "unmatched_left_examples": unmatched_left_examples,
    }

    return joined_df, report
