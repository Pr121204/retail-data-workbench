from typing import Dict, List, Set
from app.services.joins import JOIN_REGISTRY
from app.services.plan_schema import QueryPlan

ALLOWED_DATASETS: Set[str] = {"orders", "products", "customers", "stores", "inventory"}
ALLOWED_JOIN_PAIRS = JOIN_REGISTRY
MAX_LIMIT = 200

# Per-dataset allowed field names
ALLOWED_FIELDS: Dict[str, List[str]] = {
    "orders": ["order_id", "customer_id", "product_id", "store_id", "quantity", "revenue", "order_date", "region"],
    "products": ["product_id", "product_name", "category", "brand", "sku", "price"],
    "customers": ["customer_id", "name", "email", "region", "signup_date"],
    "stores": ["store_id", "store_name", "region", "city"],
    "inventory": ["product_id", "store_id", "stock_quantity", "last_restock_date"],
}

NUMERIC_FIELDS: Set[str] = {"price", "revenue", "quantity", "stock_quantity"}


class PlanValidationError(Exception):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.message = message
        self.code = code


def validate_plan(plan: QueryPlan) -> QueryPlan:
    """
    Validates a QueryPlan against strict schema and security allow-lists.
    Raises PlanValidationError with specific error codes on failure.
    """
    # 1. Dataset allow-list check
    if plan.dataset not in ALLOWED_DATASETS:
        raise PlanValidationError(
            f"Dataset '{plan.dataset}' is not in allowed datasets: {sorted(ALLOWED_DATASETS)}",
            code="unknown_dataset",
        )

    # 2. Join allow-list check
    joined_dataset = None
    if plan.join is not None:
        if not isinstance(plan.join, dict) or "with" not in plan.join:
            raise PlanValidationError(
                "Join specification must be an object with key 'with'",
                code="unsafe_join",
            )
        joined_dataset = plan.join["with"]
        if joined_dataset not in ALLOWED_DATASETS:
            raise PlanValidationError(
                f"Joined dataset '{joined_dataset}' is not in allowed datasets: {sorted(ALLOWED_DATASETS)}",
                code="unknown_dataset",
            )
        pair_forward = (plan.dataset, joined_dataset)
        pair_reverse = (joined_dataset, plan.dataset)
        if pair_forward not in ALLOWED_JOIN_PAIRS and pair_reverse not in ALLOWED_JOIN_PAIRS:
            raise PlanValidationError(
                f"Join between '{plan.dataset}' and '{joined_dataset}' is not allowed by join registry",
                code="unsafe_join",
            )

    # 3. Field allow-list checks
    allowed_fields = set(ALLOWED_FIELDS.get(plan.dataset, []))
    if joined_dataset:
        allowed_fields.update(ALLOWED_FIELDS.get(joined_dataset, []))

    for f in plan.filters:
        if f.field not in allowed_fields:
            raise PlanValidationError(
                f"Filter field '{f.field}' is not in allowed fields for dataset(s)",
                code="unknown_field",
            )

    for m in plan.metrics:
        if m.agg == "count" and m.field == "*":
            continue
        if m.field not in allowed_fields:
            raise PlanValidationError(
                f"Metric field '{m.field}' is not in allowed fields for dataset(s)",
                code="unknown_field",
            )

    for g in plan.group_by:
        if g not in allowed_fields:
            raise PlanValidationError(
                f"Group by field '{g}' is not in allowed fields for dataset(s)",
                code="unknown_field",
            )

    metric_aliases = {m.as_ for m in plan.metrics}
    sort_allowed = allowed_fields | metric_aliases
    for s in plan.sort:
        if s.field not in sort_allowed:
            raise PlanValidationError(
                f"Sort field '{s.field}' is not in allowed fields or metric aliases",
                code="unknown_field",
            )

    # 4. Limit check
    if plan.limit > MAX_LIMIT:
        raise PlanValidationError(
            f"Limit {plan.limit} exceeds maximum allowed limit of {MAX_LIMIT}",
            code="limit_exceeded",
        )
    if plan.limit < 1:
        raise PlanValidationError(
            f"Limit {plan.limit} must be greater than or equal to 1",
            code="limit_exceeded",
        )

    # 5. Invalid operation for field check
    for f in plan.filters:
        if f.op == "contains" and f.field in NUMERIC_FIELDS:
            raise PlanValidationError(
                f"Operation 'contains' is not allowed on numeric field '{f.field}'",
                code="invalid_op_for_field",
            )

    return plan
