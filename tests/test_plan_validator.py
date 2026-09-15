import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.plan_schema import Filter, Metric, QueryPlan
from app.services.plan_validator import PlanValidationError, validate_plan


def test_reject_unknown_dataset():
    plan = QueryPlan(intent="filter", dataset="secret_admin_table")
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan)
    assert exc.value.code == "unknown_dataset"


def test_reject_unknown_field():
    plan = QueryPlan(
        intent="filter",
        dataset="orders",
        filters=[Filter(field="ssn", op="eq", value="123-45-6789")],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan)
    assert exc.value.code == "unknown_field"


def test_reject_unsafe_join():
    plan = QueryPlan(
        intent="filter",
        dataset="customers",
        join={"with": "inventory"},
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan)
    assert exc.value.code == "unsafe_join"


def test_reject_limit_exceeded():
    plan = QueryPlan(
        intent="filter",
        dataset="orders",
        limit=99999,
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan)
    assert exc.value.code == "limit_exceeded"


def test_reject_invalid_op_for_field():
    plan = QueryPlan(
        intent="filter",
        dataset="products",
        filters=[Filter(field="price", op="contains", value="10")],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan)
    assert exc.value.code == "invalid_op_for_field"


def test_valid_plan_passes():
    plan = QueryPlan(
        intent="aggregate",
        dataset="orders",
        join={"with": "products"},
        filters=[Filter(field="region", op="eq", value="West")],
        group_by=["category"],
        metrics=[Metric(agg="sum", field="revenue", as_="total_sales")],
        limit=50,
    )
    validated = validate_plan(plan)
    assert validated == plan


def test_validate_endpoint_rejects_malicious_plan():
    with TestClient(app) as client:
        payload = {"intent": "filter", "dataset": "secret_admin_table"}
        res = client.post("/plans/validate", json=payload)
        assert res.status_code == 422
        body = res.json()
        assert body["valid"] is False
        assert body["error_code"] == "unknown_dataset"


def test_validate_endpoint_accepts_valid_plan():
    with TestClient(app) as client:
        payload = {
            "intent": "aggregate",
            "dataset": "orders",
            "join": {"with": "products"},
            "filters": [{"field": "region", "op": "eq", "value": "West"}],
            "group_by": ["category"],
            "metrics": [{"agg": "sum", "field": "revenue", "as": "total_sales"}],
            "limit": 50,
        }
        res = client.post("/plans/validate", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["valid"] is True
        assert body["plan"]["dataset"] == "orders"
        assert body["plan"]["join"]["with"] == "products"
