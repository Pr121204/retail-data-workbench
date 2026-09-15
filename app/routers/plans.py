from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.services.plan_schema import QueryPlan
from app.services.plan_validator import PlanValidationError, validate_plan

router = APIRouter(prefix="/plans", tags=["plans"])


@router.post("/validate", status_code=status.HTTP_200_OK)
def validate_query_plan(plan: QueryPlan):
    try:
        validated = validate_plan(plan)
        return {
            "valid": True,
            "plan": validated.model_dump(by_alias=True),
        }
    except PlanValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "valid": False,
                "error_code": e.code,
                "error_message": e.message,
            },
        )
