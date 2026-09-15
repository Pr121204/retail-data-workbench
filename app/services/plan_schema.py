from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class Filter(BaseModel):
    field: str
    op: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains"]
    value: Any


class Metric(BaseModel):
    agg: Literal["sum", "avg", "count", "min", "max", "count_distinct"]
    field: str
    as_: str = Field(alias="as")
    model_config = {"populate_by_name": True}


class Sort(BaseModel):
    field: str
    dir: Literal["asc", "desc"] = "desc"


class QueryPlan(BaseModel):
    intent: Literal["filter", "aggregate", "compare", "top_n", "describe"]
    dataset: str                      # must be one of the known clean dataset names
    join: Optional[dict] = None       # optional {"with": "<dataset>"} — only used if needed
    filters: List[Filter] = []
    group_by: List[str] = []
    metrics: List[Metric] = []
    sort: List[Sort] = []
    limit: int = 20
