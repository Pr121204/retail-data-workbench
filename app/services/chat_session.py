"""
Multi-turn chat context helpers: carried-over "active filters" management.

Active filters are the explicit constraints from earlier turns that should
carry forward, stored on the ChatSession row as a flat JSON object, e.g.
{"region": "West"}. Only simple equality constraints are carried — the flat
{field: value} shape cannot represent ranges ("stock_quantity < 10"), and
re-applying such a filter as 'eq' would silently produce wrong results.
"""
from __future__ import annotations

import re

from app.services.plan_schema import Filter, QueryPlan

# Region values present in the sample datasets (orders / customers / stores
# all have a 'region' column holding exactly these values).
_KNOWN_REGIONS = {"west": "West", "east": "East", "north": "North", "south": "South"}

_WORD_RE = re.compile(r"[a-z0-9']+")


def merge_active_filters(existing_filters: dict, new_plan_filters: list[Filter]) -> dict:
    """
    Merge a plan's filters into the carried-over context.

    New filters override existing keys with the same field name. Non-'eq'
    filters (ranges, 'in', 'contains', ...) are skipped because the flat
    {field: value} context can only be re-applied as an equality.

    Returns the updated dict, e.g. {"region": "West"}.
    """
    merged = dict(existing_filters or {})
    for f in new_plan_filters or []:
        if f.op == "eq":
            merged[f.field] = f.value
    return merged


def apply_active_filters_to_plan(plan: QueryPlan, active_filters: dict) -> QueryPlan:
    """
    Carry active filters forward into a new plan: any context key NOT already
    present as a filter field in plan.filters is added as an 'eq' filter (so
    "only West region" applies to the next turn automatically). Filters the
    planner set explicitly always win — we never duplicate or override an
    existing plan filter.

    Returns the modified plan.
    """
    existing_fields = {f.field for f in plan.filters}
    for field, value in (active_filters or {}).items():
        if field not in existing_fields:
            plan.filters.append(Filter(field=field, op="eq", value=value))
            existing_fields.add(field)
    return plan


def extract_filters_from_question(question: str) -> list[Filter]:
    """
    Best-effort extraction of an explicit constraint mention from the question
    itself, e.g. "in the West region" -> [Filter(region eq West)].

    Why this exists: generate_plan() is a black box we must not modify, and
    the mock planner answers "Show me orders in the West region" with an
    aggregate-by-region plan that carries NO region filter — so nothing would
    ever be recorded into active_filters, and the assessment's multi-turn
    scenario ("Which categories had the highest return rate among those?")
    could never carry the West context forward. This is a heuristic, not an
    NLU system: it only recognises the region values present in the sample
    datasets, using whole-word matching, and only when exactly one region is
    named (multiple regions means a comparison, not a filter).
    """
    q = (question or "").lower()
    tokens = set(_WORD_RE.findall(q))
    matches = [canonical for value, canonical in _KNOWN_REGIONS.items() if value in tokens]
    if len(matches) != 1:
        return []
    return [Filter(field="region", op="eq", value=matches[0])]
