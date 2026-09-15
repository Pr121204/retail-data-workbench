"""
Chat intent gate: a lightweight pre-check that runs BEFORE generate_plan().

NOTE: This is deliberately a crude keyword/structure heuristic, NOT a full NLU
system. Its only job is to catch the obvious cases from the assessment's
challenge list ("ambiguous question", "unanswerable question") so the chat
layer can refuse or ask for clarification instead of silently producing a
garbage plan. Expect false positives and false negatives on unusual phrasing.
"""
from __future__ import annotations

import re
from typing import Optional

# --- unanswerable signal terms, grouped by what's missing -------------------

COMPETITOR_TERMS = ["competitor"]
WEATHER_TERMS = ["weather", "temperature", "rain", "rainfall", "snow", "humidity"]
# We have no forecasting capability — the datasets are historical only.
FORECAST_TERMS = ["forecast", "predict", "what will happen", "next year"]
MARKET_TERMS = [
    "market share",
    "market trend",
    "stock price",
    "stock market",
    "inflation",
    "interest rate",
    "exchange rate",
    "gdp",
    "economic outlook",
]

UNANSWERABLE_KEYWORDS = COMPETITOR_TERMS + WEATHER_TERMS + FORECAST_TERMS + MARKET_TERMS

# Pronouns that usually refer back to a previous turn's result. Only treated
# as ambiguous when there is no carried-over context to resolve them.
# ("this" is intentionally excluded: "this month" / "this quarter" are common,
# answerable time phrases rather than pronoun references.)
PRONOUNS = {"it", "that", "those", "these", "them"}

# Words that signal a comparison request.
COMPARISON_WORDS = {"compare", "comparison", "vs", "versus"}

# Dataset names + common field nouns: a comparison is only considered
# "specified" when the question names something from the data's scope.
DATA_SCOPE_WORDS = {
    "orders", "order", "products", "product", "customers", "customer",
    "stores", "store", "inventory", "revenue", "quantity", "price",
    "category", "region", "stock", "sales", "return", "returns",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "by", "at", "from", "between", "among", "them", "it", "that", "those",
    "these", "this", "us", "our", "we", "me", "my", "is", "are", "was",
    "were", "be", "do", "does", "did", "how", "what", "which", "who",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def unanswerable_reason(question: str) -> str | None:
    """Returns the matched out-of-scope keyword, or None if none matched."""
    q = (question or "").lower()
    for keyword in UNANSWERABLE_KEYWORDS:
        if keyword in q:
            return keyword
    return None


def ambiguous_reason(question: str, active_filters: dict | None = None) -> str | None:
    """
    Returns why the question is too vague to act on, or None if it isn't.

    One of: "too_short", "unresolved_pronoun:<word>", "vague_comparison".
    A pronoun is considered resolvable when non-empty active_filters context
    from earlier turns is available.
    """
    q = (question or "").lower().strip()
    tokens = _TOKEN_RE.findall(q)
    words = q.split()

    if len(words) < 3:
        return "too_short"

    pronouns_found = [t for t in tokens if t in PRONOUNS]
    if pronouns_found and not (active_filters or {}):
        return f"unresolved_pronoun:{pronouns_found[0]}"

    if any(t in COMPARISON_WORDS for t in tokens):
        remaining = [t for t in tokens if t not in COMPARISON_WORDS]
        content = [t for t in remaining if t not in _STOPWORDS]
        # Vague when there's nothing to compare ("just 'compare'"), or the
        # comparison names nothing from the data's scope ("compare with last
        # year" — no dataset or field mentioned).
        if len(remaining) < 2 or not any(t in DATA_SCOPE_WORDS for t in content):
            return "vague_comparison"

    return None


def match_analytics_capability(question: str) -> Optional[str]:
    """
    Keyword-pattern matcher for the four retail analytics capabilities in
    analytics.py. These questions need business logic the generic
    QueryPlan/execute_plan pipeline cannot express (e.g. return rate = orders
    with revenue < 0, per category), so they must be routed to the analytics
    functions directly instead of the planner.

    Returns the capability name, or None to fall through to the generic
    plan pipeline. Simple word patterns, not NLU — see module docstring.
    """
    q = (question or "").lower()
    has = _TOKEN_RE.findall(q)
    words = set(has)

    # return rate / refunds: return(s)/refund AND (rate/category)
    if ("return" in words or "returns" in words or "refund" in words or "refunds" in words) and (
        "rate" in words or "rates" in words or "category" in words or "categories" in words
    ):
        return "return_rate_by_category"

    # revenue by category: revenue/sales AND category
    if ("revenue" in words or "sales" in words) and (
        "category" in words or "categories" in words
    ):
        return "revenue_by_category"

    # store performance: store(s) AND (performance/sales/revenue/compare)
    if ("store" in words or "stores" in words) and (
        "performance" in words
        or "sales" in words
        or "revenue" in words
        or "compare" in words
        or "comparison" in words
    ):
        return "store_performance"

    # stockout & ageing: stock/inventory AND (out/low/ageing/aging).
    # "stockout"/"stock-outs" are matched as substrings since they are single
    # tokens that contain both signal words.
    has_stock_topic = "stock" in words or "inventory" in words or "stockout" in q or "stock out" in q
    has_stock_signal = (
        "out" in words
        or "low" in words
        or "ageing" in words
        or "aging" in words
        or "stockout" in q
        or "stock out" in q
    )
    if has_stock_topic and has_stock_signal:
        return "stockout_and_ageing"

    return None


def classify_question(
    question: str,
    dataset_names: list[str],
    active_filters: dict | None = None,
) -> str:
    """
    Classifies a user question before planning.

    Returns one of:
      - "unanswerable": clearly outside the data's scope (competitors, weather,
        external market data, forecasting requests) -> caller should refuse.
      - "ambiguous": too vague to act on (under 3 words, an unresolved pronoun
        with no carried-over context, or a bare comparison request)
        -> caller should ask a targeted clarifying question.
      - "answerable": everything else -> proceed to generate_plan().

    Heuristic gate, not perfect — see module docstring. `dataset_names` is
    part of the chat-layer contract (reserved for future scope checks such as
    verifying a named dataset exists in this run); the current heuristics
    don't need it. `active_filters` is the carried-over context dict from
    earlier turns ({} on turn 1).
    """
    if unanswerable_reason(question) is not None:
        return "unanswerable"
    if ambiguous_reason(question, active_filters) is not None:
        return "ambiguous"
    return "answerable"
