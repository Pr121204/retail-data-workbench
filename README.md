# retail-data-workbench

FastAPI service that ingests retail CSVs, profiles/cleans them, and answers
natural-language questions through a **plan → validate → execute** pipeline
with a full evidence trail. Includes a minimal React UI, a headless batch
evaluator, and a prompt-injection defense model that is proven by tests.

## Architecture

```
                        ┌──────────────────────────────────────────────┐
 CSV files ── upload ──>│ Run (DB) ── profile ──> clean ──> clean CSVs │
                        └──────────────┬───────────────────────────────┘
                                       │ clean datasets (in-memory DataFrames)
          ┌────────────────────────────┼──────────────────────────────┐
          ▼                            ▼                              ▼
   POST .../chat/.../turns      POST /runs/{id}/query        python -m app.evaluate
   (multi-turn, active_filters) (single-shot)                (headless batch, same
          │                            │                       service functions)
          ▼                            ▼
   ┌─────────────────────────────────────────────┐
   │            question routing (code)          │
   │  1. match_analytics_capability?  ───────────┼──> analytics.py functions
   │     (return rate, revenue by category,      │    (business logic pandas)
   │      store performance, stockouts)          │
   │  2. classify_question:                      │
   │     unanswerable -> refused turn            │
   │     ambiguous    -> clarification turn      │
   │  3. otherwise: generate_plan ─> validate_plan ─> execute_plan
   │     (LLM suggestion)          (allow-lists)    (pandas, evidence)
   └─────────────────────────────────────────────┘
                       │
                       ▼
        ChatTurn persisted: status, plan_json, answer_text,
        evidence_json  (answer_text = template-formatted from
        the computed result — numbers are never invented)
```

Every answer path lands in a persisted `ChatTurn` with machine-readable
status (`ok`, `clarification_needed`, `refused`, `plan_rejected`,
`llm_fallback`) and raw evidence, so any answer can be audited after the
fact. Filter context carries across turns as `active_filters`
(e.g. `{"region": "West"}`), applied both to generic plans and as a
pre-filter to analytics capabilities.

## The AI-vs-code boundary

The guiding rule: **the LLM is at most a plan *suggester*; everything that
decides, enforces, or computes is deterministic code.**

| Concern | Handled by | Why |
|---|---|---|
| Question → plan *suggestion* | LLM (Gemini, optional) or mock planner | convenience only; output is untrusted |
| Scope/ambiguity gating | code (`chat_intent.py` heuristics) | predictable, testable, conservative |
| Retail business logic (return rate, stockouts…) | code (`analytics.py`) | the LLM cannot invent metrics; no returns column exists, "return" = `revenue < 0` is a *code* decision |
| Plan safety (datasets, fields, joins, limits) | code (`plan_validator.py` allow-lists) | the validator, not the LLM, is the security boundary |
| Execution & numbers | code (`plan_executor.py`, pandas) | arithmetic must be deterministic and reproducible |
| Answer text | code (string templates over computed results) | grounding: every number in prose comes from the evidence dict |

Consequences worth stating plainly:

- The default mode is `USE_MOCK_LLM=True` — the whole system runs offline
  and deterministically with **zero LLM calls**. The Gemini path is optional
  and, when enabled, still never sees row data (schema + question only).
- A hallucinated or manipulated plan cannot widen results: hostile plans are
  rejected by the same allow-lists as hand-typed ones (see
  `tests/test_prompt_injection.py`).
- Dataset cell values are inert: they can surface in result rows byte-for-byte
  but have no path into plans, filters, or instructions (proven by test).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
cp .env.example .env                            # defaults: USE_MOCK_LLM=True
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Delete `retail.db` whenever the schema changed before starting the server —
`Base.metadata.create_all` only adds missing tables, it does not migrate.

## Run Frontend

A minimal React UI (Vite) lives in `frontend/`. The backend must be running
separately on port 8000:

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Then, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The UI calls http://localhost:8000 directly
(allowed via CORS middleware in `app/main.py`); no proxy is configured.

## Batch evaluator

```bash
python -m app.evaluate --input data/samples/evaluation_cases.json \
    --output results.json --artifacts-dir ./evaluation_output
```

Runs cleaning/chat/join cases headlessly through the same service functions
as the API. One failed case never aborts the batch (`status: "error"` +
traceback artifact); results include per-case profiles, cleaning plans, chat
turns with evidence, and join reports.

A case marked `"intentional_failure": true` (e.g. `case_006_broken`, the
deliberate failure-isolation demo) is called out in the CLI output, so the
pass count reads as designed behaviour rather than a defect; unexpected
failures are flagged separately for inspection.

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

`tests/test_prompt_injection.py` documents the prompt-injection defense
properties (inert cell data, origin-independent plan validation). Mock LLM
mode keeps the whole pipeline offline and deterministic.

## Prompt Injection & Untrusted Input

The pipeline treats three distinct untrusted inputs — typed questions, text
embedded in uploaded CSV cells, and LLM output — and none of them can turn
into executable behavior:

- **Dataset cell values are never sent to the LLM.** The planner's prompt
  contains only the dataset/field schema and the user's question text (see
  `app/services/llm_planner.py`; enforced by
  `tests/test_llm_planner.py::test_llm_receives_only_schema_not_row_data`).

- **The LLM is never a trusted source of executable instructions** — only of
  plan *suggestions*. Any plan it produces is passed through the exact same
  allow-list validator (`validate_plan()`) as a manually-submitted plan on
  `POST /plans/validate`: dataset allow-list, join registry, per-dataset field
  allow-lists, and the `MAX_LIMIT` cap. A grep confirms both consumers of
  `execute_plan()` (`app/routers/runs.py` `/query`,
  `app/routers/chat.py` turns endpoint) validate immediately before
  executing — no path bypasses `validate_plan()`. Proven by
  `tests/test_prompt_injection.py`: a hostile `limit: 999999` plan is rejected
  with `limit_exceeded` no matter where it "came from".

- **Malicious text in a data cell can only ever surface as inert output
  data.** The executor is column arithmetic and boolean masks — it cannot
  parse a cell value into an instruction, filter, or plan. A cell containing
  `"IGNORE ALL PREVIOUS INSTRUCTIONS..."` flows through to result rows
  byte-for-byte (asserted in `test_poisoned_cell_flows_through_as_inert_data`).

## Design trade-offs & known limitations

Trade-offs (deliberate choices, with their costs):

- **Mock planner by default.** Deterministic, offline, reproducible — but its
  keyword coverage is narrow; questions it doesn't recognise fall through to a
  generic `describe` plan. Rich phrasing understanding would need the real
  LLM path.
- **Heuristic gates instead of NLU.** The intent/capability classifiers are
  keyword-based: predictable and unit-testable, but paraphrases can be missed.
  Failure mode is deliberately conservative — ambiguity produces a
  clarification request or refusal, never a guess.
- **Refusals/rejections are HTTP 200.** A refused question or rejected plan is
  a *modelled outcome* persisted as a turn, not a protocol error; only
  malformed requests get 4xx/5xx. Clients must read `turn.status`.
- **Evidence-first answers.** `answer_text` is a short template summary; the
  JSON evidence is the source of truth. Auditability was prioritised over
  polished prose.

Known limitations:

- **Return-rate signal is thin in the sample data.** A "return" is an order
  with `revenue < 0` (there is no returns column), and the generated samples
  contain few such rows — a region-scoped answer can rest on a single return
  (e.g. "Footwear 6.2%" = 1 of 16 West orders). Grounded, but directional.
- **Near-duplicate category names are not merged** (`Electronic` vs
  `Electronics`): cleaning normalises case/whitespace but does not fuzzy-match
  category labels, so they remain separate rows in analytics.
- Region carry-forward covers **equality constraints only**; range filters
  (e.g. `stock_quantity < 10`) are intentionally not carried, since the flat
  `active_filters` shape would misrepresent them as equalities.
- `stockout_and_ageing` cannot be region-scoped: the inventory dataset has no
  region column.
- SQLite + `create_all` only: delete `retail.db` after any model change; no
  migrations.
- The frontend intentionally has no router/state library and no polling:
  analytics is a snapshot until re-clicked after a re-clean.
