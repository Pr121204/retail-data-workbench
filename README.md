# retail-data-workbench

FastAPI service that ingests retail CSVs, profiles/cleans them, and answers
natural-language questions through a **plan → validate → execute** pipeline
with a full evidence trail. Includes a minimal React UI, a headless batch
evaluator, and a prompt-injection defense model that is proven by tests.

**Track: AI Engineering.** The differentiators here are the LLM-to-plan
translation layer (`llm_planner.py`) with mock/live modes and graceful fallback
on invalid JSON or API failure, the allow-list plan validator that is the
actual security boundary regardless of whether a plan came from a human or the
model, the multi-turn chat session with context carry-forward, the
prompt-injection defense suite (`tests/test_prompt_injection.py`), and the
batch evaluator's chat-quality assertions (`expected_status`, `answer_contains`,
`evidence_keys`, etc.). The background job mode, per-run ownership isolation,
and observability logging were added as supporting reliability infrastructure —
they hold the pipeline steady under the same conditions (retries, timeouts,
malformed input) that stress the AI layer, but they are not the specialization
being scored.

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
| Category variant merging (`Electronic`/`Electronics`) | code (`cleaning.py`) | mechanical equivalence only — deterministic, auditable merge map; synonyms are never merged |
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

Uploads are bounded by `MAX_UPLOAD_FILES` and `MAX_UPLOAD_BYTES` (configured
through environment variables, with conservative defaults). Only `.csv`
files are accepted, and duplicate dataset names are rejected before a run is
created.

Delete `retail.db` whenever the schema changed before starting the server —
`Base.metadata.create_all` only adds missing tables, it does not migrate.

### What you should see on a sample run

Upload the 5 sample CSVs, profile, clean: customers 80→76, stores 10→9,
orders 300→297, products 60→60 (the category-variant merge collapses
`Electronic` into `Electronics` without dropping product rows). Chat:
"revenue by category" → Electronics 11414.66 (62 orders, AOV 184.11), all 7
categories combined 51170.5; return-rate leader Home & Office 3.3% (1 of 30);
West-scoped: 146 of 297 orders, 24239.18, Footwear 6.2% (1 of 16).

### Long-running stages, ownership & observability

- **Background mode:** `POST /runs/{id}/profile?wait=false` and
  `POST /runs/{id}/clean?wait=false` return **202** immediately with a
  `running_profile` / `running_clean` state and execute as background tasks
  with a dedicated DB session. Any failure marks the run `failed` with a
  persisted `error_message` — state is never silently stuck. Default is
  `wait=true` (synchronous), so existing clients are unaffected.
- **Per-run ownership (data isolation):** pass `X-Run-Owner: <token>` at
  upload time and every subsequent call on that run must present the same
  header (403 otherwise). Runs uploaded without the token stay publicly
  addressable — a deliberate single-tenant/demo trade-off, documented here
  and exercised by `tests/test_ownership.py`.
- **Observability:** every HTTP request is logged with an `X-Request-Id`
  (honoured if supplied by a proxy), method, path, status and duration; the
  id is echoed in the response. Unhandled errors are logged with the same id
  before propagating.
- **Upload validation:** `.csv` extension, per-file and total size caps,
  duplicate-name rejection, and a pre-write parse + binary-sniff (NUL bytes)
  + header-only rejection — all before a run row is created, with structured
  4xx errors (`tests/test_upload.py`).

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
turns with evidence, and join reports. The committed case file includes edge
cases from the brief — an all-null column, a latin-1-encoded file, and a
100,000-row order file (generated deterministically by
`scripts/generate_edge_cases.py`) — alongside the intentional
failure-isolation demo.

Chat cases can assert `expected_status`, `min_result_rows`,
`result_row_count`, `plan_intent`, `plan_dataset`, `answer_contains`,
`evidence_keys`, `evidence_equals`, `expected_filters`, and `expected_join`.
Multi-turn cases use a `questions` list and can assert `expected_statuses`.
Unsupported expectation keys fail the case instead of being silently ignored.

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
- **Category-variant merging is mechanical, not semantic.** Labels equal after
  removing case/whitespace/punctuation — or differing only by a trailing
  plural — are merged to the most frequent spelling (`Electronic` →
  `Electronics`, 57 vs 5 on the sample data). Synonyms (`Tee`/`T-Shirt`) are
  never merged; the step reason discloses this limitation and the merge map
  travels in the plan for auditability. The proposal pass simulates earlier
  steps so rows made identical *by merging* are deduped in the same run —
  re-cleaning is idempotent with zero row loss on the sample data.

Known limitations:

- **Return-rate signal is thin in the sample data.** A "return" is an order
  with `revenue < 0` (there is no returns column), and the generated samples
  contain few such rows — a region-scoped answer can rest on a single return
  (e.g. "Footwear 6.2%" = 1 of 16 West orders). Grounded, but directional.
- Region carry-forward covers **equality constraints only**; range filters
  (e.g. `stock_quantity < 10`) are intentionally not carried, since the flat
  `active_filters` shape would misrepresent them as equalities.
- `stockout_and_ageing` cannot be region-scoped: the inventory dataset has no
  region column.
- SQLite + `create_all` only: delete `retail.db` after any model change; no
  migrations.
- The frontend intentionally has no router/state library and no polling:
  analytics is a snapshot until re-clicked after a re-clean.
