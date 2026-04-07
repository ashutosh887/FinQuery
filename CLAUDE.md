# CLAUDE.md -- FinQuery

> An OpenEnv-compatible RL environment simulating a financial data terminal for training agents on multi-step analytical reasoning.

This file tells Claude Code exactly how to work in this repo. Read it fully before touching any file.

---

## What This Project Is

FinQuery is an OpenEnv-compatible RL environment. It simulates a financial data terminal where an LLM agent must fetch data via tools, reason across multiple steps, and submit verified financial answers.

The environment is a **FastAPI server**, exposed via WebSocket and HTTP endpoints following the OpenEnv spec. Episode state is persisted via SQLite through `server/database.py`.

---

## Actual Project Structure

```
finquery/
├── baseline.py                    # OpenAI API baseline runner -- must stay runnable
├── CLAUDE.md                      # This file
├── openenv.yaml                   # Manifest -- keep in sync with models.py
├── pyproject.toml                 # Dependencies
├── Dockerfile                     # HF Spaces deployment (port 7860)
├── LICENSE
├── README.md
├── finquery/
│   ├── __init__.py                # Exports FinQueryEnv, FinQueryAction, FinQueryObservation, FinQueryState
│   ├── client.py                  # FinQueryEnv HTTP client -- no server imports
│   └── models.py                  # ALL Pydantic models -- edit this first, then propagate
├── server/
│   ├── __init__.py
│   ├── app.py                     # FastAPI app -- all routes + WebSocket + CORS
│   ├── finquery_environment.py    # Core: reset(), step(), state() -- concurrent episodes
│   ├── database.py                # SQLite -- episode/leaderboard persistence
│   ├── _baseline_runner.py        # Called internally by /baseline endpoint
│   ├── data/
│   │   ├── financials.json        # SOURCE OF TRUTH -- 12 tickers, 5 years, never fetch externally
│   │   └── sectors.json           # SOURCE OF TRUTH -- 4 sector median benchmarks
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── income_statement.py
│   │   ├── balance_sheet.py
│   │   ├── cash_flow.py
│   │   ├── price_history.py
│   │   ├── ratios.py
│   │   └── sector_compare.py
│   ├── graders/
│   │   ├── __init__.py
│   │   ├── task1_grader.py        # Deterministic -- no LLM calls ever
│   │   ├── task2_grader.py
│   │   └── task3_grader.py
│   └── rewards/
│       ├── __init__.py
│       └── reward_engine.py       # Per-step reward logic -- separate from graders
└── scripts/
    └── validate_data.py           # Data consistency checker
```

---

## Critical Rules

### 1. Data is synthetic and fixed
`server/data/financials.json` and `server/data/sectors.json` are the single source of truth. **Never** make external API calls (Yahoo Finance, Alpha Vantage, any financial API) at runtime. If a ticker or year is missing from the JSON, return an error -- do not fetch it live.

### 2. Graders must be deterministic -- no LLM calls
Files in `server/graders/` score against ground truth from `financials.json` using pure math. If you find yourself writing `openai.` anywhere in a grader file, stop -- that is wrong. Graders must return the same score for the same input every single time.

### 3. Reward engine is separate from graders
- `server/rewards/reward_engine.py` -- **per-step** rewards (tool relevance, duplicate detection, intermediate computation)
- `server/graders/` -- **terminal** rewards (final answer accuracy, called only on `submit_answer`)

Never mix these. Grader only runs when `action_type == "submit_answer"`.

### 4. models.py is the API contract
`FinQueryAction`, `FinQueryObservation`, `FinQueryState` define everything. If you change any field:
1. Update `finquery/models.py`
2. Update `finquery/client.py` to match
3. Update `openenv.yaml` if action schema changed
4. Update README.md Action/Observation Space section

### 5. Tools are pure functions
Files in `server/tools/` receive `(ticker, year, data_dict)` and return a dict. They do not read from disk themselves -- the environment passes pre-loaded data to them. No global state. No side effects.

### 6. client.py cannot import from server/
The client ships as a standalone package. It cannot depend on server code. If you need a shared type, it lives in `finquery/models.py` only.

### 7. step() never raises
All invalid actions must return an observation with `tool_error` set, `reward=0`, and `done=False`. Never let `step()` propagate an unhandled exception to the caller.

### 8. state() is read-only
`state()` returns current `FinQueryState` and never modifies episode state.

### 9. Episodes use episode_id for concurrency
The environment supports concurrent episodes. `/reset` returns an `episode_id` and `/step` requires one. Finished episodes are cleaned from memory and persisted to SQLite.

### 10. Port is 7860 for HuggingFace
The Dockerfile binds to port 7860 (HF Spaces requirement). Local dev runs on 8000.

---

## Running Locally

```bash
# Install
pip install -e .

# Run server on 8000 for local dev
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Validate data
python scripts/validate_data.py

# Validate OpenEnv spec
openenv validate
```

---

## Docker

```bash
docker build -t finquery .
docker run -p 8000:7860 finquery

# Smoke test
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/tasks
```

---

## Baseline Script

`baseline.py` must:
- Read `OPENAI_API_KEY` from environment only -- never hardcode
- Run all 3 tasks sequentially
- Exit code 0 on success, non-zero on any error
- Complete within 5 minutes total

---

## Required Endpoints -- All Must Return 200

| Endpoint | Method | What it must return |
|---|---|---|
| `/reset` | POST | `episode_id` + `FinQueryObservation` with `task_description` populated |
| `/step` | POST | Requires `episode_id` + `FinQueryAction`, returns observation + reward + done |
| `/state` | GET | `FinQueryState` for an episode (query param: `episode_id`) |
| `/tasks` | GET | List of 3 tasks with `id`, `name`, `difficulty`, `action_schema` |
| `/grader` | POST | Score breakdown dict with `score` in `(0.01, 0.99)` |
| `/baseline` | POST | Runs baseline agent, returns scores |
| `/history` | GET | Episode history from SQLite |
| `/leaderboard` | GET | Top scores grouped by agent + task |
| `/health` | GET | `{"status": "healthy"}` |
| `/ws` | WebSocket | Real-time reset/step/state over persistent connection |

---

## financials.json Schema

Each ticker/year entry contains:

```json
{
  "income_statement": { "revenue", "cogs", "gross_profit", "operating_income", "net_income", "eps" },
  "balance_sheet": { "total_assets", "total_liabilities", "total_equity", "cash", "total_debt" },
  "cash_flow": { "operating_cf", "investing_cf", "financing_cf", "fcf", "capex" },
  "price": { "open", "close", "high", "low", "avg_price" },
  "shares_outstanding": int,
  "ratios": { "pe_ratio", "pb_ratio", "ev_ebitda", "roe", "roa", "debt_equity", "current_ratio", "gross_margin", "net_margin", "fcf_margin" }
}
```

**Invariants (enforced by `scripts/validate_data.py`):**
- `gross_profit = revenue - cogs`
- `total_assets = total_liabilities + total_equity`
- `fcf = operating_cf - capex`
- `gross_margin = gross_profit / revenue` (within 0.001)
- `net_margin = net_income / revenue` (within 0.001)

---

## Reward Constants

```python
RELEVANT_FETCH_REWARD    = +0.05
IRRELEVANT_FETCH_PENALTY = -0.02
DUPLICATE_FETCH_PENALTY  = -0.01
CORRECT_COMPUTE_REWARD   = +0.10
BLIND_SUBMIT_PENALTY     = -0.05
MAX_TERMINAL_REWARD      =  0.70
EFFICIENCY_BONUS         = +0.10
EFFICIENCY_THRESHOLD     =  0.60   # steps_taken / max_steps must be <= this
```

---

## Common Mistakes to Avoid

1. **LLM calls in graders.** Never.
2. **Importing `server/` code in `finquery/client.py`.** Never.
3. **Hardcoding ticker lists in environment logic.** Load from `financials.json` keys dynamically.
4. **Using `eval()` in the compute tool.** Use a safe AST-based math parser.
5. **Letting `step()` raise.** All errors -> `tool_error` in observation, `reward=0`, `done=False`.
6. **Changing `max_steps` without updating `openenv.yaml`.** They must match.

---

## Pre-Submission Checklist

```bash
python scripts/validate_data.py && echo "Data OK"
openenv validate && echo "OpenEnv OK"
docker build -t finquery . && echo "Docker OK"
```

---

## Phase 2 Validation Fixes

### Score clamping rule

All task/episode scores returned to the caller must be strictly in the open interval `(0, 1)` — never exactly `0.0` or `1.0`. Clamp at the outermost return point:

```python
score = max(0.01, min(0.99, score))
```

**Where clamping is applied (defense in depth):**

- `server/graders/task1_grader.py` — both error return (line 21) and normal return (line 32)
- `server/graders/task2_grader.py` — error return (line 39) and final return (line 66)
- `server/graders/task3_grader.py` — error return (line 50) and final return (line 79)
- `server/rewards/reward_engine.py` — `compute_episode_total()` clamps to `[0.01, 0.99]`
- `server/app.py` — `/grader` endpoint clamps before returning `GraderResponse`
- `server/finquery_environment.py` — clamps `grader_score` (line 247), `clamped_total` (line 270), and `clamped_reward` on max_steps failure (line 330)
- `inference.py` — clamps score printed to stdout (line 180) and exception fallback (line 190)

**Do NOT clamp** per-step intermediate rewards (`+0.05`, `-0.02`, etc.) — only final task/episode scores.

### inference.py environment variable contract

```python
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
```

- Must use `from openai import OpenAI` — no httpx/requests for LLM calls
- Client: `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
- All completions: `model=MODEL_NAME`
- No hardcoded API keys, model names, or API URLs anywhere

### Stdout log format

Strict `[START]`, `[STEP]`, `[END]` format via `print()` with `flush=True`. Never `logging` or `stderr`.

```
[START] task=<task_id> env=finquery model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
```

### Infra constraints

- 2 vCPU, 8 GB RAM
- Inference must complete in < 20 minutes
- Docker image must build cleanly on constrained machine

### Danger zones — things that silently fail Phase 2

- Returning score `0.0` or `1.0` on any path (malformed input, error, perfect answer)
- Hardcoded API keys, model names, or `api.openai.com` in `inference.py`
- `print()` to stderr or using `logging` for `[START]`/`[STEP]`/`[END]` lines
- Docker build failing or OOMing on 8 GB machine
- inference.py runtime exceeding 20 minutes
