# CLAUDE.md — FinQuery

This file tells Claude Code how to work in this repo.

---

## What This Project Is

FinQuery is an OpenEnv-compatible RL environment. It simulates a financial data terminal where an LLM agent must fetch data via tools, reason across multiple steps, and submit verified financial answers.

The environment is a **FastAPI server** exposed via HTTP endpoints following the OpenEnv spec.

---

## Project Structure

```
finquery/
├── finquery/
│   ├── models.py                # ALL Pydantic models live here. Edit this first.
│   └── client.py                # HTTP client. Mirrors models.py types.
├── server/
│   ├── app.py                   # FastAPI app. All routes registered here.
│   ├── finquery_environment.py  # Core logic: reset(), step(), state()
│   ├── _baseline_runner.py      # OpenAI baseline agent
│   ├── data/
│   │   ├── financials.json      # SOURCE OF TRUTH for all financial data
│   │   └── sectors.json         # SOURCE OF TRUTH for sector medians
│   ├── tools/                   # One file per tool. Pure functions, no state.
│   ├── graders/                 # One file per task. Deterministic. No LLM calls.
│   └── rewards/
│       └── reward_engine.py     # Dense per-step reward logic
├── baseline.py                  # OpenAI API baseline runner. Must stay runnable.
├── openenv.yaml                 # Manifest. Keep in sync with models.py.
└── pyproject.toml               # Dependencies.
```

---

## Critical Rules

### Data is synthetic and fixed
`server/data/financials.json` and `server/data/sectors.json` are the single source of truth. **Never** make external API calls (Yahoo Finance, Alpha Vantage, etc.) at runtime. All data is pre-built and deterministic. If you add a new ticker or year, add it to the JSON files first.

### Graders must be deterministic
Graders in `server/graders/` must never call an LLM for scoring. All scoring is pure math against ground truth values from `financials.json`. If you find yourself writing `openai.chat.completions.create()` in a grader file, stop — that is wrong.

### Reward engine is separate from graders
- `reward_engine.py` handles **per-step** rewards (relevance of tool calls, intermediate computation correctness, efficiency)
- `graders/` handle **terminal** rewards (accuracy of final submitted answer)
- Never mix these. The grader only runs when `action_type == "submit_answer"`.

### Models.py is the contract
`FinQueryAction`, `FinQueryObservation`, and `FinQueryState` in `models.py` define the API contract. If you change a field:
1. Update `models.py`
2. Update `client.py` to match
3. Update `openenv.yaml` if the action schema changed
4. Update `README.md` Action/Observation Space section

### All tools are pure functions
Files in `server/tools/` take `(ticker: str, year: int, data: dict)` and return a dict. They do not read from disk themselves — the environment passes the pre-loaded data dict to them.

---

## Running Locally

```bash
# Install
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Validate OpenEnv spec
openenv validate
```

---

## Baseline Script

`baseline.py` must:
- Read `OPENAI_API_KEY` from environment variables only — never hardcode
- Run all 3 tasks sequentially
- Print scores in this format:

```
Task 1 (easy)   — score: 0.71
Task 2 (medium) — score: 0.44
Task 3 (hard)   — score: 0.28
```

- Exit with code 0 on success, non-zero on error
- Complete within 5 minutes total

---

## Required Endpoints

These must all return 200 and correct responses:

| Endpoint | What it must do |
|---|---|
| `POST /reset` | Returns `FinQueryObservation` with task_description populated |
| `POST /step` | Accepts `FinQueryAction`, returns `StepResult` with reward + done |
| `GET /state` | Returns `FinQueryState` for current episode |
| `GET /tasks` | Returns list of all 3 tasks with their action schema |
| `POST /grader` | Accepts completed episode, returns score breakdown |
| `POST /baseline` | Runs baseline agent on all 3 tasks, returns scores |

---

## Data Schema

### financials.json structure

```json
{
  "AAPL": {
    "2020": {
      "income_statement": {
        "revenue": 274515,
        "cogs": 169559,
        "gross_profit": 104956,
        "operating_income": 66288,
        "net_income": 57411,
        "eps": 3.28
      },
      "balance_sheet": {
        "total_assets": 323888,
        "total_liabilities": 258549,
        "total_equity": 65339,
        "cash": 38016,
        "total_debt": 112436
      },
      "cash_flow": {
        "operating_cf": 80674,
        "investing_cf": -45977,
        "financing_cf": -86820,
        "fcf": 73365,
        "capex": 7309
      }
    }
  }
}
```

All monetary values in millions USD. All values are internally consistent (e.g. FCF = operating_cf - capex always holds).

### sectors.json structure

```json
{
  "technology": {
    "2023": {
      "pe_ratio": 22.1,
      "pb_ratio": 5.4,
      "ev_ebitda": 18.3,
      "roe": 0.21,
      "debt_equity": 0.48
    }
  }
}
```

---

## Reward Engine Logic

```python
STEP_REWARDS = {
    "relevant_fetch": +0.05,
    "irrelevant_fetch": -0.02,
    "duplicate_fetch": -0.01,
    "correct_compute": +0.10,
    "blind_submit": -0.05,
}
```

- Terminal reward: scaled accuracy from grader, clipped to [0.0, 0.70]
- Efficiency bonus: +0.10 if steps_taken <= 0.6 * max_steps
- Total episode reward: clipped to [0.0, 1.0]

Relevance of a fetch is determined by checking if the fetched `(tool, ticker, year)` combination appears in the task's `required_fetches` list. This is pre-computed per task — not LLM-judged.

---

## Common Mistakes to Avoid

1. **Do not add LLM calls to graders.** Graders must be pure functions.
2. **Do not import from `server/` in `client.py`.** The client ships separately.
3. **Do not hardcode ticker lists in environment logic.** Load from `financials.json` keys dynamically.
4. **Do not reset episode state in `state()`.** `state()` is read-only.
5. **Do not let `step()` raise unhandled exceptions.** All invalid actions must return an observation with `tool_error` set, reward of 0, and `done=False`.
6. **Do not change max_steps without updating `openenv.yaml`.** They must match.
