# CLAUDE.md -- FinQuery

This file tells Claude Code how to work in this repo.

---

## What This Project Is

FinQuery is an OpenEnv-compatible RL environment. It simulates a financial data terminal where an LLM agent must fetch data via tools, reason across multiple steps, and submit verified financial answers.

The environment is a **FastAPI server** exposed via HTTP and WebSocket endpoints following the OpenEnv spec.

---

## Project Structure

```
finquery/
├── finquery/
│   ├── __init__.py              # Package exports
│   ├── models.py                # ALL Pydantic models live here. Edit this first.
│   └── client.py                # HTTP client. Mirrors models.py types.
├── server/
│   ├── app.py                   # FastAPI app. All routes + WebSocket + CORS.
│   ├── database.py              # SQLite persistence (episodes + leaderboard).
│   ├── finquery_environment.py  # Core logic: reset(), step(), state(). Concurrent episodes.
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
Graders in `server/graders/` must never call an LLM for scoring. All scoring is pure math against ground truth values from `financials.json`.

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
Files in `server/tools/` take `(ticker: str, year: int, data: dict)` and return a dict. They do not read from disk themselves -- the environment passes the pre-loaded data dict to them.

### Episodes use episode_id for concurrency
The environment supports concurrent episodes. `/reset` returns an `episode_id` and `/step` requires one. Do not use a single shared episode -- always pass the episode_id.

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

## Required Endpoints

All 10 must return correct responses:

| Endpoint | Method | What it must do |
|---|---|---|
| `/reset` | POST | Returns `episode_id` + `FinQueryObservation` with task_description populated |
| `/step` | POST | Accepts `episode_id` + `FinQueryAction`, returns observation + reward + done |
| `/state` | GET | Returns `FinQueryState` for an episode (query param: `episode_id`) |
| `/tasks` | GET | Returns list of all 3 tasks with their action schema |
| `/grader` | POST | Accepts task_id + answer, returns score breakdown |
| `/baseline` | POST | Runs baseline agent on all 3 tasks, returns scores |
| `/history` | GET | Returns recent episode history from SQLite |
| `/leaderboard` | GET | Returns top scores grouped by agent + task |
| `/health` | GET | Returns `{"status": "healthy"}` |
| `/ws` | WebSocket | Real-time reset/step/state over a single connection |

---

## Database

SQLite at `server/data/finquery.db` (auto-created on first startup, gitignored).

Tables:
- `episodes` -- one row per episode (episode_id, task_id, agent_name, score, status, timestamps)
- `leaderboard` -- one row per completed episode (agent_name, task_id, score, steps)

The DB is managed by `server/database.py`. `init_db()` is called in the FastAPI lifespan handler.

---

## Common Mistakes to Avoid

1. **Do not add LLM calls to graders.** Graders must be pure functions.
2. **Do not import from `server/` in `client.py`.** The client ships separately.
3. **Do not hardcode ticker lists in environment logic.** Load from `financials.json` keys dynamically.
4. **Do not reset episode state in `state()`.** `state()` is read-only.
5. **Do not let `step()` raise unhandled exceptions.** All invalid actions must return an observation with `tool_error` set, reward of 0, and `done=False`.
6. **Do not change max_steps without updating `openenv.yaml`.** They must match.
7. **Do not use a single shared episode.** Always pass `episode_id` to `step()` and `state()`.
