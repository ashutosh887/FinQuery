# FinQuery

> An OpenEnv-compatible RL environment simulating a financial data terminal for training agents on multi-step analytical reasoning.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

**FinQuery** is an RL environment where an agent operates like a financial analyst at a data terminal. The agent does not receive a pre-packaged dataset — it must decide *which data to fetch, in what order, and how to combine it* to answer verifiable financial questions.

The agent interacts with a suite of deterministic financial data tools across multi-step episodes. Rewards are issued at every step, making the reward function dense across the full trajectory rather than sparse at the end.

The hard task reliably defeats frontier models that hallucinate intermediate values under multi-hop reasoning pressure — making this a meaningful training signal for financial reasoning agents.

---

## Motivation

Bloomberg terminals charge ~$25,000/user/year. Every investment firm, hedge fund, and financial research team in the world pays this. The bottleneck is not data — it's the analyst skill to navigate and reason across that data.

FinQuery provides the first open RL training environment for this skill. Epoch AI's January 2026 research report on frontier lab RL environment procurement explicitly cites a "Bloomberg terminal clone" as a key domain labs are actively building. FinQuery is the open-source version of that.

---

## Action & Observation Space

### Action Space

```python
class FinQueryAction(BaseModel):
    action_type: Literal[
        "get_income_statement",
        "get_balance_sheet",
        "get_cash_flow",
        "get_price_history",
        "get_ratios",
        "compare_to_sector",
        "compute",
        "submit_answer"
    ]
    ticker: Optional[str] = None          # e.g. "AAPL", "MSFT"
    year: Optional[int] = None            # e.g. 2023
    years: Optional[List[int]] = None     # e.g. [2020, 2021, 2022, 2023]
    metric: Optional[str] = None          # e.g. "revenue", "net_income", "fcf"
    expression: Optional[str] = None      # arithmetic expression for compute
    answer: Optional[Any] = None          # final answer for submit_answer
    reasoning: Optional[str] = None       # agent's chain of thought (logged, not graded)
```

### Observation Space

```python
class FinQueryObservation(BaseModel):
    task_description: str                  # what the agent must solve
    tool_result: Optional[Dict[str, Any]]  # result of last action
    tool_error: Optional[str]              # error message if action was invalid
    steps_taken: int
    steps_remaining: int
    tickers_queried: List[str]             # audit trail of what was fetched
    episode_status: Literal["ongoing", "answered", "failed_max_steps"]
    feedback: Optional[str]               # non-graded hint on last action relevance
```

### State

```python
class FinQueryState(BaseModel):
    episode_id: str
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    fetched_data: Dict[str, Any]          # accumulated data across all fetches
    answer_submitted: bool
    score_so_far: float
```

---

## Tools Reference

All tools return deterministic JSON from pre-built synthetic datasets. No external API calls are made at runtime.

| Tool | Parameters | Returns | Example |
|------|-----------|---------|---------|
| `get_income_statement` | `ticker`, `year` | Revenue, COGS, gross profit, operating income, net income, EPS | `{"revenue": 394328, "net_income": 96995, ...}` |
| `get_balance_sheet` | `ticker`, `year` | Total assets, liabilities, equity, cash, debt | `{"total_assets": 352583, "cash": 29965, ...}` |
| `get_cash_flow` | `ticker`, `year` | Operating CF, investing CF, financing CF, FCF, capex | `{"operating_cf": 110543, "fcf": 99584, ...}` |
| `get_price_history` | `ticker`, `years` | Annual open/close/high/low, avg price per year | `{"2020": {"close": 132.69}, ...}` |
| `get_ratios` | `ticker`, `year` | P/E, P/B, EV/EBITDA, ROE, ROA, debt/equity | `{"pe_ratio": 28.4, "roe": 0.147, ...}` |
| `compare_to_sector` | `ticker`, `metric`, `year` | Returns metric value vs sector median and percentile rank | `{"value": 28.4, "sector_median": 22.1, "percentile": 71}` |
| `compute` | `expression` | Evaluates arithmetic expression, returns float | `{"result": 0.1847}` |
| `submit_answer` | `answer` | Triggers grader, returns score breakdown | `{"score": 0.87, "breakdown": {...}}` |

All financial figures are in millions USD unless otherwise noted. All data is synthetic but internally consistent across years and tickers.

---

## Tasks

### Task 1 -- Easy: Single-Metric Lookup with Computation

**Description:** Given a single company, compute a specific financial metric that requires fetching data and one calculation step.

**Example prompt:**
> *"What was Apple's net profit margin for fiscal year 2022? Express as a percentage rounded to 2 decimal places."*

**Expected trajectory:** `get_income_statement` -> `compute` -> `submit_answer` (3 steps)

**Difficulty rationale:** Single company, single year, formula is standard. Tests basic tool use and arithmetic.

---

### Task 2 -- Medium: Multi-Company Ratio Comparison

**Description:** Compare a specific valuation ratio across 3 companies against their sector median.

**Example prompt:**
> *"Among Microsoft, Google, and Meta, which company had the most favorable EV/EBITDA relative to the tech sector median in 2023? By how many points did it differ?"*

**Expected trajectory:** `get_ratios` x 3 -> `compare_to_sector` x 3 -> `compute` -> `submit_answer` (8 steps)

**Difficulty rationale:** Requires coordinating data across multiple companies and understanding sector relative valuation.

---

### Task 3 -- Hard: Multi-Year Anomaly Detection

**Description:** Identify which company in a set had a specific anomalous pattern across 4 years of financial history, requiring cross-referencing cash flow and ratio data.

**Example prompt:**
> *"Among Tesla, Ford, and GM -- which company had negative free cash flow in at least 2 of the 4 fiscal years from 2020-2023, AND had a P/E ratio above 30 in any of those same years?"*

**Expected trajectory:** `get_cash_flow` x 12 + `get_ratios` x 12 -> cross-reference -> `submit_answer` (25-30 steps)

**Difficulty rationale:** Frontier models hallucinate intermediate values under multi-hop cross-referencing. GPT-4o baseline scores ~0.28 on this task.

---

## Reward Function

Rewards are dense -- issued at every step, not just at episode end.

```
Step-level rewards:
  +0.05  Fetching data directly relevant to the task
  +0.10  Correct intermediate computation via compute tool
  -0.02  Fetching data not relevant to the task
  -0.05  Calling submit_answer with no prior data fetches (blind guessing)
  -0.01  Duplicate fetch (same ticker + year + tool called twice)

Terminal rewards (on submit_answer):
  +0.00 to +0.70  Scaled by grader accuracy score
  +0.10           Efficiency bonus if completed in <= 60% of max steps
  -0.10           Penalty if submitted without minimum required data fetches

Episode total: sum of all step rewards + terminal reward, clipped to [0.0, 1.0]
```

---

## Baseline Scores

Baseline agent: `gpt-4o-mini` via OpenAI API, zero-shot, no chain-of-thought.

| Task | Difficulty | Baseline Score | Notes |
|------|-----------|---------------|-------|
| Net Profit Margin | Easy | 0.71 | Occasionally miscalculates from wrong revenue line |
| Multi-Company EV/EBITDA | Medium | 0.44 | Frequently confuses sector delta direction |
| Multi-Year FCF + P/E Anomaly | Hard | 0.28 | Hallucinates ~40% of intermediate values |

---

## Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://github.com/ashutosh887/finquery
cd finquery

# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Use as Client (HTTP)

```python
from finquery import FinQueryEnv, FinQueryAction

with FinQueryEnv(base_url="http://localhost:8000").sync() as env:
    # Start an episode
    obs = env.reset(agent_name="my_agent")
    print(obs.episode_id)              # unique episode ID
    print(obs.observation.task_description)

    # Fetch data
    result = env.step(FinQueryAction(
        action_type="get_income_statement",
        ticker="AAPL",
        year=2022
    ))
    print(result.reward)  # +0.05 for relevant fetch

    # Submit answer
    result = env.step(FinQueryAction(
        action_type="submit_answer",
        answer=25.31
    ))
    print(result.reward)  # terminal reward
    print(result.done)    # True

    # Check leaderboard
    print(env.leaderboard())
```

### Use via WebSocket

```python
import json, websockets, asyncio

async def run():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "task_id": "task1_easy"}))
        result = json.loads(await ws.recv())
        episode_id = result["episode_id"]

        # Step
        await ws.send(json.dumps({
            "type": "step",
            "episode_id": episode_id,
            "action": {"action_type": "get_income_statement", "ticker": "AAPL", "year": 2022}
        }))
        result = json.loads(await ws.recv())
        print(result["reward"])

asyncio.run(run())
```

### Run Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline.py
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns `episode_id` + initial observation |
| `/step` | POST | Take action (requires `episode_id`), returns observation + reward + done |
| `/state` | GET | Episode metadata (query param: `episode_id`) |
| `/tasks` | GET | List all tasks with action schema |
| `/grader` | POST | Score an answer against ground truth |
| `/baseline` | POST | Run baseline agent on all tasks |
| `/history` | GET | Episode history (query params: `limit`, `task_id`) |
| `/leaderboard` | GET | Top scores by agent (query params: `limit`, `task_id`) |
| `/health` | GET | Health check |
| `/ws` | WebSocket | Real-time episode interaction |

---

## Project Structure

```
finquery/
├── finquery/
│   ├── models.py              # FinQueryAction, FinQueryObservation, FinQueryState
│   └── client.py              # FinQueryEnv HTTP client
├── server/
│   ├── app.py                 # FastAPI app + all endpoints + WebSocket
│   ├── database.py            # SQLite persistence (episodes + leaderboard)
│   ├── finquery_environment.py  # Core environment logic (concurrent episodes)
│   ├── _baseline_runner.py    # OpenAI baseline agent
│   ├── data/
│   │   ├── financials.json    # Synthetic financial dataset (10 companies, 4 years)
│   │   └── sectors.json       # Sector median benchmarks
│   ├── tools/
│   │   ├── income_statement.py
│   │   ├── balance_sheet.py
│   │   ├── cash_flow.py
│   │   ├── price_history.py
│   │   ├── ratios.py
│   │   └── sector_compare.py
│   ├── graders/
│   │   ├── task1_grader.py
│   │   ├── task2_grader.py
│   │   └── task3_grader.py
│   └── rewards/
│       └── reward_engine.py   # Dense per-step reward computation
├── baseline.py                # CLI baseline runner
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml
└── README.md
```

---

## License

MIT
