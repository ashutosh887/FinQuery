---
title: FinQuery
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - finance
  - agents
license: mit
short_description: OpenEnv RL environment for financial agent reasoning
---

# FinQuery

> An OpenEnv-compatible RL environment simulating a financial data terminal for training agents on multi-step analytical reasoning.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/ashutosh887/FinQuery)
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

## Project Structure

```
finquery/
├── finquery/
│   ├── __init__.py
│   ├── models.py                 # FinQueryAction, FinQueryObservation, FinQueryState
│   └── client.py                 # FinQueryEnv HTTP client
├── server/
│   ├── app.py                    # FastAPI app + all endpoints + WebSocket + CORS
│   ├── database.py               # SQLite persistence (episodes + leaderboard)
│   ├── finquery_environment.py   # Core environment (concurrent episodes)
│   ├── _baseline_runner.py       # OpenAI baseline agent
│   ├── data/
│   │   ├── financials.json       # Synthetic dataset — 12 tickers, 6 years
│   │   └── sectors.json          # Sector median benchmarks (4 sectors)
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
│       └── reward_engine.py      # Dense per-step reward computation
├── scripts/
│   └── validate_data.py          # Data consistency checker
├── inference.py                  # Hackathon submission inference script
├── baseline.py                   # CLI baseline runner
├── validation_script.sh          # OpenEnv submission validator
├── Dockerfile                    # HF Spaces deployment (port 7860)
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

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
    ticker: Optional[str] = None        # e.g. "AAPL", "MSFT"
    year: Optional[int] = None          # e.g. 2024
    years: Optional[List[int]] = None   # e.g. [2020, 2021, 2022, 2023, 2024]
    metric: Optional[str] = None        # e.g. "pe_ratio"
    expression: Optional[str] = None    # arithmetic for compute action
    answer: Optional[Any] = None        # final answer for submit_answer
    reasoning: Optional[str] = None     # chain of thought (logged, not graded)
```

### Observation Space

```python
class FinQueryObservation(BaseModel):
    task_description: str
    tool_result: Optional[Dict[str, Any]] = None
    tool_error: Optional[str] = None
    steps_taken: int
    steps_remaining: int
    tickers_queried: List[str]
    episode_status: Literal["ongoing", "answered", "failed_max_steps"]
    feedback: Optional[str] = None
```

### State

```python
class FinQueryState(BaseModel):
    episode_id: str
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    fetched_data: Dict[str, Any]
    answer_submitted: bool
    score_so_far: float
```

---

## Tools Reference

All tools return deterministic JSON from `server/data/`. No external API calls at runtime.

| Tool | Parameters | Returns |
|------|-----------|---------|
| `get_income_statement` | `ticker`, `year` | Revenue, COGS, gross profit, operating income, net income, EPS |
| `get_balance_sheet` | `ticker`, `year` | Total assets, liabilities, equity, cash, total debt |
| `get_cash_flow` | `ticker`, `year` | Operating CF, investing CF, financing CF, FCF, capex |
| `get_price_history` | `ticker`, `years` | Annual open/close/high/low/avg_price per year |
| `get_ratios` | `ticker`, `year` | P/E, P/B, EV/EBITDA, ROE, ROA, debt/equity, margins |
| `compare_to_sector` | `ticker`, `metric`, `year` | Value vs sector median + percentile rank + above_median |
| `compute` | `expression` | Safe arithmetic evaluation, returns float |
| `submit_answer` | `answer` | Triggers grader, returns score breakdown |

All monetary figures are in millions USD. Data is synthetic but internally consistent — `FCF = operating_cf - capex`, `gross_profit = revenue - cogs`, `gross_margin = gross_profit / revenue` always hold.

**Coverage:** 12 tickers (AAPL, MSFT, GOOGL, META, NVDA, TSLA, F, GM, JPM, BAC, AMZN, WMT) across 6 years (2019–2024) with 4 sectors (technology, automotive, banking, retail).

---

## Tasks

### Task 1 — Easy: Single-Metric Computation

**Description:** Compute a standard financial metric for a single company. Requires 1–2 data fetches and one calculation step.

**Example prompt:**
> *"What was Apple's net profit margin for fiscal year 2022? Express as a percentage rounded to 2 decimal places."*

**Expected trajectory:** `get_income_statement` → `compute` → `submit_answer` (3 steps, max 10)

**Grader:**

| Error range | Score |
|---|---|
| < 0.05% | 0.99 |
| < 0.50% | 0.50 |
| ≥ 0.50% | 0.01 |

---

### Task 2 — Medium: Multi-Company Ratio Comparison

**Description:** Compare a valuation ratio across 3 companies against their sector median. Identify which is most attractive.

**Example prompt:**
> *"Among Microsoft, Google, and Meta, which had the most favorable EV/EBITDA relative to the tech sector median in 2023? By how many points did it differ?"*

**Expected trajectory:** `get_ratios` ×3 → `compare_to_sector` ×3 → `compute` → `submit_answer` (~8 steps, max 20)

**Grader:**

| Dimension | Weight |
|---|---|
| Correct company identified | 0.40 |
| Correct sector delta (within 0.5) | 0.40 |
| Efficiency bonus (steps ≤ 10) | 0.20 |

---

### Task 3 — Hard: Multi-Year Anomaly Detection

**Description:** Identify anomalous financial patterns across multiple years for multiple companies, requiring cross-referencing cash flow and ratio data.

**Example prompt:**
> *"Among Tesla, Ford, and GM — which had negative free cash flow in at least 2 of 4 fiscal years 2020–2024, AND had a P/E ratio above 30 in any of those years?"*

**Expected trajectory:** `get_cash_flow` ×12 + `get_ratios` ×12 → cross-reference → `submit_answer` (~28 steps, max 40)

**Grader:**

| Dimension | Weight |
|---|---|
| Correct companies identified | 0.30 |
| Correct FCF-negative years (partial per company/year) | 0.30 |
| Correct P/E > 30 years (partial per company/year) | 0.30 |
| Efficiency bonus (steps ≤ 20) | 0.10 |

---

## Reward Function

Dense rewards issued at every step.

| Signal | Reward | Condition |
|--------|--------|-----------|
| Relevant fetch | +0.05 | Fetched data the task requires |
| Irrelevant fetch | −0.02 | Fetched data unrelated to task |
| Duplicate fetch | −0.01 | Same tool + ticker + year called twice |
| Correct intermediate | +0.10 | `compute` result matches expected value |
| Blind submit | −0.05 | `submit_answer` with no prior data fetches |
| Terminal (accuracy) | 0.01–0.69 | Scaled from grader score (clamped) |
| Efficiency bonus | +0.10 | Completed in ≤ 60% of max steps |

Total episode reward clipped to `(0.01, 0.99)`.

---

## Baseline Scores

Baseline agent: `gpt-4o-mini`, zero-shot, no chain-of-thought.

| Task | Difficulty | Score | Notes |
|------|-----------|-------|-------|
| Single-Metric Computation | Easy | 0.71 | Occasionally miscalculates from wrong revenue line |
| Multi-Company Ratio Comparison | Medium | 0.44 | Frequently confuses sector delta direction |
| Multi-Year Anomaly Detection | Hard | 0.28 | Hallucinates ~40% of intermediate values |

**Reproduce:**
```bash
export OPENAI_API_KEY=your_key
python baseline.py
```

---

## Setup & Usage

### Local

```bash
git clone https://github.com/ashutosh887/FinQuery.git
cd FinQuery
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t finquery .
docker run -p 8000:7860 finquery
curl -X POST http://localhost:8000/reset
```

### Client Usage

```python
from finquery import FinQueryEnv, FinQueryAction

with FinQueryEnv(base_url="https://ashutosh887-finquery.hf.space").sync() as env:
    obs = env.reset(agent_name="my_agent")
    print(obs.episode_id)
    print(obs.observation.task_description)

    result = env.step(FinQueryAction(
        action_type="get_income_statement",
        ticker="AAPL",
        year=2024
    ))
    print(result.reward)

    result = env.step(FinQueryAction(
        action_type="submit_answer",
        answer=26.50,
        reasoning="Net income / Revenue = 110712 / 417781 * 100"
    ))
    print(result.reward)
    print(result.done)
```

### WebSocket

```python
import json, websockets, asyncio

async def run():
    async with websockets.connect("wss://ashutosh887-finquery.hf.space/ws") as ws:
        await ws.send(json.dumps({"type": "reset", "task_id": "task1_easy"}))
        result = json.loads(await ws.recv())
        episode_id = result["episode_id"]

        await ws.send(json.dumps({
            "type": "step",
            "episode_id": episode_id,
            "action": {"action_type": "get_income_statement", "ticker": "AAPL", "year": 2024}
        }))
        result = json.loads(await ws.recv())
        print(result["reward"])

asyncio.run(run())
```

---

## API Endpoints

Base URL: `https://ashutosh887-finquery.hf.space`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment metadata and endpoint reference |
| `/reset` | POST | Start new episode, returns `episode_id` + initial observation |
| `/step` | POST | Take action (requires `episode_id`), returns observation + reward + done |
| `/state` | GET | Episode metadata (query param: `episode_id`) |
| `/tasks` | GET | All tasks with action schema |
| `/grader` | POST | Score an answer against ground truth |
| `/baseline` | POST | Run baseline agent on all tasks |
| `/history` | GET | Episode history (query params: `limit`, `task_id`) |
| `/leaderboard` | GET | Top scores by agent (query params: `limit`, `task_id`) |
| `/health` | GET | `{"status": "healthy"}` |
| `/ws` | WebSocket | Real-time episode interaction |

---

## License

MIT