---
title: FinQuery
emoji: "\U0001F4CA"
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
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/ashutosh887/FinQuery)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

**FinQuery** is an RL environment where an agent operates like a financial analyst at a data terminal. The agent must decide *which data to fetch, in what order, and how to combine it* to answer verifiable financial questions.

Every `reset()` call generates a **unique** question through procedural task generation. The agent interacts with deterministic financial data tools across multi-step episodes with dense per-step rewards.

**Data coverage:** 25 companies across 7 sectors, 9 years (2017-2025), producing tens of thousands of unique training episodes.

---

## Data Coverage

| Sector | Companies | Count |
|---|---|---|
| Technology | AAPL, MSFT, GOOGL, META, NVDA, ORCL, CRM | 7 |
| Automotive | TSLA, F, GM, TM | 4 |
| Banking | JPM, BAC, WFC, GS | 4 |
| Retail | AMZN, WMT, COST | 3 |
| Healthcare | JNJ, UNH, PFE | 3 |
| Energy | XOM, CVX | 2 |
| Industrials | CAT, BA | 2 |

**Years:** 2017-2025 (9 years)
**Total records:** 25 tickers x 9 years = 225 financial records, each with 5 financial statements and 10 computed ratios.

All monetary figures are in millions USD. Data is synthetic but internally consistent.

---

## Task Variety

Each task is **procedurally generated** on every `reset()`. No two episodes are identical.

| Difficulty | Randomized Parameters | Unique Tasks |
|---|---|---|
| Easy | 25 tickers x 9 years x 11 metrics | ~2,475 |
| Medium | 7 sectors x company combos x 5 ratios x 9 years | hundreds |
| Hard | C(25,3) companies x year windows x 6 anomaly patterns | thousands |
| Composite | Weighted mix of all difficulties | unlimited |

---

## Tasks

### Task 1 -- Easy: Single-Metric Computation

Compute a financial metric for a randomly selected company and year. Metrics: net profit margin, gross margin, operating margin, debt-to-equity, ROA, ROE, FCF margin, annual price change, capex-to-revenue, EPS growth, revenue growth.

**Example prompts:**
> *"What was Apple's net profit margin for fiscal year 2022?"*
> *"What was NVIDIA's return on equity (ROE) for fiscal year 2024?"*
> *"What was Pfizer's year-over-year revenue growth rate for fiscal year 2022?"*

**Grader:** |error| < 0.05 = 0.99, < 0.50 = 0.50, else 0.01

### Task 2 -- Medium: Multi-Company Ratio Comparison

Compare a financial ratio across companies within a sector against the sector median. 5 ratios: P/E, P/B, EV/EBITDA, ROE, debt-to-equity.

**Example prompts:**
> *"Among Apple, Microsoft, and NVIDIA, which had the most favorable P/E ratio relative to the Technology sector median in 2023?"*
> *"Between ExxonMobil and Chevron, which had the more favorable debt-to-equity ratio relative to the Energy sector median in 2022?"*

**Grader:** correct company (0.40) + correct delta (0.40) + efficiency (0.20)

### Task 3 -- Hard: Multi-Year Anomaly Detection

Detect financial anomaly patterns across 3 companies over a 3-5 year window. 6 patterns: negative FCF + high P/E, high debt + low ROA, negative income + high P/B, negative operating CF + low margins, cash burn + price decline, high P/E + low ROA.

**Example prompts:**
> *"Among Boeing, Tesla, and Ford -- which had negative free cash flow in at least 2 of 5 years from 2019-2023, AND had a P/E ratio above 30 in any of those years?"*
> *"Among JPMorgan, Bank of America, and Wells Fargo -- which had a debt-to-equity ratio above 2.0 in at least 2 of 4 years from 2020-2023?"*

**Grader:** correct companies (0.30) + condition A years (0.30) + condition B years (0.30) + efficiency (0.10)

### Composite -- Mixed Difficulty

Weighted mix of easy/medium/hard tasks for curriculum learning. Configure via `task_specs` parameter.

---

## Configurable Reset

```python
# Basic
POST /reset {"task_id": "task1_easy"}

# Seed for reproducibility
POST /reset {"task_id": "task1_easy", "seed": 42}

# Batch: generate N tasks, iterate with bare resets
POST /reset {"task_id": "task2_medium", "seed": 42, "size": 50}
POST /reset {}  # next question from batch

# Composite: weighted difficulty mixing
POST /reset {"task_id": "composite", "size": 30, "task_specs": [
    {"difficulty": "easy", "weight": 3},
    {"difficulty": "medium", "weight": 2},
    {"difficulty": "hard", "weight": 1}
]}
```

---

## Action & Observation Space

### Action Space

```python
class FinQueryAction(BaseModel):
    action_type: Literal[
        "get_income_statement", "get_balance_sheet", "get_cash_flow",
        "get_price_history", "get_ratios", "compare_to_sector",
        "compute", "submit_answer"
    ]
    ticker: Optional[str]
    year: Optional[int]
    years: Optional[List[int]]
    metric: Optional[str]
    expression: Optional[str]
    answer: Optional[Any]
    reasoning: Optional[str]
```

### Observation Space

```python
class FinQueryObservation(BaseModel):
    task_description: str
    tool_result: Optional[Dict]
    tool_error: Optional[str]
    steps_taken: int
    steps_remaining: int
    tickers_queried: List[str]
    episode_status: Literal["ongoing", "answered", "failed_max_steps"]
    feedback: Optional[str]
    task_metadata: Optional[Dict]  # difficulty, companies, years, metric type
```

---

## Tools Reference

| Tool | Parameters | Returns |
|---|---|---|
| `get_income_statement` | `ticker`, `year` | Revenue, COGS, gross profit, operating income, net income, EPS |
| `get_balance_sheet` | `ticker`, `year` | Total assets, liabilities, equity, cash, total debt |
| `get_cash_flow` | `ticker`, `year` | Operating CF, investing CF, financing CF, FCF, capex |
| `get_price_history` | `ticker`, `years` | Annual open/close/high/low/avg_price per year |
| `get_ratios` | `ticker`, `year` | P/E, P/B, EV/EBITDA, ROE, ROA, debt/equity, margins |
| `compare_to_sector` | `ticker`, `metric`, `year` | Value vs sector median + percentile rank |
| `compute` | `expression` | Safe arithmetic evaluation |
| `submit_answer` | `answer` | Triggers grader, returns score breakdown |

---

## Reward Function

Dense rewards issued at every step.

| Signal | Reward | Condition |
|---|---|---|
| Relevant fetch | +0.05 | Fetched data the task requires |
| Irrelevant fetch | -0.02 | Fetched data unrelated to task |
| Duplicate fetch | -0.01 | Same tool + ticker + year called twice |
| Correct intermediate | +0.10 | `compute` result matches expected value |
| Blind submit | -0.05 | `submit_answer` with no prior data fetches |
| Terminal (accuracy) | 0.01-0.69 | Scaled from grader score |
| Efficiency bonus | +0.10 | Completed in <= 60% of max steps |

Total episode reward clipped to `(0.01, 0.99)`.

---

## Baseline Scores

Baseline agent: `gpt-4o-mini`, zero-shot. Averaged over 5 random episodes per task.

| Task | Difficulty | Score |
|---|---|---|
| Single-Metric Computation | Easy | ~0.71 |
| Multi-Company Ratio Comparison | Medium | ~0.44 |
| Multi-Year Anomaly Detection | Hard | ~0.28 |

Scores vary per episode since tasks are procedurally generated.

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

---

## API Endpoints

Base URL: `https://ashutosh887-finquery.hf.space`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment metadata |
| `/reset` | POST | Start new episode (supports seed, size, task_specs) |
| `/step` | POST | Take action, returns observation + reward + done |
| `/state` | GET | Episode metadata |
| `/tasks` | GET | All tasks with action schema |
| `/grader` | POST | Score an answer against ground truth |
| `/baseline` | POST | Run baseline agent |
| `/history` | GET | Episode history |
| `/leaderboard` | GET | Top scores by agent |
| `/health` | GET | `{"status": "healthy"}` |
| `/ws` | WebSocket | Real-time episode interaction |

---

## License

MIT
