"""FinQuery Inference Script — Scaler OpenEnv Hackathon."""

import os
import json
from typing import List, Optional
from openai import OpenAI

SPACE_URL = os.getenv("SPACE_URL", "https://ashutosh887-finquery.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "finquery"
TASK_IDS = ["task1_easy", "task2_medium", "task3_hard"]
SUCCESS_THRESHOLD = 0.3

SYSTEM_PROMPT = """You are a financial analyst agent at a data terminal.

TOOLS:
- get_income_statement(ticker, year): revenue, cogs, gross_profit, operating_income, net_income, eps
- get_balance_sheet(ticker, year): total_assets, total_liabilities, total_equity, cash, total_debt
- get_cash_flow(ticker, year): operating_cf, investing_cf, financing_cf, fcf, capex
- get_price_history(ticker, years): open, close, high, low, avg_price per year
- get_ratios(ticker, year): pe_ratio, pb_ratio, ev_ebitda, roe, roa, debt_equity, gross_margin, net_margin, fcf_margin
- compare_to_sector(ticker, metric, year): value vs sector median + percentile
- compute(expression): safe arithmetic evaluation
- submit_answer(answer): submit final answer

TICKERS: AAPL, MSFT, GOOGL, META, NVDA, ORCL, CRM, TSLA, F, GM, TM, JPM, BAC, WFC, GS, AMZN, WMT, COST, JNJ, UNH, PFE, XOM, CVX, CAT, BA
YEARS: 2017-2025

Respond ONLY with valid JSON:
{
  "action_type": "get_income_statement|get_balance_sheet|get_cash_flow|get_price_history|get_ratios|compare_to_sector|compute|submit_answer",
  "ticker": "AAPL",
  "year": 2022,
  "years": [2020, 2021, 2022, 2023],
  "metric": "pe_ratio",
  "expression": "99803/394328*100",
  "answer": 25.31,
  "reasoning": "brief explanation"
}
Include only fields relevant to your chosen action_type. Fetch data before computing. Never guess.
For comparison tasks: fetch ratios for each company, compare to sector, identify the best.
For anomaly tasks: systematically check each company across each year.
For single-metric tasks: fetch the right statement, compute, submit.
"""


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_llm_action(client: OpenAI, task_description: str, history: List[dict]) -> tuple:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    if not history:
        messages.append({"role": "user", "content": task_description})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        return json.loads(raw), raw
    except Exception as e:
        fallback = {"action_type": "submit_answer", "answer": 0, "reasoning": f"error: {e}"}
        return fallback, json.dumps(fallback)


def run_episode(task_id: str, client: OpenAI) -> None:
    import httpx

    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        resp = httpx.post(
            f"{SPACE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        episode_id = data.get("episode_id")
        obs = data.get("observation", {})
        task_description = obs.get("task_description", "")
        max_steps = obs.get("steps_remaining", 40)
        done = data.get("done", False)

        history = [{"role": "user", "content": task_description}]

        for step in range(1, max_steps + 1):
            if done:
                break

            action_dict, action_raw = get_llm_action(client, task_description, history)
            history.append({"role": "assistant", "content": action_raw})
            error_msg = None
            try:
                step_resp = httpx.post(
                    f"{SPACE_URL}/step",
                    json={"episode_id": episode_id, "action": action_dict},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()

                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))
                step_obs = step_data.get("observation", {})
                tool_error = step_obs.get("tool_error")
                tool_result = step_obs.get("tool_result")

                context = {
                    "tool_result": tool_result,
                    "tool_error": tool_error,
                    "steps_remaining": step_obs.get("steps_remaining", 0),
                    "episode_status": step_obs.get("episode_status", "ongoing"),
                }
                history.append({"role": "user", "content": json.dumps(context)})

                if tool_error:
                    error_msg = tool_error

            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_raw, reward=reward, done=done, error=error_msg)

            if done:
                break

        total_reward = sum(rewards)
        score = max(0.01, min(0.99, total_reward))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        steps_taken = steps_taken or 0
        error_log = str(e)
        if steps_taken == 0:
            rewards = [0.0]
            log_step(step=1, action="reset_failed", reward=0.0, done=True, error=error_log)
            steps_taken = 1
        score = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN not set — export HF_TOKEN=your_token")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASK_IDS:
        run_episode(task_id=task_id, client=client)


if __name__ == "__main__":
    main()
