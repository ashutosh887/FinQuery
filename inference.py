"""
FinQuery Inference Script
Scaler OpenEnv Hackathon — Round 1

Mandatory env vars:
  API_BASE_URL  - LLM API endpoint
  MODEL_NAME    - Model identifier
  HF_TOKEN      - HuggingFace / API key

Stdout format:
  [START] task=<task_id> env=finquery model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
from typing import List, Optional
from openai import OpenAI

# -- Environment config -------------------------------------------------------
SPACE_URL = os.getenv("SPACE_URL", "https://ashutosh887-finquery.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
BENCHMARK = "finquery"
TASK_IDS = ["task1_easy", "task2_medium", "task3_hard"]
SUCCESS_THRESHOLD = 0.3
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst at a data terminal.
Answer financial questions by calling tools in sequence.
Always fetch data before computing. Never guess.

Respond ONLY with valid JSON matching this schema:
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
Include only fields relevant to your chosen action_type.
For submit_answer: include "answer" with your final answer value.
For compute: include "expression" with a valid arithmetic expression.
For data fetches: include "ticker" and "year" (or "years" for price_history).
"""


# -- Logging helpers -----------------------------------------------------------
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
# ------------------------------------------------------------------------------


def get_llm_action(client: OpenAI, task_description: str, history: List[dict]) -> tuple:
    """Call LLM and return parsed action dict."""
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
    """Run one full episode for a task. Emits START, STEPs, END to stdout."""
    import httpx

    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        # Reset
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

            # Get LLM action
            action_dict, action_raw = get_llm_action(client, task_description, history)

            # Add to history
            history.append({"role": "assistant", "content": action_raw})

            # Step the environment
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

                # Build context for next LLM call
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

        # Compute score
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
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    main()
