"""Internal baseline runner used by the /baseline endpoint."""

from __future__ import annotations

import json
import re

from openai import OpenAI

SYSTEM_PROMPT = """You are a financial analyst using a data terminal. You must answer financial questions by fetching data and computing results.

Available actions (respond with JSON):
- {"action_type": "get_income_statement", "ticker": "AAPL", "year": 2022}
- {"action_type": "get_balance_sheet", "ticker": "AAPL", "year": 2022}
- {"action_type": "get_cash_flow", "ticker": "AAPL", "year": 2022}
- {"action_type": "get_price_history", "ticker": "AAPL", "years": [2020, 2021, 2022]}
- {"action_type": "get_ratios", "ticker": "AAPL", "year": 2022}
- {"action_type": "compare_to_sector", "ticker": "AAPL", "metric": "pe_ratio", "year": 2022}
- {"action_type": "compute", "expression": "99803 / 394328 * 100"}
- {"action_type": "submit_answer", "answer": YOUR_ANSWER}

Rules:
1. Fetch all data you need BEFORE computing or submitting.
2. Use the compute tool for arithmetic.
3. Submit your final answer when ready.
4. Respond with ONLY a valid JSON action object, no other text."""


def _extract_json(text: str) -> dict:
    """Extract JSON object from model response text."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def run_single_task(env, task_id: str, api_key: str) -> float:
    """Run a single task and return the episode total reward."""
    client = OpenAI(api_key=api_key)
    result = env.reset(task_id=task_id, agent_name="gpt-4o-mini-baseline")
    episode_id = result["episode_id"]
    obs = result["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {obs['task_description']}"},
    ]

    total_reward = 0.0
    max_turns = 35

    for _ in range(max_turns):
        if obs["episode_status"] != "ongoing":
            break

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        assistant_msg = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_msg})

        try:
            action = _extract_json(assistant_msg)
        except ValueError:
            messages.append({
                "role": "user",
                "content": "Invalid response. Please respond with a valid JSON action object.",
            })
            continue

        step_result = env.step(episode_id=episode_id, action=action)
        obs = step_result["observation"]
        total_reward += step_result["reward"]

        if step_result["done"]:
            break

        obs_summary = []
        if obs.get("tool_result"):
            obs_summary.append(f"Result: {json.dumps(obs['tool_result'])}")
        if obs.get("tool_error"):
            obs_summary.append(f"Error: {obs['tool_error']}")
        if obs.get("feedback"):
            obs_summary.append(f"Feedback: {obs['feedback']}")
        obs_summary.append(f"Steps: {obs['steps_taken']}/{obs['steps_taken'] + obs['steps_remaining']}")

        messages.append({"role": "user", "content": "\n".join(obs_summary)})

    return round(total_reward, 4)


def run_all_tasks(env, api_key: str) -> dict[str, float]:
    """Run all 3 tasks and return scores dict."""
    scores = {}
    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        scores[task_id] = run_single_task(env, task_id, api_key)
    return scores
