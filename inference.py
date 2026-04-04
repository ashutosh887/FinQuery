#!/usr/bin/env python3
"""FinQuery inference script — runs an LLM agent against the deployed FinQuery environment.

Required environment variables:
    API_BASE_URL   — Base URL of the deployed FinQuery HF Space (e.g. https://ashutosh887-finquery.hf.space)
    MODEL_NAME     — LLM model name (e.g. gpt-4o-mini)
    HF_TOKEN       — HuggingFace token for Space auth and/or OpenAI-compatible inference

Optional:
    OPENAI_API_KEY   — Explicit OpenAI API key (falls back to HF_TOKEN)
    OPENAI_BASE_URL  — Override base URL for the LLM endpoint (for HF Inference Endpoints)
"""

import json
import os
import re
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or HF_TOKEN
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

MAX_RETRIES_HEALTH = 12  # up to ~2 min waiting for cold-start
RETRY_DELAY = 10  # seconds between health retries

TASKS = [
    ("task1_easy", "easy"),
    ("task2_medium", "medium"),
    ("task3_hard", "hard"),
]

# ---------------------------------------------------------------------------
# System prompt — identical structure to _baseline_runner.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial analyst using a data terminal. You must answer "
    "financial questions by fetching data and computing results.\n\n"
    "Available actions (respond with ONLY a JSON object, no other text):\n"
    '- {"action_type": "get_income_statement", "ticker": "AAPL", "year": 2022}\n'
    '- {"action_type": "get_balance_sheet", "ticker": "AAPL", "year": 2022}\n'
    '- {"action_type": "get_cash_flow", "ticker": "AAPL", "year": 2022}\n'
    '- {"action_type": "get_price_history", "ticker": "AAPL", "years": [2020, 2021, 2022]}\n'
    '- {"action_type": "get_ratios", "ticker": "AAPL", "year": 2022}\n'
    '- {"action_type": "compare_to_sector", "ticker": "AAPL", "metric": "pe_ratio", "year": 2022}\n'
    '- {"action_type": "compute", "expression": "99803 / 394328 * 100"}\n'
    '- {"action_type": "submit_answer", "answer": YOUR_ANSWER}\n\n'
    "Rules:\n"
    "1. Fetch ALL data you need BEFORE computing or submitting.\n"
    "2. Use the compute tool for arithmetic — never guess numbers.\n"
    "3. Submit your final answer only when you have verified it.\n"
    "4. Respond with ONLY a valid JSON action object, nothing else."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_headers() -> dict:
    """Build auth headers for HF Space requests."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _env_request(method: str, endpoint: str, json_body=None, params=None) -> dict:
    """Make an authenticated request to the FinQuery environment."""
    url = f"{API_BASE_URL}{endpoint}"
    resp = requests.request(
        method,
        url,
        json=json_body,
        params=params,
        headers=_env_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM response text."""
    text = text.strip()
    # Try direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try fenced code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first JSON object in text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _wait_for_server() -> None:
    """Wait for the environment server to be healthy (handles HF cold starts)."""
    for attempt in range(1, MAX_RETRIES_HEALTH + 1):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/health",
                headers=_env_headers(),
                timeout=15,
            )
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        print(
            f"[WAIT] Environment not ready (attempt {attempt}/{MAX_RETRIES_HEALTH}), "
            f"retrying in {RETRY_DELAY}s...",
            file=sys.stderr,
        )
        time.sleep(RETRY_DELAY)
    print(f"Error: Environment at {API_BASE_URL} not reachable after retries", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_task(client: OpenAI, task_id: str, difficulty: str) -> float:
    """Run a single task episode and return the total reward."""
    # Reset episode
    result = _env_request("POST", "/reset", {"task_id": task_id})
    episode_id = result["episode_id"]
    obs = result["observation"]

    print(f"[START] episode_id={episode_id} task_id={task_id} difficulty={difficulty}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {obs['task_description']}"},
    ]

    total_reward = 0.0
    step_num = 0
    max_turns = 50  # safety ceiling across all tasks

    for _ in range(max_turns):
        if obs.get("episode_status") != "ongoing":
            break

        # Ask LLM for next action
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        assistant_msg = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse action JSON
        try:
            action = _extract_json(assistant_msg)
        except ValueError:
            messages.append({
                "role": "user",
                "content": "Invalid response. Respond with ONLY a valid JSON action object.",
            })
            continue

        # Execute step
        step_result = _env_request("POST", "/step", {
            "episode_id": episode_id,
            "action": action,
        })
        obs = step_result["observation"]
        reward = step_result["reward"]
        total_reward += reward
        step_num = obs["steps_taken"]

        action_type = action.get("action_type", "unknown")
        print(
            f"[STEP] step={step_num} action={action_type} "
            f"reward={reward} score_so_far={round(total_reward, 4)}"
        )

        if step_result["done"]:
            break

        # Feed observation back to the LLM
        obs_parts = []
        if obs.get("tool_result"):
            obs_parts.append(f"Result: {json.dumps(obs['tool_result'])}")
        if obs.get("tool_error"):
            obs_parts.append(f"Error: {obs['tool_error']}")
        if obs.get("feedback"):
            obs_parts.append(f"Feedback: {obs['feedback']}")
        obs_parts.append(
            f"Steps: {obs['steps_taken']}/{obs['steps_taken'] + obs['steps_remaining']}"
        )
        messages.append({"role": "user", "content": "\n".join(obs_parts)})

    print(
        f"[END] episode_id={episode_id} final_score={round(total_reward, 4)} "
        f"steps_taken={step_num}"
    )
    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not OPENAI_API_KEY:
        print(
            "Error: Set OPENAI_API_KEY or HF_TOKEN environment variable",
            file=sys.stderr,
        )
        sys.exit(1)

    # Wait for HF Space to wake up
    _wait_for_server()

    # Build OpenAI client
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    all_scores = {}
    for task_id, difficulty in TASKS:
        try:
            score = run_task(client, task_id, difficulty)
            all_scores[task_id] = round(score, 4)
        except Exception as e:
            print(f"Error on {task_id}: {e}", file=sys.stderr)
            sys.exit(1)

    # Summary
    print("\n--- Summary ---")
    for task_id, score in all_scores.items():
        print(f"  {task_id}: {score}")
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0
    print(f"  average: {round(avg, 4)}")


if __name__ == "__main__":
    main()
