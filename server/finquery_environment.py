"""Core FinQueryGym environment — reset(), step(), state().

Manages episode state, routes actions to tools, computes rewards.
"""

from __future__ import annotations

import ast
import json
import operator
import uuid
from pathlib import Path
from typing import Any, Optional

from server.tools import income_statement, balance_sheet, cash_flow, price_history, ratios, sector_compare
from server.graders import task1_grader, task2_grader, task3_grader
from server.rewards.reward_engine import (
    compute_episode_total,
    compute_step_reward,
    compute_terminal_reward,
)

DATA_DIR = Path(__file__).parent / "data"
MAX_STEPS = 40

TASKS = {
    "task1_easy": {
        "name": "Single-Metric Computation",
        "difficulty": "easy",
        "description": (
            "What was Apple's net profit margin for fiscal year 2022? "
            "Express as a percentage rounded to 2 decimal places."
        ),
        "max_steps": MAX_STEPS,
        "required_fetches": task1_grader.GROUND_TRUTH["required_fetches"],
        "min_fetches": task1_grader.GROUND_TRUTH["min_fetches"],
        "expected_intermediates": [25.31],
        "grader": task1_grader,
    },
    "task2_medium": {
        "name": "Multi-Company Comparison",
        "difficulty": "medium",
        "description": (
            "Among Microsoft, Google, and Meta, which company had the most favorable "
            "EV/EBITDA relative to the tech sector median in 2023? By how many points "
            "did it differ from the sector median? Submit your answer as "
            '{\"company\": \"TICKER\", \"delta\": NUMBER}.'
        ),
        "max_steps": MAX_STEPS,
        "required_fetches": task2_grader.GROUND_TRUTH["required_fetches"],
        "min_fetches": task2_grader.GROUND_TRUTH["min_fetches"],
        "expected_intermediates": [12.83, 17.34, 26.29, -5.47, -0.96, 7.99],
        "grader": task2_grader,
    },
    "task3_hard": {
        "name": "Multi-Year Anomaly Detection",
        "difficulty": "hard",
        "description": (
            "Among Tesla, Ford, and GM — which company had negative free cash flow in "
            "at least 2 of the 4 fiscal years from 2020–2023, AND had a P/E ratio above "
            "30 in any of those same years? For each qualifying company, state which "
            "specific years had negative FCF and which years had P/E > 30. Submit as "
            '{\"qualifying_companies\": [\"TICKER\"], \"details\": {\"TICKER\": '
            '{\"negative_fcf_years\": [YEAR, ...], \"pe_above_30_years\": [YEAR, ...]}}}.'
        ),
        "max_steps": MAX_STEPS,
        "required_fetches": task3_grader.GROUND_TRUTH["required_fetches"],
        "min_fetches": task3_grader.GROUND_TRUTH["min_fetches"],
        "expected_intermediates": [],
        "grader": task3_grader,
    },
}

# Safe expression evaluator — restricted to arithmetic only
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression safely."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e
    return _eval_node(tree.body)


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


class FinQueryEnvironment:
    """Core environment managing episode lifecycle."""

    def __init__(self):
        with open(DATA_DIR / "financials.json") as f:
            self.data: dict = json.load(f)
        with open(DATA_DIR / "sectors.json") as f:
            self.sectors: dict = json.load(f)
        self._episode: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> dict:
        import random

        if task_id is None:
            task_id = random.choice(list(TASKS.keys()))
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")

        task = TASKS[task_id]
        self._episode = {
            "episode_id": str(uuid.uuid4()),
            "task_id": task_id,
            "task_difficulty": task["difficulty"],
            "task_description": task["description"],
            "step_count": 0,
            "max_steps": task["max_steps"],
            "fetched_data": {},
            "fetch_log": [],
            "tickers_queried": [],
            "answer_submitted": False,
            "final_answer": None,
            "cumulative_reward": 0.0,
            "done": False,
        }

        return {
            "observation": {
                "task_description": task["description"],
                "tool_result": None,
                "tool_error": None,
                "steps_taken": 0,
                "steps_remaining": task["max_steps"],
                "tickers_queried": [],
                "episode_status": "ongoing",
                "feedback": None,
            },
            "reward": 0.0,
            "done": False,
        }

    def step(self, action: dict) -> dict:
        if self._episode is None:
            return self._error_response("No active episode. Call /reset first.")
        if self._episode["done"]:
            return self._error_response("Episode already finished.")

        ep = self._episode
        ep["step_count"] += 1
        action_type = action.get("action_type", "")
        task = TASKS[ep["task_id"]]

        tool_result = None
        tool_error = None
        fetch_key = None
        computed_result = None

        try:
            if action_type == "get_income_statement":
                ticker, year = self._validate_ticker_year(action)
                tool_result = income_statement.get(ticker, year, self.data)
                fetch_key = f"income_statement:{ticker}:{year}"
                self._record_fetch(ticker, fetch_key, tool_result)

            elif action_type == "get_balance_sheet":
                ticker, year = self._validate_ticker_year(action)
                tool_result = balance_sheet.get(ticker, year, self.data)
                fetch_key = f"balance_sheet:{ticker}:{year}"
                self._record_fetch(ticker, fetch_key, tool_result)

            elif action_type == "get_cash_flow":
                ticker, year = self._validate_ticker_year(action)
                tool_result = cash_flow.get(ticker, year, self.data)
                fetch_key = f"cash_flow:{ticker}:{year}"
                self._record_fetch(ticker, fetch_key, tool_result)

            elif action_type == "get_price_history":
                ticker = self._require_field(action, "ticker").upper()
                years = action.get("years") or []
                if not years:
                    raise ValueError("Missing required field: years")
                tool_result = price_history.get(ticker, years, self.data)
                for y in years:
                    fk = f"price_history:{ticker}:{y}"
                    if fk not in ep["fetch_log"]:
                        ep["fetch_log"].append(fk)
                if ticker not in ep["tickers_queried"]:
                    ep["tickers_queried"].append(ticker)
                ep["fetched_data"][f"price_history:{ticker}"] = tool_result
                fetch_key = f"price_history:{ticker}:{years[0]}"

            elif action_type == "get_ratios":
                ticker, year = self._validate_ticker_year(action)
                tool_result = ratios.get(ticker, year, self.data)
                fetch_key = f"ratios:{ticker}:{year}"
                self._record_fetch(ticker, fetch_key, tool_result)

            elif action_type == "compare_to_sector":
                ticker, year = self._validate_ticker_year(action)
                metric = self._require_field(action, "metric")
                tool_result = sector_compare.get(ticker, metric, year, self.data, self.sectors)
                fetch_key = f"sector_compare:{ticker}:{metric}:{year}"
                self._record_fetch(ticker, fetch_key, tool_result)

            elif action_type == "compute":
                expression = self._require_field(action, "expression")
                result = _safe_eval(expression)
                computed_result = result
                tool_result = {"expression": expression, "result": round(result, 6)}

            elif action_type == "submit_answer":
                answer = action.get("answer")
                if answer is None:
                    raise ValueError("Missing required field: answer")
                ep["answer_submitted"] = True
                ep["final_answer"] = answer
                ep["done"] = True

                grader_result = task["grader"].grade(answer)
                grader_score = grader_result["score"]

                terminal_reward = compute_terminal_reward(
                    grader_score=grader_score,
                    step_count=ep["step_count"],
                    max_steps=ep["max_steps"],
                    fetch_count=len(ep["fetch_log"]),
                    min_fetches=task["min_fetches"],
                )

                step_reward, feedback = compute_step_reward(
                    action_type="submit_answer",
                    fetch_key=None,
                    fetch_log=ep["fetch_log"],
                    required_fetches=task["required_fetches"],
                    min_fetches=task["min_fetches"],
                )

                total_step = step_reward + terminal_reward
                ep["cumulative_reward"] = compute_episode_total(
                    ep["cumulative_reward"] + step_reward, terminal_reward
                )

                tool_result = {
                    "grader_score": grader_score,
                    "breakdown": grader_result["breakdown"],
                    "terminal_reward": round(terminal_reward, 4),
                    "episode_total": round(ep["cumulative_reward"], 4),
                }

                return self._build_response(
                    tool_result=tool_result,
                    tool_error=None,
                    reward=round(total_step, 4),
                    feedback=feedback,
                    status="answered",
                )
            else:
                raise ValueError(f"Unknown action_type: {action_type}")

        except ValueError as e:
            tool_error = str(e)
            ep["step_count"] -= 1  # Don't count invalid actions
            return self._build_response(
                tool_result=None,
                tool_error=tool_error,
                reward=0.0,
                feedback=None,
                status="ongoing",
            )

        # Compute step reward for non-submit actions
        step_reward, feedback = compute_step_reward(
            action_type=action_type,
            fetch_key=fetch_key,
            fetch_log=ep["fetch_log"][:-1] if fetch_key and fetch_key in ep["fetch_log"] else ep["fetch_log"],
            required_fetches=task["required_fetches"],
            min_fetches=task["min_fetches"],
            computed_result=computed_result,
            expected_intermediates=task.get("expected_intermediates"),
        )

        ep["cumulative_reward"] += step_reward

        # Check max steps
        status = "ongoing"
        done = False
        if ep["step_count"] >= ep["max_steps"]:
            status = "failed_max_steps"
            done = True
            ep["done"] = True

        return self._build_response(
            tool_result=tool_result,
            tool_error=tool_error,
            reward=round(step_reward, 4),
            feedback=feedback,
            status=status,
            done_override=done,
        )

    def state(self) -> dict:
        if self._episode is None:
            return {
                "episode_id": "",
                "task_id": "",
                "task_difficulty": "easy",
                "step_count": 0,
                "fetched_data": {},
                "answer_submitted": False,
                "score_so_far": 0.0,
            }
        ep = self._episode
        return {
            "episode_id": ep["episode_id"],
            "task_id": ep["task_id"],
            "task_difficulty": ep["task_difficulty"],
            "step_count": ep["step_count"],
            "fetched_data": ep["fetched_data"],
            "answer_submitted": ep["answer_submitted"],
            "score_so_far": round(ep["cumulative_reward"], 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_ticker_year(self, action: dict) -> tuple[str, int]:
        ticker = self._require_field(action, "ticker").upper()
        year = self._require_field(action, "year")
        try:
            year = int(year)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid year: {year}")
        return ticker, year

    def _require_field(self, action: dict, field: str) -> Any:
        val = action.get(field)
        if val is None:
            raise ValueError(f"Missing required field: {field}")
        return val

    def _record_fetch(self, ticker: str, fetch_key: str, result: dict):
        ep = self._episode
        if ticker not in ep["tickers_queried"]:
            ep["tickers_queried"].append(ticker)
        ep["fetched_data"][fetch_key] = result
        if fetch_key not in ep["fetch_log"]:
            ep["fetch_log"].append(fetch_key)

    def _build_response(
        self,
        tool_result,
        tool_error,
        reward,
        feedback,
        status,
        done_override=None,
    ) -> dict:
        ep = self._episode
        done = done_override if done_override is not None else ep["done"]
        return {
            "observation": {
                "task_description": ep["task_description"],
                "tool_result": tool_result,
                "tool_error": tool_error,
                "steps_taken": ep["step_count"],
                "steps_remaining": ep["max_steps"] - ep["step_count"],
                "tickers_queried": list(ep["tickers_queried"]),
                "episode_status": status,
                "feedback": feedback,
            },
            "reward": reward,
            "done": done,
        }

    def _error_response(self, msg: str) -> dict:
        return {
            "observation": {
                "task_description": "",
                "tool_result": None,
                "tool_error": msg,
                "steps_taken": 0,
                "steps_remaining": 0,
                "tickers_queried": [],
                "episode_status": "ongoing",
                "feedback": None,
            },
            "reward": 0.0,
            "done": False,
        }
