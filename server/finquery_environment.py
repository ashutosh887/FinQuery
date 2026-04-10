"""Core FinQuery environment — reset(), step(), state().

Manages episode state, routes actions to tools, computes rewards.
Supports concurrent episodes via episode_id-based session management.
"""

from __future__ import annotations

import ast
import json
import operator
import uuid
from pathlib import Path
from typing import Any

from server.tools import income_statement, balance_sheet, cash_flow, price_history, ratios, sector_compare
from server.graders import task1_grader, task2_grader, task3_grader
from server.rewards.reward_engine import (
    compute_episode_total,
    compute_step_reward,
    compute_terminal_reward,
)
from server.database import save_episode, finish_episode, record_leaderboard
from server.tasks.task_generator import (
    EasyTaskGenerator,
    HardTaskGenerator,
    MediumTaskGenerator,
)

DATA_DIR = Path(__file__).parent / "data"

# Static metadata for /tasks endpoint and task_id validation
TASK_META = {
    "task1_easy": {
        "name": "Single-Metric Computation",
        "difficulty": "easy",
        "description": (
            "Compute a financial metric for a randomly selected company and year. "
            "Metrics include margins, ratios, and returns. Each reset generates a unique question."
        ),
        "max_steps": 10,
    },
    "task2_medium": {
        "name": "Multi-Company Ratio Comparison",
        "difficulty": "medium",
        "description": (
            "Compare a financial ratio across multiple companies against their sector median. "
            "Identify which company is most favorably positioned. Each reset generates a unique question."
        ),
        "max_steps": 20,
    },
    "task3_hard": {
        "name": "Multi-Year Anomaly Detection",
        "difficulty": "hard",
        "description": (
            "Detect financial anomaly patterns across multiple companies over a multi-year window. "
            "Identify qualifying companies and supporting evidence. Each reset generates a unique question."
        ),
        "max_steps": 40,
    },
}

# Grader dispatch by grader_id stored in each episode
GRADERS = {
    "task1": task1_grader,
    "task2": task2_grader,
    "task3": task3_grader,
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
    """Core environment managing multiple concurrent episodes."""

    def __init__(self):
        with open(DATA_DIR / "financials.json") as f:
            self.data: dict = json.load(f)
        with open(DATA_DIR / "sectors.json") as f:
            self.sectors: dict = json.load(f)
        self._episodes: dict[str, dict] = {}

        # Initialize procedural task generators
        self._generators = {
            "task1_easy": EasyTaskGenerator(self.data, self.sectors),
            "task2_medium": MediumTaskGenerator(self.data, self.sectors),
            "task3_hard": HardTaskGenerator(self.data, self.sectors),
        }

    def reset(self, task_id: str | None = None, agent_name: str = "anonymous") -> dict:
        import random

        if task_id is None:
            task_id = random.choice(list(TASK_META.keys()))
        if task_id not in TASK_META:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASK_META.keys())}")

        # Generate a unique task instance
        task_instance = self._generators[task_id].generate()

        episode_id = str(uuid.uuid4())
        ep = {
            "episode_id": episode_id,
            "task_id": task_id,
            "task_difficulty": task_instance.difficulty,
            "task_description": task_instance.task_description,
            "agent_name": agent_name,
            "step_count": 0,
            "max_steps": task_instance.max_steps,
            "fetched_data": {},
            "fetch_log": [],
            "tickers_queried": [],
            "answer_submitted": False,
            "final_answer": None,
            "cumulative_reward": 0.0,
            "done": False,
            # Task instance data for grading and rewards
            "ground_truth": task_instance.ground_truth,
            "required_fetches": task_instance.required_fetches,
            "min_fetches": task_instance.min_fetches,
            "expected_intermediates": task_instance.expected_intermediates,
            "grader_id": task_instance.grader_id,
        }
        self._episodes[episode_id] = ep
        save_episode(episode_id, task_id, task_instance.difficulty, agent_name)

        return {
            "episode_id": episode_id,
            "observation": {
                "task_description": task_instance.task_description,
                "tool_result": None,
                "tool_error": None,
                "steps_taken": 0,
                "steps_remaining": task_instance.max_steps,
                "tickers_queried": [],
                "episode_status": "ongoing",
                "feedback": None,
            },
            "reward": 0.0,
            "done": False,
        }

    def step(self, episode_id: str, action: dict) -> dict:
        ep = self._episodes.get(episode_id)
        if ep is None:
            return self._error_response("Unknown episode_id. Call /reset first.")
        if ep["done"]:
            return self._error_response("Episode already finished.")

        ep["step_count"] += 1
        action_type = action.get("action_type", "")

        tool_result = None
        tool_error = None
        fetch_key = None
        computed_result = None

        try:
            if action_type == "get_income_statement":
                ticker, year = self._validate_ticker_year(action)
                tool_result = income_statement.get(ticker, year, self.data)
                fetch_key = f"income_statement:{ticker}:{year}"
                self._record_fetch(ep, ticker, fetch_key, tool_result)

            elif action_type == "get_balance_sheet":
                ticker, year = self._validate_ticker_year(action)
                tool_result = balance_sheet.get(ticker, year, self.data)
                fetch_key = f"balance_sheet:{ticker}:{year}"
                self._record_fetch(ep, ticker, fetch_key, tool_result)

            elif action_type == "get_cash_flow":
                ticker, year = self._validate_ticker_year(action)
                tool_result = cash_flow.get(ticker, year, self.data)
                fetch_key = f"cash_flow:{ticker}:{year}"
                self._record_fetch(ep, ticker, fetch_key, tool_result)

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
                self._record_fetch(ep, ticker, fetch_key, tool_result)

            elif action_type == "compare_to_sector":
                ticker, year = self._validate_ticker_year(action)
                metric = self._require_field(action, "metric")
                tool_result = sector_compare.get(ticker, metric, year, self.data, self.sectors)
                fetch_key = f"sector_compare:{ticker}:{metric}:{year}"
                self._record_fetch(ep, ticker, fetch_key, tool_result)

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

                grader = GRADERS[ep["grader_id"]]
                grader_result = grader.grade(
                    answer,
                    ground_truth=ep["ground_truth"],
                    step_count=ep["step_count"],
                    max_steps=ep["max_steps"],
                )
                grader_score = max(0.01, min(0.99, grader_result["score"]))

                terminal_reward = compute_terminal_reward(
                    grader_score=grader_score,
                    step_count=ep["step_count"],
                    max_steps=ep["max_steps"],
                    fetch_count=len(ep["fetch_log"]),
                    min_fetches=ep["min_fetches"],
                )

                step_reward, feedback = compute_step_reward(
                    action_type="submit_answer",
                    fetch_key=None,
                    fetch_log=ep["fetch_log"],
                    required_fetches=ep["required_fetches"],
                    min_fetches=ep["min_fetches"],
                )

                total_step = step_reward + terminal_reward
                ep["cumulative_reward"] = compute_episode_total(
                    ep["cumulative_reward"] + step_reward, terminal_reward
                )

                clamped_total = max(0.01, min(0.99, ep["cumulative_reward"]))
                tool_result = {
                    "grader_score": grader_score,
                    "breakdown": grader_result["breakdown"],
                    "terminal_reward": round(terminal_reward, 4),
                    "episode_total": round(clamped_total, 4),
                }

                finish_episode(
                    episode_id, ep["step_count"], clamped_total,
                    answer, "answered",
                )
                record_leaderboard(
                    ep["agent_name"], ep["task_id"],
                    clamped_total, ep["step_count"],
                )

                response = self._build_response(
                    ep,
                    tool_result=tool_result,
                    tool_error=None,
                    reward=round(total_step, 4),
                    feedback=feedback,
                    status="answered",
                )
                del self._episodes[episode_id]
                return response
            else:
                raise ValueError(f"Unknown action_type: {action_type}")

        except ValueError as e:
            tool_error = str(e)
            ep["step_count"] -= 1
            return self._build_response(
                ep,
                tool_result=None,
                tool_error=tool_error,
                reward=0.0,
                feedback=None,
                status="ongoing",
            )

        step_reward, feedback = compute_step_reward(
            action_type=action_type,
            fetch_key=fetch_key,
            fetch_log=ep["fetch_log"][:-1] if fetch_key and fetch_key in ep["fetch_log"] else ep["fetch_log"],
            required_fetches=ep["required_fetches"],
            min_fetches=ep["min_fetches"],
            computed_result=computed_result,
            expected_intermediates=ep.get("expected_intermediates"),
        )

        ep["cumulative_reward"] += step_reward

        status = "ongoing"
        done = False
        if ep["step_count"] >= ep["max_steps"]:
            status = "failed_max_steps"
            done = True
            ep["done"] = True
            clamped_reward = max(0.01, min(0.99, ep["cumulative_reward"]))
            finish_episode(
                episode_id, ep["step_count"], clamped_reward,
                None, "failed_max_steps",
            )
            del self._episodes[episode_id]

        return self._build_response(
            ep,
            tool_result=tool_result,
            tool_error=tool_error,
            reward=round(step_reward, 4),
            feedback=feedback,
            status=status,
            done_override=done,
        )

    def state(self, episode_id: str | None = None) -> dict:
        if episode_id and episode_id in self._episodes:
            ep = self._episodes[episode_id]
        elif not episode_id and self._episodes:
            ep = list(self._episodes.values())[-1]
        else:
            return {
                "episode_id": "",
                "task_id": "",
                "task_difficulty": "easy",
                "step_count": 0,
                "fetched_data": {},
                "answer_submitted": False,
                "score_so_far": 0.0,
            }
        return {
            "episode_id": ep["episode_id"],
            "task_id": ep["task_id"],
            "task_difficulty": ep["task_difficulty"],
            "step_count": ep["step_count"],
            "fetched_data": ep["fetched_data"],
            "answer_submitted": ep["answer_submitted"],
            "score_so_far": round(ep["cumulative_reward"], 4),
        }

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

    def _record_fetch(self, ep: dict, ticker: str, fetch_key: str, result: dict):
        if ticker not in ep["tickers_queried"]:
            ep["tickers_queried"].append(ticker)
        ep["fetched_data"][fetch_key] = result
        if fetch_key not in ep["fetch_log"]:
            ep["fetch_log"].append(fetch_key)

    def _build_response(
        self,
        ep: dict,
        tool_result,
        tool_error,
        reward,
        feedback,
        status,
        done_override=None,
    ) -> dict:
        done = done_override if done_override is not None else ep["done"]
        return {
            "episode_id": ep["episode_id"],
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
