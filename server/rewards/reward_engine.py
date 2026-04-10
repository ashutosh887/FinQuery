"""Dense per-step reward engine. Episode total clipped to (0.01, 0.99)."""

from __future__ import annotations

from typing import List, Optional, Tuple

STEP_REWARDS = {
    "relevant_fetch": 0.05,
    "irrelevant_fetch": -0.02,
    "duplicate_fetch": -0.01,
    "correct_compute": 0.10,
    "blind_submit": -0.05,
}

TERMINAL_REWARD_MAX = 0.70
EFFICIENCY_BONUS = 0.10
MIN_FETCH_PENALTY = -0.10


def compute_step_reward(
    action_type: str,
    fetch_key: str | None,
    fetch_log: list[str],
    required_fetches: list[str],
    min_fetches: int,
    computed_result: float | None = None,
    expected_intermediates: list[float] | None = None,
) -> tuple[float, str | None]:
    if action_type == "submit_answer":
        if len(fetch_log) == 0:
            return STEP_REWARDS["blind_submit"], "Blind submit penalty"
        if len(fetch_log) < min_fetches:
            return MIN_FETCH_PENALTY, f"Only {len(fetch_log)} fetches (min {min_fetches})"
        return 0.0, None

    if action_type == "compute":
        if computed_result is not None and expected_intermediates:
            for exp in expected_intermediates:
                if abs(computed_result - exp) < 0.01:
                    return STEP_REWARDS["correct_compute"], "Correct intermediate"
        return 0.0, None

    if fetch_key is None:
        return 0.0, None
    if fetch_key in fetch_log:
        return STEP_REWARDS["duplicate_fetch"], "Duplicate fetch"
    if fetch_key in required_fetches:
        return STEP_REWARDS["relevant_fetch"], "Relevant data fetched"
    return STEP_REWARDS["irrelevant_fetch"], "Irrelevant to task"


def compute_terminal_reward(grader_score, step_count, max_steps, fetch_count, min_fetches):
    accuracy_reward = grader_score * TERMINAL_REWARD_MAX
    efficiency_bonus = EFFICIENCY_BONUS if step_count <= 0.6 * max_steps else 0.0
    min_fetch_penalty = MIN_FETCH_PENALTY if fetch_count < min_fetches else 0.0
    return accuracy_reward + efficiency_bonus + min_fetch_penalty


def compute_episode_total(cumulative_step_reward, terminal_reward):
    return max(0.01, min(0.99, cumulative_step_reward + terminal_reward))
