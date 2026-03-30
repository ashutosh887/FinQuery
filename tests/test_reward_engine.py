"""Tests for the reward engine."""

from server.rewards.reward_engine import (
    compute_episode_total,
    compute_step_reward,
    compute_terminal_reward,
)


def test_relevant_fetch():
    reward, feedback = compute_step_reward(
        action_type="get_income_statement",
        fetch_key="income_statement:AAPL:2022",
        fetch_log=[],
        required_fetches=["income_statement:AAPL:2022"],
        min_fetches=1,
    )
    assert reward == 0.05
    assert feedback is not None


def test_irrelevant_fetch():
    reward, feedback = compute_step_reward(
        action_type="get_income_statement",
        fetch_key="income_statement:MSFT:2020",
        fetch_log=[],
        required_fetches=["income_statement:AAPL:2022"],
        min_fetches=1,
    )
    assert reward == -0.02


def test_duplicate_fetch():
    reward, feedback = compute_step_reward(
        action_type="get_income_statement",
        fetch_key="income_statement:AAPL:2022",
        fetch_log=["income_statement:AAPL:2022"],
        required_fetches=["income_statement:AAPL:2022"],
        min_fetches=1,
    )
    assert reward == -0.01


def test_blind_submit():
    reward, feedback = compute_step_reward(
        action_type="submit_answer",
        fetch_key=None,
        fetch_log=[],
        required_fetches=["income_statement:AAPL:2022"],
        min_fetches=1,
    )
    assert reward == -0.05


def test_correct_compute():
    reward, feedback = compute_step_reward(
        action_type="compute",
        fetch_key=None,
        fetch_log=[],
        required_fetches=[],
        min_fetches=0,
        computed_result=25.31,
        expected_intermediates=[25.31],
    )
    assert reward == 0.10


def test_terminal_reward_full():
    reward = compute_terminal_reward(
        grader_score=1.0,
        step_count=3,
        max_steps=40,
        fetch_count=2,
        min_fetches=1,
    )
    assert abs(reward - 0.80) < 1e-9  # 0.70 + 0.10 efficiency


def test_terminal_reward_no_efficiency():
    reward = compute_terminal_reward(
        grader_score=1.0,
        step_count=30,
        max_steps=40,
        fetch_count=2,
        min_fetches=1,
    )
    assert reward == 0.70  # no efficiency bonus


def test_episode_total_clipping():
    assert compute_episode_total(0.5, 0.8) == 1.0
    assert compute_episode_total(-0.5, 0.3) == 0.0
    assert compute_episode_total(0.1, 0.3) == 0.4
