#!/usr/bin/env python3
"""Baseline runner — executes all 3 FinQuery tasks using gpt-4o-mini.

Usage:
    export OPENAI_API_KEY=your_key
    python baseline.py
"""

import os
import sys


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    from server.database import init_db
    from server.finquery_environment import FinQueryEnvironment
    from server._baseline_runner import run_single_task

    init_db()
    env = FinQueryEnvironment()

    tasks = [
        ("task1_easy", "easy"),
        ("task2_medium", "medium"),
        ("task3_hard", "hard"),
    ]

    for task_id, difficulty in tasks:
        try:
            score = run_single_task(env, task_id, api_key)
            label = f"Task {tasks.index((task_id, difficulty)) + 1} ({difficulty})"
            print(f"{label:<22} — score: {score:.2f}")
        except Exception as e:
            print(f"Error on {task_id}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
