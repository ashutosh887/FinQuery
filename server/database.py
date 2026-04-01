"""SQLite database for episode history and leaderboard."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "data" / "finquery.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id   TEXT PRIMARY KEY,
            task_id      TEXT NOT NULL,
            difficulty   TEXT NOT NULL,
            agent_name   TEXT DEFAULT 'anonymous',
            step_count   INTEGER DEFAULT 0,
            score        REAL DEFAULT 0.0,
            final_answer TEXT,
            status       TEXT DEFAULT 'ongoing',
            started_at   REAL NOT NULL,
            finished_at  REAL
        );

        CREATE TABLE IF NOT EXISTS leaderboard (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name  TEXT NOT NULL,
            task_id     TEXT NOT NULL,
            score       REAL NOT NULL,
            steps       INTEGER NOT NULL,
            recorded_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
        CREATE INDEX IF NOT EXISTS idx_leaderboard_task ON leaderboard(task_id);
        CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(score DESC);
    """)
    conn.close()


def save_episode(
    episode_id: str,
    task_id: str,
    difficulty: str,
    agent_name: str = "anonymous",
):
    """Insert a new episode record on reset."""
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO episodes (episode_id, task_id, difficulty, agent_name, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (episode_id, task_id, difficulty, agent_name, time.time()),
    )
    conn.commit()
    conn.close()


def finish_episode(
    episode_id: str,
    step_count: int,
    score: float,
    final_answer: Any,
    status: str,
):
    """Update episode record when it finishes."""
    conn = _conn()
    conn.execute(
        "UPDATE episodes SET step_count=?, score=?, final_answer=?, status=?, finished_at=? "
        "WHERE episode_id=?",
        (step_count, score, json.dumps(final_answer), status, time.time(), episode_id),
    )
    conn.commit()
    conn.close()


def record_leaderboard(agent_name: str, task_id: str, score: float, steps: int):
    """Add a leaderboard entry after a completed episode."""
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard (agent_name, task_id, score, steps, recorded_at) VALUES (?, ?, ?, ?, ?)",
        (agent_name, task_id, score, steps, time.time()),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50, task_id: str | None = None) -> list[dict]:
    """Return recent episode history."""
    conn = _conn()
    if task_id:
        rows = conn.execute(
            "SELECT * FROM episodes WHERE task_id=? ORDER BY started_at DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leaderboard(task_id: str | None = None, limit: int = 20) -> list[dict]:
    """Return top scores, optionally filtered by task."""
    conn = _conn()
    if task_id:
        rows = conn.execute(
            "SELECT agent_name, task_id, MAX(score) as best_score, MIN(steps) as best_steps, "
            "COUNT(*) as attempts "
            "FROM leaderboard WHERE task_id=? "
            "GROUP BY agent_name, task_id ORDER BY best_score DESC, best_steps ASC LIMIT ?",
            (task_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT agent_name, task_id, MAX(score) as best_score, MIN(steps) as best_steps, "
            "COUNT(*) as attempts "
            "FROM leaderboard GROUP BY agent_name, task_id "
            "ORDER BY best_score DESC, best_steps ASC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
