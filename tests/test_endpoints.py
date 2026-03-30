"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_tasks(client):
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 3
    ids = {t["id"] for t in tasks}
    assert ids == {"task1_easy", "task2_medium", "task3_hard"}


def test_reset(client):
    resp = client.post("/reset", json={"task_id": "task1_easy"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reward"] == 0.0
    assert data["done"] is False
    obs = data["observation"]
    assert "Apple" in obs["task_description"]
    assert obs["steps_taken"] == 0
    assert obs["episode_status"] == "ongoing"


def test_reset_random(client):
    resp = client.post("/reset", json={})
    assert resp.status_code == 200


def test_state(client):
    client.post("/reset", json={"task_id": "task1_easy"})
    resp = client.get("/state")
    assert resp.status_code == 200
    state = resp.json()
    assert state["task_id"] == "task1_easy"
    assert state["step_count"] == 0
    assert state["answer_submitted"] is False


def test_step_fetch(client):
    client.post("/reset", json={"task_id": "task1_easy"})
    resp = client.post("/step", json={
        "action": {
            "action_type": "get_income_statement",
            "ticker": "AAPL",
            "year": 2022,
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is False
    assert data["reward"] == 0.05  # relevant fetch
    assert data["observation"]["tool_result"]["revenue"] == 394328


def test_step_invalid_ticker(client):
    client.post("/reset", json={"task_id": "task1_easy"})
    resp = client.post("/step", json={
        "action": {
            "action_type": "get_income_statement",
            "ticker": "FAKE",
            "year": 2022,
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["observation"]["tool_error"] is not None
    assert data["reward"] == 0.0


def test_step_compute(client):
    client.post("/reset", json={"task_id": "task1_easy"})
    resp = client.post("/step", json={
        "action": {
            "action_type": "compute",
            "expression": "99803 / 394328 * 100",
        }
    })
    assert resp.status_code == 200
    result = resp.json()["observation"]["tool_result"]
    assert abs(result["result"] - 25.31) < 0.01


def test_full_episode_task1(client):
    """Run a complete Task 1 episode: fetch -> compute -> submit."""
    client.post("/reset", json={"task_id": "task1_easy"})

    # Step 1: Fetch income statement
    resp = client.post("/step", json={
        "action": {"action_type": "get_income_statement", "ticker": "AAPL", "year": 2022}
    })
    assert resp.json()["done"] is False

    # Step 2: Compute margin
    resp = client.post("/step", json={
        "action": {"action_type": "compute", "expression": "99803 / 394328 * 100"}
    })
    assert resp.json()["done"] is False

    # Step 3: Submit answer
    resp = client.post("/step", json={
        "action": {"action_type": "submit_answer", "answer": 25.31}
    })
    data = resp.json()
    assert data["done"] is True
    assert data["observation"]["episode_status"] == "answered"
    assert data["observation"]["tool_result"]["grader_score"] == 1.0


def test_grader_endpoint(client):
    resp = client.post("/grader", json={
        "task_id": "task1_easy",
        "final_answer": 25.31,
    })
    assert resp.status_code == 200
    assert resp.json()["score"] == 1.0


def test_grader_unknown_task(client):
    resp = client.post("/grader", json={
        "task_id": "unknown_task",
        "final_answer": 0,
    })
    assert resp.status_code == 400
