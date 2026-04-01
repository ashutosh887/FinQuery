"""FinQuery client — talks to the server over HTTP.

This file must NOT import anything from server/.
"""

from typing import Any, Dict, List, Optional

import requests

from .models import (
    BaselineResponse,
    EpisodeRecord,
    FinQueryAction,
    FinQueryObservation,
    FinQueryState,
    GraderRequest,
    GraderResponse,
    LeaderboardEntry,
    StepResponse,
    TaskInfo,
)


class FinQueryEnv:
    """HTTP client for FinQuery server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._episode_id: str | None = None

    # -- context manager -------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._session.close()

    def sync(self):
        return self

    # -- core API --------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, agent_name: str = "anonymous") -> StepResponse:
        payload: dict = {}
        if task_id is not None:
            payload["task_id"] = task_id
        payload["agent_name"] = agent_name
        resp = self._session.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._episode_id = data.get("episode_id")
        return StepResponse(**data)

    def step(self, action: FinQueryAction, episode_id: Optional[str] = None) -> StepResponse:
        eid = episode_id or self._episode_id
        if not eid:
            raise ValueError("No episode_id. Call reset() first.")
        payload = {"episode_id": eid, "action": action.model_dump(exclude_none=True)}
        resp = self._session.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return StepResponse(**resp.json())

    def state(self, episode_id: Optional[str] = None) -> FinQueryState:
        params = {}
        eid = episode_id or self._episode_id
        if eid:
            params["episode_id"] = eid
        resp = self._session.get(f"{self.base_url}/state", params=params)
        resp.raise_for_status()
        return FinQueryState(**resp.json())

    # -- extra endpoints -------------------------------------------------------

    def tasks(self) -> List[TaskInfo]:
        resp = self._session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return [TaskInfo(**t) for t in resp.json()]

    def grade(self, task_id: str, final_answer: Any, episode_id: Optional[str] = None) -> GraderResponse:
        req = GraderRequest(task_id=task_id, final_answer=final_answer, episode_id=episode_id)
        resp = self._session.post(f"{self.base_url}/grader", json=req.model_dump())
        resp.raise_for_status()
        return GraderResponse(**resp.json())

    def run_baseline(self) -> BaselineResponse:
        resp = self._session.post(f"{self.base_url}/baseline", json={})
        resp.raise_for_status()
        return BaselineResponse(**resp.json())

    def history(self, limit: int = 50, task_id: Optional[str] = None) -> List[dict]:
        params: dict = {"limit": limit}
        if task_id:
            params["task_id"] = task_id
        resp = self._session.get(f"{self.base_url}/history", params=params)
        resp.raise_for_status()
        return resp.json()

    def leaderboard(self, limit: int = 20, task_id: Optional[str] = None) -> List[dict]:
        params: dict = {"limit": limit}
        if task_id:
            params["task_id"] = task_id
        resp = self._session.get(f"{self.base_url}/leaderboard", params=params)
        resp.raise_for_status()
        return resp.json()
