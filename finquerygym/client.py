"""FinQueryGym client — talks to the server over HTTP.

This file must NOT import anything from server/.
"""

from typing import Any, Dict, List, Optional

import requests

from .models import (
    BaselineResponse,
    FinQueryAction,
    FinQueryObservation,
    FinQueryState,
    GraderRequest,
    GraderResponse,
    StepResponse,
    TaskInfo,
)


class FinQueryEnv:
    """HTTP client for FinQueryGym server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # -- context manager -------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._session.close()

    def sync(self):
        return self

    # -- core API --------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> StepResponse:
        payload = {}
        if task_id is not None:
            payload["task_id"] = task_id
        resp = self._session.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return StepResponse(**resp.json())

    def step(self, action: FinQueryAction) -> StepResponse:
        payload = {"action": action.model_dump(exclude_none=True)}
        resp = self._session.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return StepResponse(**resp.json())

    def state(self) -> FinQueryState:
        resp = self._session.get(f"{self.base_url}/state")
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
