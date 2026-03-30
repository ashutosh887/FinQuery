"""FastAPI app — all routes registered here.

Endpoints: /reset, /step, /state, /tasks, /grader, /baseline, /health
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from finquerygym.models import (
    BaselineResponse,
    FinQueryAction,
    GraderRequest,
    GraderResponse,
    ResetRequest,
    StepRequest,
    StepResponse,
    TaskInfo,
)
from server.finquery_environment import TASKS, FinQueryEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.env = FinQueryEnvironment()
    yield


app = FastAPI(
    title="FinQueryGym",
    description="OpenEnv-compatible RL environment for financial analysis agents",
    version="0.1.0",
    lifespan=lifespan,
)


def _env() -> FinQueryEnvironment:
    return app.state.env


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    try:
        result = _env().reset(task_id=req.task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    action_dict = req.action.model_dump(exclude_none=True)
    result = _env().step(action_dict)
    return result


@app.get("/state")
async def state():
    return _env().state()


@app.get("/tasks")
async def tasks():
    action_schema = FinQueryAction.model_json_schema()
    return [
        TaskInfo(
            id=task_id,
            name=t["name"],
            difficulty=t["difficulty"],
            description=t["description"],
            max_steps=t["max_steps"],
            action_schema=action_schema,
        ).model_dump()
        for task_id, t in TASKS.items()
    ]


@app.post("/grader")
async def grader(req: GraderRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    task = TASKS[req.task_id]
    result = task["grader"].grade(req.final_answer)
    return GraderResponse(score=result["score"], breakdown=result["breakdown"]).model_dump()


@app.post("/baseline")
async def baseline():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")

    try:
        from server._baseline_runner import run_all_tasks
        scores = run_all_tasks(env=_env(), api_key=api_key)
        return BaselineResponse(scores=scores).model_dump()
    except ImportError:
        raise HTTPException(status_code=501, detail="Baseline runner not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
