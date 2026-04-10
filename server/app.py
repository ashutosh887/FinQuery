"""FastAPI app — all routes, WebSocket, CORS."""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from finquery.models import (
    BaselineResponse, FinQueryAction, GraderRequest, GraderResponse,
    ResetRequest, StepRequest, StepResponse, TaskInfo,
)
from server.database import get_history, get_leaderboard, init_db
from server.finquery_environment import TASK_META, FinQueryEnvironment
from server.graders import task1_grader, task2_grader, task3_grader

_GRADER_MAP = {
    "task1_easy": task1_grader,
    "task2_medium": task2_grader,
    "task3_hard": task3_grader,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.env = FinQueryEnvironment()
    yield


app = FastAPI(title="FinQuery", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _env() -> FinQueryEnvironment:
    return app.state.env


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "name": "FinQuery",
        "description": "OpenEnv RL environment for financial agent reasoning. 25 tickers, 7 sectors, 9 years, procedurally generated tasks.",
        "version": "0.2.0",
        "tasks": list(TASK_META.keys()),
        "tickers": sorted([
            "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM",
            "TSLA", "F", "GM", "TM", "JPM", "BAC", "WFC", "GS",
            "AMZN", "WMT", "COST", "JNJ", "UNH", "PFE", "XOM", "CVX", "CAT", "BA",
        ]),
        "years": list(range(2017, 2026)),
        "sectors": ["technology", "automotive", "banking", "retail", "healthcare", "energy", "industrials"],
        "endpoints": {
            "reset": "POST /reset", "step": "POST /step", "state": "GET /state",
            "tasks": "GET /tasks", "grader": "POST /grader", "baseline": "POST /baseline",
            "history": "GET /history", "leaderboard": "GET /leaderboard",
            "health": "GET /health", "websocket": "WS /ws",
        },
    }


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    try:
        result = _env().reset(
            task_id=req.task_id,
            agent_name=req.agent_name or "anonymous",
            seed=req.seed,
            size=req.size,
            config=req.config,
            task_specs=req.task_specs,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    action_dict = req.action.model_dump(exclude_none=True)
    result = _env().step(episode_id=req.episode_id, action=action_dict)
    if result.get("observation", {}).get("tool_error") == "Unknown episode_id. Call /reset first.":
        raise HTTPException(status_code=404, detail="Unknown episode_id. Call /reset first.")
    return result


@app.get("/state")
async def state(episode_id: Optional[str] = Query(None)):
    return _env().state(episode_id=episode_id)


@app.get("/tasks")
async def tasks():
    action_schema = FinQueryAction.model_json_schema()
    return [
        TaskInfo(
            id=task_id, name=t["name"], difficulty=t["difficulty"],
            description=t["description"], max_steps=t["max_steps"],
            action_schema=action_schema,
        ).model_dump()
        for task_id, t in TASK_META.items()
    ]


@app.post("/grader")
async def grader(req: GraderRequest):
    if req.task_id not in _GRADER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    result = _GRADER_MAP[req.task_id].grade(req.final_answer)
    clamped_score = max(0.01, min(0.99, result["score"]))
    return GraderResponse(score=clamped_score, breakdown=result["breakdown"]).model_dump()


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


@app.get("/history")
async def history(limit: int = Query(50, ge=1, le=500), task_id: Optional[str] = Query(None)):
    return get_history(limit=limit, task_id=task_id)


@app.get("/leaderboard")
async def leaderboard(limit: int = Query(20, ge=1, le=100), task_id: Optional[str] = Query(None)):
    return get_leaderboard(limit=limit, task_id=task_id)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    env = _env()
    current_episode_id: str | None = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")
            if msg_type == "reset":
                try:
                    result = env.reset(
                        task_id=msg.get("task_id"),
                        agent_name=msg.get("agent_name", "anonymous"),
                        seed=msg.get("seed"),
                        size=msg.get("size"),
                        task_specs=msg.get("task_specs"),
                    )
                    current_episode_id = result["episode_id"]
                    await ws.send_json({"type": "reset_result", **result})
                except ValueError as e:
                    await ws.send_json({"type": "error", "detail": str(e)})
            elif msg_type == "step":
                eid = msg.get("episode_id", current_episode_id)
                if not eid:
                    await ws.send_json({"type": "error", "detail": "No episode_id. Send reset first."})
                    continue
                result = env.step(episode_id=eid, action=msg.get("action", {}))
                await ws.send_json({"type": "step_result", **result})
            elif msg_type == "state":
                eid = msg.get("episode_id", current_episode_id)
                result = env.state(episode_id=eid)
                await ws.send_json({"type": "state_result", **result})
            else:
                await ws.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})
    except WebSocketDisconnect:
        pass


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
