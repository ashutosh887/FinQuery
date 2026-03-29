"""Pydantic models defining the FinQueryGym API contract.

This is the single source of truth for action, observation, and state types.
If you change a field here, update client.py, openenv.yaml, and README.md.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class FinQueryAction(BaseModel):
    action_type: Literal[
        "get_income_statement",
        "get_balance_sheet",
        "get_cash_flow",
        "get_price_history",
        "get_ratios",
        "compare_to_sector",
        "compute",
        "submit_answer",
    ]
    ticker: Optional[str] = Field(None, description="e.g. 'AAPL', 'MSFT'")
    year: Optional[int] = Field(None, description="e.g. 2023")
    years: Optional[List[int]] = Field(None, description="e.g. [2020, 2021, 2022, 2023]")
    metric: Optional[str] = Field(None, description="e.g. 'revenue', 'pe_ratio'")
    expression: Optional[str] = Field(None, description="Arithmetic expression for compute")
    answer: Optional[Any] = Field(None, description="Final answer for submit_answer")
    reasoning: Optional[str] = Field(None, description="Agent's chain of thought (logged, not graded)")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FinQueryObservation(BaseModel):
    task_description: str
    tool_result: Optional[Dict[str, Any]] = None
    tool_error: Optional[str] = None
    steps_taken: int
    steps_remaining: int
    tickers_queried: List[str]
    episode_status: Literal["ongoing", "answered", "failed_max_steps"]
    feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FinQueryState(BaseModel):
    episode_id: str
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    fetched_data: Dict[str, Any]
    answer_submitted: bool
    score_so_far: float


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: FinQueryObservation
    reward: float
    done: bool


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: FinQueryAction


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    action_schema: Dict[str, Any]


class GraderRequest(BaseModel):
    episode_id: Optional[str] = None
    task_id: str
    final_answer: Any


class GraderResponse(BaseModel):
    score: float
    breakdown: Dict[str, Any]


class BaselineResponse(BaseModel):
    scores: Dict[str, float]
