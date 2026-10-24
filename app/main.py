from typing import Any, Dict, List

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from . import agent as agent_mod
from . import tools as toolbox
from .config import settings
from .seed import seed

app = FastAPI(title="Toolsmith: Multi-Tool AI Agent")
seed(settings.sample_db_path)


class RunRequest(BaseModel):
    task: str = Field(min_length=3, max_length=2000)


class StepOut(BaseModel):
    step: int
    tool: str
    args: Dict[str, Any]
    observation: str


class UsageOut(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    est_cost_usd: float


class RunResponse(BaseModel):
    answer: str
    stopped_reason: str
    steps: List[StepOut]
    usage: UsageOut


def get_agent_runner():
    return agent_mod.run_agent


@app.get("/tools")
def list_tools():
    return [{"name": d["name"], "description": d["description"]} for d in toolbox.DECLARATIONS]


@app.post("/agent/run", response_model=RunResponse)
def run(req: RunRequest, runner=Depends(get_agent_runner)):
    res = runner(req.task)
    return RunResponse(
        answer=res.answer,
        stopped_reason=res.stopped_reason,
        steps=[StepOut(step=s.step, tool=s.tool, args=s.args, observation=s.observation) for s in res.steps],
        usage=UsageOut(
            llm_calls=res.llm_calls,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            total_tokens=res.total_tokens,
            est_cost_usd=round(res.est_cost_usd, 6),
        ),
    )
