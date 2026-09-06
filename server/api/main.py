"""FastAPI 服务：POST /api/query（ChatBI 问答）、GET /api/schema、GET /api/health。

启动：uvicorn server.api.main:app --reload --port 8000
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.agents.pipeline import build_pipeline
from server.core.config import get_settings

app = FastAPI(title="DataPilot ChatBI API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="中文分析问题")
    history: list[dict] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    need_clarify: bool = False
    clarify_question: str = ""
    sql: str = ""
    columns: list[str] = []
    rows: list[list] = []
    rowcount: int = 0
    report: str = ""
    chart: str = "table"
    degraded: bool = False
    metrics: dict = {}


@lru_cache(maxsize=1)
def _pipeline():
    return build_pipeline()


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "mock": s.mock,
        "llm": s.llm.provider if s.mock else s.llm.model,
        "db": str(s.db_path),
    }


@app.get("/api/schema")
def schema() -> dict:
    tables = _pipeline().tools.get_schema()
    return {"tables": tables}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        state = _pipeline().run(req.question, history=req.history)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"pipeline error: {e}") from e
    result = state.get("result") or {}
    return QueryResponse(
        question=req.question,
        need_clarify=state.get("need_clarify", False),
        clarify_question=state.get("clarify_question", ""),
        sql=state.get("sql", ""),
        columns=result.get("columns", []),
        rows=[list(r) for r in result.get("rows", [])],
        rowcount=result.get("rowcount", 0),
        report=state.get("report", ""),
        chart=state.get("chart", "table"),
        degraded=state.get("degraded", False),
        metrics={
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "llm_calls": state.get("llm_calls", 0),
            "stage_latency_ms": state.get("stage_latency_ms", {}),
            "repair_round": state.get("repair_round", 0),
        },
    )
