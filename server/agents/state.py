"""Agent 流水线的共享状态定义（LangGraph TypedDict）。"""

from __future__ import annotations

from typing import Any, TypedDict


class ChatBIState(TypedDict, total=False):
    # 输入
    question: str
    history: list[dict]  # 多轮历史 [{"role": ..., "content": ...}]
    context: dict  # 上一轮结构化分析上下文（问题/SQL/结果/图表/表），用于追问的增量修改

    # 澄清
    need_clarify: bool
    clarify_question: str

    # 检索产物
    linked_schema_text: str
    linked_tables: list[str]
    fewshot_pairs: list[dict]  # [{"question":..., "sql":...}]

    # SQL 生成与执行
    sql: str
    sql_valid: bool
    sql_error: str
    repair_round: int
    executed: bool

    # 执行结果
    result: dict  # {"columns":..., "rows":..., "rowcount":..., "elapsed_ms":...}

    # 报告
    report: str
    chart: str  # bar | line | pie | table
    degraded: bool  # True = 未成功产出结果，走了兜底回复

    # 观测性（指标采集）
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    stage_latency_ms: dict[str, int]
    trace: list[str]


def new_state(
    question: str, history: list[dict] | None = None, context: dict | None = None
) -> dict[str, Any]:
    return {
        "question": question,
        "history": history or [],
        "context": context or {},
        "need_clarify": False,
        "clarify_question": "",
        "repair_round": 0,
        "degraded": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        "stage_latency_ms": {},
        "trace": [],
    }
