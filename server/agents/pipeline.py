"""ChatBI 多阶段 Agent 流水线（LangGraph 编排）。

图结构：
  START → clarify →(需澄清→ END)→ link → generate ─┬─ 校验通过 → execute ─┬─ 成功 → report → END
                                                   ├─ 校验失败 → repair → generate（自修复回环）
                                                   └─ 修复超限 → report(degraded)

设计取舍（面试讲点）：
- Workflow 主干 + 局部 Agent 回环：SQL 生成主干固定（可控、可测、省 token），
  仅在"校验/执行失败"时把控制权交给 LLM 自修复（灵活性用在刀刃上）；
- 每个节点是纯函数：只通过【返回值】更新状态（LangGraph 语义），不原地改入参，
  观测指标（tokens/trace/延迟）因此可完整持久化，供评估与演示展示。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from server.agents.prompts import (
    CLARIFY_SYSTEM,
    GENERATE_SYSTEM,
    REPAIR_USER,
    REPORT_SYSTEM,
    REPORT_USER,
)
from server.agents.state import ChatBIState, new_state
from server.core.config import get_settings
from server.core.database import DuckDBAdapter
from server.core.llm import LLMClient, LLMResponse, MockLLMClient, build_llm, extract_sql
from server.mcp_tools.tools import ToolCore
from server.retrieval.fewshot import FewShotStore
from server.retrieval.linking import SchemaLinker


def _rows_to_markdown(columns: list[str], rows: list[list], max_rows: int = 15) -> str:
    if not columns:
        return "（空结果）"
    shown = rows[:max_rows]
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in shown)
    tail = f"\n（仅展示前 {len(shown)} 行，共 {len(rows)} 行）" if len(rows) > max_rows else ""
    return f"{head}\n{sep}\n{body}{tail}"


class ChatBIPipeline:
    def __init__(
        self,
        tools: ToolCore,
        llm: LLMClient | MockLLMClient,
        linker: SchemaLinker | None = None,
        fewshot: FewShotStore | None = None,
        max_repair_rounds: int | None = None,
        dialect: str = "DuckDB",
        enable_clarify: bool = True,
        enable_linking: bool = True,
        enable_fewshot: bool = True,
    ):
        self.tools = tools
        self.llm = llm
        self.linker = linker
        self.fewshot = fewshot
        settings = get_settings()
        self.max_repair_rounds = settings.max_repair_rounds if max_repair_rounds is None else max_repair_rounds
        self.dialect = dialect
        self.enable_clarify = enable_clarify and not isinstance(llm, MockLLMClient)
        self.enable_linking = enable_linking
        self.enable_fewshot = enable_fewshot
        self._graph = self._build_graph().compile()

    # ------------------------------------------------------------------ 图结构
    def _build_graph(self) -> StateGraph:
        g = StateGraph(ChatBIState)
        g.add_node("clarify", self._clarify)
        g.add_node("link", self._link)
        g.add_node("generate", self._generate)
        g.add_node("repair", self._repair)
        g.add_node("execute", self._execute)
        g.add_node("report", self._report)

        g.add_edge(START, "clarify" if self.enable_clarify else "link")
        if self.enable_clarify:
            g.add_conditional_edges(
                "clarify",
                lambda s: END if s.get("need_clarify") else "link",
                {END: END, "link": "link"},
            )
        g.add_edge("link", "generate")
        g.add_conditional_edges(
            "generate",
            self._route_after_generate,
            {"execute": "execute", "repair": "repair", "report": "report"},
        )
        g.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"report": "report", "repair": "repair"},
        )
        g.add_edge("repair", "generate")
        g.add_edge("report", END)
        return g

    # ------------------------------------------------------------------ 观测工具
    def _chat(
        self, state: ChatBIState, stage: str, messages: list[dict[str, Any]]
    ) -> tuple[LLMResponse, dict[str, Any]]:
        """调用 LLM，返回 (响应, 应并入节点返回值的观测增量)。"""
        resp = self.llm.chat(messages, temperature=0.0)
        stage_ms = state.get("stage_latency_ms", {})
        updates: dict[str, Any] = {
            "prompt_tokens": state.get("prompt_tokens", 0) + resp.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + resp.completion_tokens,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "stage_latency_ms": {
                **stage_ms,
                stage: stage_ms.get(stage, 0) + resp.latency_ms,
            },
        }
        return resp, updates

    def _trace(self, state: ChatBIState, message: str) -> dict[str, Any]:
        return {"trace": state.get("trace", []) + [message]}

    # ------------------------------------------------------------------ 节点
    def _clarify(self, state: ChatBIState) -> dict[str, Any]:
        tables = ", ".join(self.tools.list_tables())
        messages = [
            {"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user", "content": f"可用表：{tables}\n用户问题：{state['question']}"},
        ]
        resp, updates = self._chat(state, "clarify", messages)
        need, question = False, ""
        try:
            obj = json.loads(re.search(r"\{.*\}", resp.text, re.DOTALL).group(0))
            need = bool(obj.get("need_clarify", False))
            question = str(obj.get("question", ""))
        except (AttributeError, ValueError):
            pass
        return {
            **updates,
            **self._trace(state, f"clarify: need={need}"),
            "need_clarify": need,
            "clarify_question": question,
        }

    def _link(self, state: ChatBIState) -> dict[str, Any]:
        start = time.perf_counter()
        linked_text, linked_tables, pairs = "", [], []
        if self.enable_linking and self.linker is not None:
            linked = self.linker.link(state["question"], top_k_tables=3)
            linked_text = linked.to_prompt_text()
            linked_tables = [t.name for t in linked.tables]
        if self.enable_fewshot and self.fewshot is not None:
            pairs = [
                {"question": ex.question, "sql": ex.sql}
                for ex in self.fewshot.top_k(state["question"], k=3)
            ]
        elapsed = int((time.perf_counter() - start) * 1000)
        return {
            "linked_schema_text": linked_text,
            "linked_tables": linked_tables,
            "fewshot_pairs": pairs,
            "stage_latency_ms": {**state.get("stage_latency_ms", {}), "link": elapsed},
            **self._trace(state, f"link: tables={linked_tables}, fewshot={len(pairs)}"),
        }

    def _generate(self, state: ChatBIState) -> dict[str, Any]:
        schema_text = state.get("linked_schema_text") or ""
        if not schema_text:
            # 未启用 linking 的消融模式：直接给全量 schema
            schema_text = "\n\n".join(
                f"TABLE {s['name']}\n"
                + "\n".join(
                    f"  {c['name']} {c['type']}" + (f"  -- {c['comment']}" if c["comment"] else "")
                    for c in s["columns"]
                )
                for s in self.tools.get_schema()
            )

        fewshot_text = ""
        for i, pair in enumerate(state.get("fewshot_pairs", []), 1):
            fewshot_text += f"\n示例{i}\n问题：{pair['question']}\nSQL：\n```sql\n{pair['sql']}\n```\n"

        user_parts = [f"数据库 schema：\n{schema_text}"]
        if fewshot_text:
            user_parts.append(f"参考示例（相似问题及其 SQL）：{fewshot_text}")
        if state.get("repair_round", 0) > 0 and state.get("sql_error"):
            user_parts.append(REPAIR_USER.format(sql=state.get("sql", ""), error=state["sql_error"]))
        if state.get("history"):
            user_parts.append("对话历史：" + json.dumps(state["history"], ensure_ascii=False))
        user_parts.append(f"用户问题：{state['question']}")

        messages = [
            {"role": "system", "content": GENERATE_SYSTEM.format(dialect=self.dialect)},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        resp, updates = self._chat(state, "generate", messages)
        sql = extract_sql(resp.text)
        verdict = self.tools.validate_sql(sql)  # {"ok": bool, "message": str}
        ok = bool(verdict.get("ok"))
        message = str(verdict.get("message", ""))
        return {
            **updates,
            **self._trace(state, f"generate(round={state.get('repair_round', 0)}): valid={ok}, sql={sql[:80]!r}"),
            "sql": sql,
            "sql_valid": ok,
            "sql_error": "" if ok else message,
        }

    def _repair(self, state: ChatBIState) -> dict[str, Any]:
        return {
            "repair_round": state.get("repair_round", 0) + 1,
            **self._trace(state, f"repair: 进入第 {state.get('repair_round', 0) + 1} 轮自修复"),
        }

    def _execute(self, state: ChatBIState) -> dict[str, Any]:
        result = self.tools.execute_sql(state["sql"])
        if result.get("ok"):
            return {
                "result": result,
                "executed": True,
                "sql_error": "",
                "stage_latency_ms": {
                    **state.get("stage_latency_ms", {}),
                    "execute": int(result.get("elapsed_ms", 0)),
                },
                **self._trace(state, f"execute: ok rows={result.get('rowcount')}"),
            }
        return {
            "executed": False,
            "sql_error": result.get("error", "执行失败"),
            **self._trace(state, f"execute: failed {str(result.get('error', ''))[:80]!r}"),
        }

    def _report(self, state: ChatBIState) -> dict[str, Any]:
        result = state.get("result")
        if not result:
            return {
                "degraded": True,
                "report": "抱歉，这个问题暂时没能生成可执行的查询。可以尝试换个说法，或补充时间范围/统计口径后重试。",
                "chart": "table",
                **self._trace(state, "report: degraded 兜底"),
            }
        table_md = _rows_to_markdown(result["columns"], result["rows"])
        messages = [
            {"role": "system", "content": REPORT_SYSTEM},
            {
                "role": "user",
                "content": REPORT_USER.format(
                    question=state["question"],
                    rowcount=result.get("rowcount", 0),
                    shown=min(len(result.get("rows", [])), 15),
                    table_md=table_md,
                ),
            },
        ]
        resp, updates = self._chat(state, "report", messages)
        chart = "table"
        m = re.search(r"图表类型[:：]\s*(bar|line|pie|table)", resp.text)
        if m:
            chart = m.group(1)
        return {
            **updates,
            **self._trace(state, f"report: chart={chart}"),
            "report": resp.text,
            "chart": chart,
        }

    # ------------------------------------------------------------------ 路由
    def _route_after_generate(self, state: ChatBIState) -> str:
        if state.get("sql_valid"):
            return "execute"
        if state.get("repair_round", 0) < self.max_repair_rounds:
            return "repair"
        return "report"

    def _route_after_execute(self, state: ChatBIState) -> str:
        if state.get("executed"):
            return "report"
        if state.get("repair_round", 0) < self.max_repair_rounds:
            return "repair"
        return "report"

    # ------------------------------------------------------------------ 对外接口
    def run(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        final: ChatBIState = self._graph.invoke(new_state(question, history))
        return dict(final)


def build_pipeline(db_path: str | None = None, **overrides) -> ChatBIPipeline:
    """装配完整流水线（默认从配置读取）。供 API / Streamlit / 演示使用。"""
    adapter = DuckDBAdapter(path=db_path)
    tools = ToolCore(adapter)
    from pathlib import Path

    from server.core.embeddings import build_embedder

    linker = SchemaLinker(adapter, build_embedder())
    fewshot: FewShotStore | None = None
    default_fewshot = Path(__file__).resolve().parents[2] / "data" / "fewshot_examples.jsonl"
    if default_fewshot.exists():
        fewshot = FewShotStore.from_jsonl(default_fewshot, build_embedder())
    return ChatBIPipeline(tools=tools, llm=build_llm("llm"), linker=linker, fewshot=fewshot, **overrides)
