"""流水线端到端测试（mock 模式，无网络）。"""

from __future__ import annotations

from server.agents.pipeline import ChatBIPipeline
from server.core.database import DuckDBAdapter
from server.core.llm import LLMResponse, MockLLMClient
from server.mcp_tools.tools import ToolCore


def test_monthly_question_end_to_end(pipeline):
    state = pipeline.run("每月的订单量和收入趋势是怎样的")
    assert state["executed"] is True
    assert "orders" in state["sql"].lower()
    assert "strftime" in state["sql"]
    assert state["report"], "报告不应为空"
    assert state["prompt_tokens"] + state["completion_tokens"] > 0
    assert state["trace"], "应有决策日志"


def test_top_products_question(pipeline):
    state = pipeline.run("销量最高的10个商品是哪些")
    assert state["executed"]
    assert "order_items" in state["sql"] and "products" in state["sql"]


def test_clarify_disabled_in_mock(pipeline):
    state = pipeline.run("总订单数是多少")
    assert state.get("need_clarify") is False


def test_repair_loop_and_degraded_path():
    """空库 + mock：生成的 SQL 必然校验失败 → 走满自修复回环 → 兜底报告。"""
    adapter = DuckDBAdapter(path=":memory:")  # 空库，任何表都不存在
    pipe = ChatBIPipeline(
        tools=ToolCore(adapter),
        llm=MockLLMClient(),
        linker=None,
        fewshot=None,
        max_repair_rounds=2,
        dialect="DuckDB",
    )
    state = pipe.run("总订单数是多少")
    assert state["degraded"] is True
    assert state["repair_round"] == 2  # 修复到上限
    assert not state.get("executed")
    assert "暂时" in state["report"]


def test_ablation_flags_disable_components(adapter, tools):
    pipe = ChatBIPipeline(
        tools=tools,
        llm=MockLLMClient(),
        linker=None,
        fewshot=None,
        dialect="DuckDB",
    )
    state = pipe.run("各订单状态分别有多少订单")
    assert state["linked_schema_text"] == ""
    assert state["fewshot_pairs"] == []
    # 全量 schema 兜底路径仍可执行
    assert state["executed"] is True


class _RecordingLLM:
    """记录每次 chat 调用消息的假 LLM：生成阶段返回固定可执行 SQL，报告阶段返回固定结论。"""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def chat(self, messages, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)
        last = messages[-1]["content"]
        if "请按系统要求输出分析报告" in last:
            text = "结论正确，无需改动。\n图表类型：bar"
        else:
            text = "```sql\nSELECT 1 AS x\n```"
        return LLMResponse(text=text, prompt_tokens=1, completion_tokens=1, latency_ms=0)


def test_followup_context_is_threaded_into_generate_and_report(tools):
    """追问场景：结构化上下文 + 历史应同时进入 generate 与 report 的提示词。"""
    llm = _RecordingLLM()
    pipe = ChatBIPipeline(
        tools=tools,
        llm=llm,
        linker=None,
        fewshot=None,
        enable_clarify=False,
        enable_linking=False,
        enable_fewshot=False,
    )
    ctx = {
        "question": "各品类销售额",
        "sql": "SELECT category, SUM(amount) FROM orders GROUP BY category",
        "chart": "bar",
        "columns": ["category", "sum(amount)"],
        "rows": [["A", 100]],
        "rowcount": 1,
        "tables": ["orders"],
    }
    state = pipe.run(
        "只看前 3 个",
        history=[{"role": "user", "content": "各品类销售额"}],
        context=ctx,
    )
    assert state["executed"] is True
    assert len(llm.calls) == 2  # generate + report

    generate_user = llm.calls[0][-1]["content"]
    assert "上一轮分析" in generate_user
    assert ctx["sql"] in generate_user
    assert "只看前 3 个" in generate_user
    assert "用户：各品类销售额" in generate_user  # 历史已读入

    report_user = llm.calls[1][-1]["content"]
    assert "上一轮分析" in report_user
    assert ctx["sql"] in report_user


def test_plain_question_does_not_inject_followup_guidance(tools):
    """单轮问题（无 context）不应出现追问引导，保持现有行为不变。"""
    llm = _RecordingLLM()
    pipe = ChatBIPipeline(
        tools=tools,
        llm=llm,
        linker=None,
        fewshot=None,
        enable_clarify=False,
        enable_linking=False,
        enable_fewshot=False,
    )
    pipe.run("总订单数是多少")
    generate_user = llm.calls[0][-1]["content"]
    assert "上一轮分析" not in generate_user
    assert "对话上下文" not in generate_user
