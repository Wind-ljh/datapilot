"""流水线端到端测试（mock 模式，无网络）。"""

from __future__ import annotations

from server.agents.pipeline import ChatBIPipeline
from server.core.database import DuckDBAdapter
from server.core.llm import MockLLMClient
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
