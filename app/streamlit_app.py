"""DataPilot ChatBI 演示界面(Streamlit)。

启动:streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataPilot · ChatBI 演示", page_icon="📊", layout="wide")

st.title("📊 DataPilot · 中文 ChatBI 数据分析助手")
st.caption("自然语言 → SQL → 图表与结论 | MCP 工具集 + LangGraph 多阶段 Agent + 混合检索 RAG")


@st.cache_resource(show_spinner="初始化流水线（加载演示库 + 构建检索索引）…")
def get_pipeline():
    from server.agents.pipeline import build_pipeline

    return build_pipeline()


PIPE = get_pipeline()

SAMPLE_QUESTIONS = [
    "每月的订单量和收入趋势是怎样的",
    "销量最高的10个商品是哪些",
    "各品类的销售收入排名",
    "各城市收入 Top10",
    "退款订单占比是多少",
    "周末和工作日的订单量对比",
]

with st.sidebar:
    st.header("设置")
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("示例问题")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
    st.divider()
    st.caption(
        "技术栈:GLM-4.7-Flash / LangGraph / FastMCP / DuckDB / BGE-M3\n\n"
        "无 API Key 时自动降级为 mock 模式（固定演示 SQL)"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_df") is not None and not msg["chart_df"].empty:
            _render = {
                "bar": st.bar_chart,
                "line": st.line_chart,
                "area": st.area_chart,
            }.get(msg.get("chart"), st.dataframe)
            _render(msg["chart_df"])
        if msg.get("sql"):
            with st.expander("查看生成的 SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("metrics"):
            with st.expander("运行指标（Token / 各阶段延迟）"):
                st.json(msg["metrics"])

question = st.chat_input("用中文问一个数据分析问题，例如：每个月的订单量和收入趋势")
pending = st.session_state.pop("pending_question", None)
if pending:
    question = pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("分析中：检索 schema → 生成 SQL → 执行校验 → 撰写结论…"):
            state = PIPE.run(question)

        if state.get("need_clarify"):
            content = f"🤔 {state.get('clarify_question') or '能补充一下时间范围或统计口径吗？'}"
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
        elif state.get("degraded"):
            content = state.get("report", "这个问题暂时没能生成可执行的查询，换个说法试试？")
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
        else:
            result = state.get("result", {})
            df = pd.DataFrame(result.get("rows", []), columns=result.get("columns", []))
            st.markdown(state.get("report", "（无报告）"))
            if not df.empty and state.get("chart") in {"bar", "line", "area"} and df.select_dtypes("number").shape[1] > 0:
                # 数值列作为 y，第一列作为 x
                try:
                    df_indexed = df.set_index(df.columns[0])
                    {"bar": st.bar_chart, "line": st.line_chart, "area": st.area_chart}[state["chart"]](df_indexed)
                except Exception:
                    st.dataframe(df)
            elif not df.empty:
                st.dataframe(df)
            if state.get("sql"):
                with st.expander("查看生成的 SQL"):
                    st.code(state["sql"], language="sql")
            metrics = {
                "tokens": {"prompt": state.get("prompt_tokens", 0), "completion": state.get("completion_tokens", 0)},
                "llm_calls": state.get("llm_calls", 0),
                "stage_latency_ms": state.get("stage_latency_ms", {}),
                "repair_round": state.get("repair_round", 0),
                "linked_tables": state.get("linked_tables", []),
            }
            with st.expander("运行指标"):
                st.json(metrics)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": state.get("report", ""),
                    "sql": state.get("sql", ""),
                    "chart": state.get("chart"),
                    "chart_df": df if state.get("chart") in {"bar", "line", "area"} else None,
                    "metrics": metrics,
                }
            )
