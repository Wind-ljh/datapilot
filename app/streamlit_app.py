"""DataPilot ChatBI 演示界面(Streamlit)。

启动:streamlit run app/streamlit_app.py

多轮对话数据分析：每条消息只保存 JSON 可序列化的结构化结果
（columns/rows/rowcount），图表 DataFrame 在渲染时从结果重建，保证刷新/切换
会话后仍可恢复；上一轮分析上下文（问题/SQL/结果/图表/表）随下一次提问传给
pipeline，支持"第二个 / 只看前 N / 换成饼图"等指代追问。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app.chart_utils import classify_result, format_number, metric_label, prepare_chart_df
from app.conversation_store import ConversationStore, derive_title, new_id
from server.agents.context import build_compact_history, build_context

st.set_page_config(page_title="DataPilot · ChatBI 演示", page_icon="📊", layout="wide")

st.title("📊 DataPilot · 中文 ChatBI 数据分析助手")
st.caption("自然语言 → SQL → 图表与结论 | MCP 工具集 + LangGraph 多阶段 Agent + 混合检索 RAG")

_CHART_FNS = {
    "bar": st.bar_chart,
    "line": st.line_chart,
    "area": st.area_chart,
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STORE_PATH = _REPO_ROOT / "data" / "conversations.json"
STORE = ConversationStore(Path(os.environ.get("DATAPILOT_CONVERSATIONS_PATH", str(_DEFAULT_STORE_PATH))))


def _render_chart(df: pd.DataFrame, chart: str | None) -> None:
    """统一渲染：按数据规模与意图选择 KPI / 紧凑图表 / 正常图表。

    历史消息与当前消息共用本函数，保证 rerun 前后渲染一致；始终显式指定
    x/y，绝不原地修改传入 df。
    """
    if df is None or df.empty:
        return
    work = prepare_chart_df(df)  # 渲染用副本，不污染历史 df
    spec = classify_result(work, chart)

    if spec.mode == "metric":
        st.metric(
            label=metric_label(work, spec.x, spec.y),
            value=format_number(work.iloc[0][spec.y]),
            border=True,
        )
    elif spec.mode in {"bar", "hbar", "line", "area"}:
        fn = _CHART_FNS["bar" if spec.mode == "hbar" else spec.mode]
        kwargs: dict = {}
        if spec.height is not None:
            kwargs["height"] = spec.height
        if spec.mode == "hbar":
            kwargs["horizontal"] = True
        try:
            if spec.x is not None:
                fn(work, x=spec.x, y=spec.y, **kwargs)
            else:
                fn(work, y=spec.y, **kwargs)
        except Exception as exc:  # 兜底：显式回退并给出提示，不隐藏错误
            st.caption(f"图表渲染失败，已回退为表格展示：{exc}")
    elif spec.y is None:
        st.caption("未能可靠识别图表的 X/Y 轴，已改为表格展示。")

    st.dataframe(df)


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


# --------------------------------------------------------------------------- 会话状态
def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_id" not in st.session_state:
        st.session_state.current_id = new_id()
    if "context" not in st.session_state:
        st.session_state.context = None


def _result_to_df(result: dict | None) -> pd.DataFrame | None:
    """从 JSON 安全的结构化结果重建 DataFrame（刷新/切换会话后仍可渲染）。"""
    if not result:
        return None
    columns = list(result.get("columns") or [])
    rows = [list(r) for r in (result.get("rows") or [])]
    if not columns:
        return None
    return pd.DataFrame(rows, columns=columns)


def _persist_current() -> None:
    """把当前会话写入磁盘；空会话不落盘。"""
    if not st.session_state.messages:
        return
    STORE.upsert(
        {
            "id": st.session_state.current_id,
            "title": derive_title(st.session_state.messages),
            "messages": st.session_state.messages,
            "context": st.session_state.context,
        }
    )


def _start_new_chat() -> None:
    """新对话：彻底清空消息与上下文，换取全新 id（与其它会话隔离）。"""
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.current_id = new_id()


def _open_chat(conv_id: str) -> None:
    """打开历史会话：恢复其消息与上下文（不污染其它会话）。"""
    conv = STORE.get(conv_id)
    if not conv:
        return
    st.session_state.messages = list(conv.get("messages") or [])
    st.session_state.context = conv.get("context")
    st.session_state.current_id = conv_id


_ensure_state()

# --------------------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.subheader("对话历史")
    if st.button("➕ 新对话", use_container_width=True):
        _persist_current()
        _start_new_chat()
        st.rerun()

    for meta in STORE.list_metas():
        label = meta["title"] or "未命名对话"
        if meta["id"] == st.session_state.current_id:
            label = "● " + label
        if st.button(label, key=f"open_{meta['id']}", use_container_width=True):
            _persist_current()
            _open_chat(meta["id"])
            st.rerun()

    st.divider()
    st.subheader("示例问题")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
    st.divider()
    st.caption(
        "技术栈:GLM-4.7-Flash / LangGraph / FastMCP / DuckDB / BGE-M3\n\n"
        "无 API Key 时自动降级为 mock 模式（固定演示 SQL)\n\n"
        "对话自动保存在本地 data/conversations.json"
    )

# --------------------------------------------------------------------------- 历史渲染
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        df = _result_to_df(msg.get("result"))
        if df is not None and not df.empty:
            _render_chart(df, msg.get("chart"))
        if msg.get("sql"):
            with st.expander("查看生成的 SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("metrics"):
            with st.expander("运行指标（Token / 各阶段延迟）"):
                st.json(msg["metrics"])

# --------------------------------------------------------------------------- 输入处理
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
            # 只传压缩后的最近 N 轮 + 上一轮结构化上下文，避免无限膨胀
            state = PIPE.run(
                question,
                history=build_compact_history(st.session_state.messages[:-1]),
                context=st.session_state.context,
            )

        if state.get("need_clarify"):
            content = f"🤔 {state.get('clarify_question') or '能补充一下时间范围或统计口径吗？'}"
            st.markdown(content)
            st.session_state.context = None
            st.session_state.messages.append({"role": "assistant", "content": content})
        elif state.get("degraded"):
            content = state.get("report", "这个问题暂时没能生成可执行的查询，换个说法试试？")
            st.markdown(content)
            st.session_state.context = None
            st.session_state.messages.append({"role": "assistant", "content": content})
        else:
            result = state.get("result", {})
            rows = [list(r) for r in (result.get("rows") or [])]
            columns = list(result.get("columns") or [])
            df = pd.DataFrame(rows, columns=columns)
            st.markdown(state.get("report", "（无报告）"))
            if not df.empty:
                _render_chart(df, state.get("chart"))
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
                    "result": {
                        "columns": columns,
                        "rows": rows,
                        "rowcount": result.get("rowcount", len(rows)),
                    },
                    "metrics": metrics,
                }
            )
            st.session_state.context = build_context(state)

    _persist_current()
