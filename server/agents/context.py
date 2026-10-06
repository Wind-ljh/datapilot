"""多轮对话的上下文构造与序列化（纯函数，不依赖 Streamlit / LLM，便于单测）。

设计目标：让后续问题（"第二个 / 比较一下 / 只看前 N / 换成饼图"等指代）能
可靠地基于上一轮分析做增量修改，同时避免把整段冗长历史塞给 LLM。

三个职责：
1. build_compact_history —— 只取最近 N 轮，助手内容压缩为首句结论；
2. build_context —— 把上一轮的结构化分析状态（问题 / SQL / 结果 / 图表 / 表）打包；
3. context_to_prompt —— 把结构化上下文转成给 LLM 的提示词片段。
"""

from __future__ import annotations

from typing import Any

MAX_CONTEXT_ROWS = 10   # 上下文里最多携带的结果行数
MAX_HISTORY_TURNS = 3   # 最近 N 轮对话（每轮 = 用户 + 助手）
MAX_SUMMARY_CHARS = 200  # 助手结论压缩后的最大长度


def _first_sentence(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """取首句（到最早的句号/换行/感叹/问号为止），超长则截断加省略号。"""
    text = (text or "").strip()
    if not text:
        return ""
    # 取所有结束符中最早出现的位置（而非按固定字符顺序匹配）
    ends = [idx for ch in "。！？!?\n" if (idx := text.find(ch)) > 0]
    if ends:
        text = text[: min(ends)]
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def build_compact_history(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """把消息列表压缩为最近 max_turns 轮的 {role, content} 序列。

    助手消息只保留首句结论，避免把完整报告重复喂回模型；保留 role 语义，
    兼容现有 ChatBIState.history 的 {role, content} 结构。
    """
    compact: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content", "") or "")
        if role == "assistant":
            content = _first_sentence(content)
        compact.append({"role": role, "content": content})
    limit = max(0, max_turns) * 2
    if len(compact) > limit:
        compact = compact[-limit:]
    return compact


def build_context(state: dict[str, Any]) -> dict[str, Any]:
    """从一次 pipeline 结果中抽取下一轮要用的结构化分析上下文。"""
    result = state.get("result") or {}
    rows = result.get("rows") or []
    return {
        "question": state.get("question", ""),
        "sql": state.get("sql", ""),
        "chart": state.get("chart", "table"),
        "columns": list(result.get("columns") or []),
        "rows": [list(r) for r in rows][:MAX_CONTEXT_ROWS],
        "rowcount": int(result.get("rowcount", len(rows))),
        "tables": list(state.get("linked_tables") or []),
    }


def rows_to_markdown(columns: list, rows: list[list], max_rows: int | None = None) -> str:
    """把查询结果渲染成 markdown 表格；空列返回空串，超 max_rows 截断并附注。

    context 与 report 节点共用，避免两处重复实现。
    """
    if not columns:
        return ""
    shown = rows if max_rows is None else rows[:max_rows]
    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in shown)
    tail = ""
    if max_rows is not None and len(rows) > max_rows:
        tail = f"\n（仅展示前 {len(shown)} 行，共 {len(rows)} 行）"
    return f"{head}\n{sep}\n{body}{tail}"


def context_to_prompt(context: dict | None) -> str:
    """把结构化上下文转成提示词片段；空上下文返回空串。"""
    if not context:
        return ""
    lines = [f"- 问题：{context.get('question', '')}"]
    if context.get("sql"):
        lines.append(f"- SQL：{context['sql']}")
    lines.append(f"- 图表类型：{context.get('chart', 'table')}")
    tables = context.get("tables") or []
    if tables:
        lines.append(f"- 涉及表：{', '.join(str(t) for t in tables)}")
    columns = list(context.get("columns") or [])
    rows = [list(r) for r in (context.get("rows") or [])]
    rowcount = context.get("rowcount", len(rows))
    lines.append(f"- 结果（{rowcount} 行，展示前 {len(rows)} 行）：")
    table = rows_to_markdown(columns, rows)
    if table:
        lines.append(table)
    return "\n".join(lines)
