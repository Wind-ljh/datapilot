"""多轮对话上下文构造与会话持久化（纯函数 / 文件存储）单元测试。"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal

from app.conversation_store import ConversationStore, derive_title, new_id
from server.agents.context import build_compact_history, build_context, context_to_prompt


# --------------------------------------------------------------------------- 上下文构造
def test_build_compact_history_compresses_assistant_and_windows():
    msgs = []
    for i in range(5):  # 5 轮 = 10 条
        msgs.append({"role": "user", "content": f"问题{i}"})
        msgs.append({"role": "assistant", "content": f"结论{i}。这里是第二句很长很长很长的补充说明。"})
    compact = build_compact_history(msgs, max_turns=3)
    # 只保留最近 3 轮 = 6 条
    assert len(compact) == 6
    assert compact[0]["content"] == "问题2"
    # 助手内容只保留首句
    assert compact[-1]["role"] == "assistant"
    assert compact[-1]["content"] == "结论4"


def test_build_compact_history_ignores_non_chat_roles():
    msgs = [{"role": "system", "content": "忽略"}, {"role": "user", "content": "你好"}]
    compact = build_compact_history(msgs)
    assert compact == [{"role": "user", "content": "你好"}]


def test_build_compact_history_cuts_at_earliest_terminator():
    """首句压缩应取最早出现的结束符，而非固定字符顺序（"！"早于"。"时应在"！"截断）。"""
    msgs = [
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "环比增长 15%！其中电子品类贡献最大。整体向好。"},
    ]
    compact = build_compact_history(msgs)
    assert compact[-1]["content"] == "环比增长 15%"


def test_build_context_extracts_structured_state():
    state = {
        "question": "各品类销售额",
        "sql": "SELECT c, SUM(s) FROM t GROUP BY c",
        "chart": "bar",
        "result": {"columns": ["c", "s"], "rows": [["A", 1], ["B", 2]], "rowcount": 2},
        "linked_tables": ["orders"],
    }
    ctx = build_context(state)
    assert ctx["question"] == "各品类销售额"
    assert ctx["sql"] == state["sql"]
    assert ctx["chart"] == "bar"
    assert ctx["columns"] == ["c", "s"]
    assert ctx["rows"] == [["A", 1], ["B", 2]]
    assert ctx["rowcount"] == 2
    assert ctx["tables"] == ["orders"]


def test_context_to_prompt_contains_key_fields():
    ctx = {
        "question": "各品类销售额",
        "sql": "SELECT category FROM orders",
        "chart": "bar",
        "columns": ["category", "sales"],
        "rows": [["A", 10]],
        "rowcount": 1,
        "tables": ["orders"],
    }
    text = context_to_prompt(ctx)
    assert "各品类销售额" in text
    assert "SELECT category FROM orders" in text
    assert "bar" in text
    assert "orders" in text
    assert "category" in text


def test_context_to_prompt_empty_returns_empty():
    assert context_to_prompt(None) == ""
    assert context_to_prompt({}) == ""


# --------------------------------------------------------------------------- 标题
def test_derive_title_uses_first_user_message_and_truncates():
    assert derive_title([{"role": "user", "content": "每月的订单量趋势"}]) == "每月的订单量趋势"
    long_q = "这是一个非常非常非常非常非常长的问题标题需要被截断处理"
    assert derive_title([{"role": "user", "content": long_q}]).endswith("…")
    assert derive_title([]) == "未命名对话"


# --------------------------------------------------------------------------- 会话存储
def test_store_roundtrip_and_listing(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    store.upsert({"id": "a1", "title": "对话A", "messages": [], "context": None})
    store.upsert({"id": "b2", "title": "对话B", "messages": [], "context": None})

    assert store.get("a1")["title"] == "对话A"
    metas = store.list_metas()
    assert {m["id"] for m in metas} == {"a1", "b2"}
    # updated_at 倒序：后写的 b2 在前
    assert metas[0]["id"] == "b2"


def test_store_upsert_preserves_created_at(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    store.upsert({"id": "x", "title": "旧标题", "messages": [], "context": None})
    first = store.get("x")
    store.upsert({"id": "x", "title": "新标题", "messages": [], "context": None})
    second = store.get("x")
    assert second["title"] == "新标题"
    assert second["created_at"] == first["created_at"]  # 创建时间不因更新而变


def test_store_delete_isolates_other_conversations(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    store.upsert({"id": "a", "title": "A", "messages": [], "context": None})
    store.upsert({"id": "b", "title": "B", "messages": [], "context": None})
    store.delete("a")
    assert store.get("a") is None
    assert store.get("b")["id"] == "b"  # B 不受影响


def test_store_serializes_decimal_and_date(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    store.upsert(
        {
            "id": "x",
            "title": "t",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "result": {
                        "columns": ["金额", "日期"],
                        "rows": [[Decimal("1.5"), dt.date(2024, 1, 1)]],
                        "rowcount": 1,
                    },
                }
            ],
            "context": None,
        }
    )
    loaded = store.get("x")
    row = loaded["messages"][0]["result"]["rows"][0]
    assert row[0] == 1.5  # Decimal → float
    assert row[1] == "2024-01-01"  # date → ISO 字符串


def test_store_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ConversationStore(path)
    assert store.list_metas() == []
    assert store.get("whatever") is None
    # 后续写盘可正常恢复
    store.upsert({"id": "ok", "title": "t", "messages": [], "context": None})
    assert store.get("ok") is not None


def test_new_id_unique():
    ids = {new_id() for _ in range(100)}
    assert len(ids) == 100


def test_store_file_is_valid_json(tmp_path):
    store = ConversationStore(tmp_path / "conv.json")
    store.upsert({"id": "a", "title": "A", "messages": [{"role": "user", "content": "中文"}], "context": None})
    raw = json.loads((tmp_path / "conv.json").read_text(encoding="utf-8"))
    assert raw[0]["title"] == "A"
