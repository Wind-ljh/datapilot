"""本地会话持久化（单个 JSON 文件，无数据库）。

设计取舍（面试讲点）：个人演示项目不引入数据库与账号系统。浏览器 localStorage
需要自定义前端组件（iframe + postMessage）桥接，复杂且脆弱；服务端单文件 JSON
原子写盘即可满足"刷新后历史仍在、多会话隔离"，简单可靠、零新依赖。

每个会话的结构（全部 JSON 可序列化，不含 DataFrame）：
{
  "id": str, "title": str, "created_at": iso, "updated_at": iso,
  "messages": [{"role","content","sql","chart","result":{columns,rows,rowcount},"metrics"}],
  "context": {上一轮结构化分析上下文} | None,
}
"""

from __future__ import annotations

import decimal
import json
import os
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any


def new_id() -> str:
    """生成短会话 id（uuid hex 前 12 位，足够本演示唯一性）。"""
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    """带时区的 ISO 时间戳，用于记录更新时间（展示用，不参与排序）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    """把 DuckDB 原生返回值（Decimal/date/datetime）转成 JSON 可序列化类型。

    日期转 ISO 字符串后，恢复时由 chart_utils 的日期识别/转换逻辑再转回时间轴；
    金额 Decimal 转 float（本演示金额量级远小于 float 精确范围，展示无精度损失）。
    """
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, (date, datetime, time)):
        return o.isoformat()
    raise TypeError(f"无法序列化类型 {type(o).__name__}")


def derive_title(messages: list[dict], max_chars: int = 20) -> str:
    """用第一条用户消息作为会话标题，过长则截断。"""
    for m in messages:
        if m.get("role") == "user":
            text = str(m.get("content", "") or "").strip().replace("\n", " ")
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            return text or "未命名对话"
    return "未命名对话"


class ConversationStore:
    """会话 JSON 文件读写；写操作原子落盘，避免中途崩溃损坏文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ------------------------------------------------------------------ 读
    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            return data
        except (json.JSONDecodeError, OSError):
            # 文件损坏或读取失败：返回空列表，不让界面崩溃（下次写盘会重建）
            return []

    def list_metas(self) -> list[dict]:
        """供侧边栏展示：按最近更新倒序，只返回轻量元信息。

        文件内天然保持"旧 → 新"追加序，反转即得"最近更新在前"，无需依赖时钟精度。
        """
        data = self._load()
        metas = [
            {
                "id": c.get("id"),
                "title": c.get("title") or "未命名对话",
                "updated_at": c.get("updated_at", ""),
            }
            for c in data
            if c.get("id")
        ]
        metas.reverse()
        return metas

    def get(self, conv_id: str) -> dict | None:
        for c in self._load():
            if c.get("id") == conv_id:
                return c
        return None

    # ------------------------------------------------------------------ 写
    def upsert(self, conversation: dict[str, Any]) -> None:
        """插入或更新一个会话（按 id 匹配）；保留首次 created_at。"""
        conv_id = conversation.get("id")
        if not conv_id:
            return
        data = self._load()
        existing = next((c for c in data if c.get("id") == conv_id), None)
        now = now_iso()
        updated: dict[str, Any] = {
            "id": conv_id,
            "title": conversation.get("title") or "未命名对话",
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "messages": conversation.get("messages", []),
            "context": conversation.get("context"),
        }
        data = [c for c in data if c.get("id") != conv_id]
        data.append(updated)  # 追加在末尾 → 文件保持"旧 → 新"顺序
        self._write(data)

    def delete(self, conv_id: str) -> None:
        data = [c for c in self._load() if c.get("id") != conv_id]
        self._write(data)

    def _write(self, data: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
