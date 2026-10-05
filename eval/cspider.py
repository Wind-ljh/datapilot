"""CSpider 数据加载与执行准确率（EX）判定。

EX 定义（对齐官方 Spider 评测思路）：预测 SQL 与 gold SQL 在同一数据库上执行，
结果集一致则记 1 分；含 ORDER BY 时按有序序列比较，否则按多重集合比较。
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CSpiderExample:
    db_id: str
    question: str
    gold_sql: str
    gold_tables: list[str]  # gold 涉及的表（tables.json 解析，供 recall@k 用）


def _extract_tables(gold_sql: str, all_tables: list[str]) -> list[str]:
    """从 gold SQL 中粗提取涉及的表名（FROM/JOIN 后的标识符）。"""
    import re

    pattern = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", re.IGNORECASE)
    found = {m.group(1) for m in pattern.finditer(gold_sql)}
    return sorted(t for t in found if t in set(all_tables))


def load_dataset(split_path: str | Path, tables_path: str | Path, limit: int = 0) -> list[CSpiderExample]:
    """加载 CSpider dev/train json 与 tables.json（schema 元数据）。"""
    with open(tables_path, encoding="utf-8") as f:
        table_meta: dict[str, list[str]] = {}
        for obj in json.load(f):
            table_meta[obj["db_id"]] = [t for t in obj.get("table_names_original", [])]

    examples: list[CSpiderExample] = []
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    for obj in data:
        db_id = obj["db_id"]
        examples.append(
            CSpiderExample(
                db_id=db_id,
                question=obj["question"],
                gold_sql=obj["query"],
                gold_tables=_extract_tables(obj["query"], table_meta.get(db_id, [])),
            )
        )
    return examples


def _norm_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    # DuckDB 的 DATE()/TIMESTAMP 返回 date/datetime 对象，而 strftime() 返回字符串；
    # 二者语义等价，统一转 ISO 字符串再比较，避免「相同日期不同表示」被误判为不等。
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v


# ---- ORDER BY 解析（用于 tie-aware 有序比较） ----

_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_ORDER_TAIL_STOP_RE = re.compile(r"\bLIMIT\b|\bOFFSET\b|;", re.IGNORECASE)
_ALIAS_RE = re.compile(r"\bAS\s+(\w+)", re.IGNORECASE)
_DESC_RE = re.compile(r"\s+(ASC|DESC)\s*$", re.IGNORECASE)
_NULLS_RE = re.compile(r"\s+NULLS\s+(FIRST|LAST)\s*$", re.IGNORECASE)
_COL_REF_RE = re.compile(r"^(?:\w+\.)?(\w+)$")


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """按 sep 切分，忽略括号与引号内部的 sep。"""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    cur: list[str] = []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            cur.append(ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _output_candidates(expr: str) -> list[str]:
    """SELECT 单项的可匹配名：优先别名，其次裸列名（col 或 tbl.col）。"""
    expr = expr.strip()
    candidates: list[str] = []
    am = _ALIAS_RE.search(expr)
    if am:
        candidates.append(am.group(1))
        left = expr[: am.start()].strip()
    else:
        left = expr
    cm = _COL_REF_RE.match(left)
    if cm:
        candidates.append(cm.group(1))
    return candidates


def _top_level_tokens(sql: str) -> list[tuple[int, str]]:
    """返回括号深度 0 的词法 token（位置, 文本），跳过引号字符串与括号内部内容。

    用「括号深度」而非正则定位关键字，避免把 EXTRACT(year FROM ...) 里的 FROM
    误当作 SELECT→FROM 边界。
    """
    tokens: list[tuple[int, str]] = []
    depth = 0
    quote: str | None = None
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0 and (ch.isalnum() or ch == "_"):
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            tokens.append((i, sql[i:j]))
            i = j
            continue
        i += 1
    return tokens


def _select_output_candidates(sql: str) -> list[list[str]]:
    """返回 SELECT 各输出列的可匹配名列表（与输出位置一一对应）。"""
    m = _SELECT_RE.search(sql)
    if not m:
        return []
    start = m.end()
    from_pos = next(
        (pos for pos, tok in _top_level_tokens(sql) if tok.upper() == "FROM" and pos > start),
        len(sql),
    )
    select_part = sql[start:from_pos]
    return [_output_candidates(expr) for expr in _split_top_level(select_part)]


def extract_order_by(sql: str) -> list[int] | None:
    """解析 ORDER BY 排序键，映射为输出列下标（0-based）。

    返回 None 表示无法可靠解析（调用方应回退到严格的有序比较）；
    无 ORDER BY 时返回 []。
    """
    m = _ORDER_RE.search(sql)
    if not m:
        return []
    tail = _ORDER_TAIL_STOP_RE.split(sql[m.end() :], maxsplit=1)[0]
    candidates = _select_output_candidates(sql)
    keys: list[int] = []
    for term in _split_top_level(tail):
        term = _NULLS_RE.sub("", term).strip()
        term = _DESC_RE.sub("", term).strip()
        if not term:
            continue
        if term.isdigit():
            keys.append(int(term) - 1)
            continue
        idx = next((j for j, names in enumerate(candidates) if term in names), None)
        if idx is None:
            return None
        keys.append(idx)
    return keys


def results_equal(
    rows_a: list[list],
    rows_b: list[list],
    ordered: bool,
    order_keys_a: list[int] | None = None,
    order_keys_b: list[int] | None = None,
) -> bool:
    """结果集一致性比较。

    - ordered=False：多重集合比较（行序无关）；
    - ordered=True 且未提供（可对齐的）排序键：严格逐行比较（旧语义）；
    - ordered=True 且排序键已知且一致：多重集合相等 + 排序键序列相等，
      即只允许「并列组内换序」，仍能识别真正的排序错误。
    """
    a = [[_norm_value(v) for v in row] for row in rows_a]
    b = [[_norm_value(v) for v in row] for row in rows_b]
    if not ordered:
        return sorted(map(repr, a)) == sorted(map(repr, b))
    if order_keys_a is None or order_keys_b is None or order_keys_a != order_keys_b:
        return a == b
    if sorted(map(repr, a)) != sorted(map(repr, b)):
        return False
    width = len(a[0]) if a else 0
    if any(k < 0 or k >= width for k in order_keys_a):
        return a == b
    key_a = [tuple(row[k] for k in order_keys_a) for row in a]
    key_b = [tuple(row[k] for k in order_keys_b) for row in b]
    return key_a == key_b


def has_order_by(sql: str) -> bool:
    return _ORDER_RE.search(sql) is not None
