"""CSpider 数据加载与执行准确率（EX）判定。

EX 定义（对齐官方 Spider 评测思路）：预测 SQL 与 gold SQL 在同一数据库上执行，
结果集一致则记 1 分；含 ORDER BY 时按有序序列比较，否则按多重集合比较。
"""

from __future__ import annotations

import json
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
    return v


def results_equal(rows_a: list[list], rows_b: list[list], ordered: bool) -> bool:
    """结果集一致性比较。"""
    a = [[_norm_value(v) for v in row] for row in rows_a]
    b = [[_norm_value(v) for v in row] for row in rows_b]
    if ordered:
        return a == b
    return sorted(map(repr, a)) == sorted(map(repr, b))


def has_order_by(sql: str) -> bool:
    import re

    return re.search(r"\bORDER\s+BY\b", sql, re.IGNORECASE) is not None
