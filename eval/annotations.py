"""mini 评测的 gold 注解修正：独立 override 机制。

原始 benchmark（data/mini_eval.jsonl）一律不改；已确认的「gold 口径与题目不一致」
问题（如退款过滤）在此集中覆盖，保证修正可追溯、可复现。详见 docs/evaluation_notes.md。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_PATH = REPO_ROOT / "data" / "eval_corrections.json"


def load_corrections(path: str | Path = CORRECTIONS_PATH) -> dict[int, dict]:
    """加载修正表：{index: correction}。index 对应 mini_eval.jsonl 的 0-based 行号。"""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["index"]: c for c in data.get("corrections", [])}


def apply_corrections(
    pairs: list[dict], corrections: dict[int, dict] | None = None
) -> list[dict]:
    """返回应用修正后的副本（不修改原 pairs），并把 gold SQL 替换为修正版本。"""
    if corrections is None:
        corrections = load_corrections()
    out: list[dict] = []
    for i, pair in enumerate(pairs):
        p = dict(pair)
        corr = corrections.get(i)
        if corr:
            p["sql"] = corr["gold_sql_override"]
            p["_correction"] = corr
        out.append(p)
    return out
