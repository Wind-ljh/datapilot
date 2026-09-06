"""SFT 数据构造：CSpider train → ChatBI 格式对话数据（LLaMA-Factory sharegpt 兼容）。

设计要点（面试讲点：训练数据质量 > 训练技巧）：
1. 输入侧注入 schema linking 的产物结构（表+列+中文注释），与线上推理 prompt 分布对齐，
   避免"训练看全量 schema、推理看迷你 schema"的分布漂移；
2. 输出仅含 SQL 代码块，与线上 extract_sql 的解析格式一致；
3. 按 db_id 划分 train/val，杜绝同库泄漏（比随机划分更严格）；
4. 过滤：gold SQL 含写操作/多条语句/超长的一律剔除。

用法：
  python -m training.build_sft_data --data-dir data/cspider --out-dir training/data --val-dbs 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from server.core.database import SQLSafetyError, assert_readonly_sql

SYSTEM_PROMPT = """你是资深数据分析师，把中文分析问题转成单条只读 SQL（方言：SQLite）。
规则：
1. 只输出一条 SQL，放在 ```sql 代码块中，禁止任何解释文字；
2. 只能使用给定 schema 中的表与列；
3. 善用列注释理解业务词；
4. 聚合结果请起中文别名；TopN 类问题默认 LIMIT 10；
5. 禁止写操作。"""


def _schema_text_from_tables(tables_obj: dict) -> str:
    """把 tables.json 中一个库的 schema 渲染成与线上 SchemaLinker 相同风格的文本。"""
    lines = []
    for i, table in enumerate(tables_obj["table_names"]):
        original = tables_obj["table_names_original"][i]
        col_lines = []
        for j, (col_name, col_type) in enumerate(tables_obj["column_names_original"]):
            if col_name == "*":
                continue
            if tables_obj["column_names"][j][0] == i:
                comment = tables_obj["column_names"][j][1]
                col_lines.append(f"  {col_name} {col_type} -- {comment}")
        lines.append(f"TABLE {original} -- {table}\n" + "\n".join(col_lines))
    return "\n\n".join(lines)


def build(
    data_dir: Path,
    out_dir: Path,
    val_dbs: int = 20,
    seed: int = 42,
    max_samples: int = 0,
) -> dict:
    with open(data_dir / "tables.json", encoding="utf-8") as f:
        tables_meta = {obj["db_id"]: obj for obj in json.load(f)}
    with open(data_dir / "train_spider.json", encoding="utf-8") as f:
        train = json.load(f)

    rng = random.Random(seed)
    db_ids = sorted(tables_meta.keys())
    rng.shuffle(db_ids)
    val_db_set = set(db_ids[:val_dbs])

    n_written = n_filtered = 0
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for obj in train:
        sql = obj["query"].strip()
        try:
            assert_readonly_sql(sql)
        except SQLSafetyError:
            n_filtered += 1
            continue
        if len(sql) > 600 or len(obj["question"]) > 120:
            n_filtered += 1
            continue
        tables_obj = tables_meta.get(obj["db_id"])
        if tables_obj is None:
            n_filtered += 1
            continue
        row = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {
                    "from": "human",
                    "value": f"数据库 schema：\n{_schema_text_from_tables(tables_obj)}\n\n用户问题：{obj['question']}",
                },
                {"from": "gpt", "value": f"```sql\n{sql}\n```"},
            ],
            "db_id": obj["db_id"],
        }
        (val_rows if obj["db_id"] in val_db_set else train_rows).append(row)
        n_written += 1
        if max_samples and n_written >= max_samples:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "written": n_written,
        "filtered": n_filtered,
        "train": len(train_rows),
        "val": len(val_rows),
        "val_dbs": sorted(val_db_set),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 ChatBI SFT 数据")
    parser.add_argument("--data-dir", default="data/cspider")
    parser.add_argument("--out-dir", default="training/data")
    parser.add_argument("--val-dbs", type=int, default=20, help="留出多少个库做验证（按库划分防泄漏）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    build(Path(args.data_dir), Path(args.out_dir), val_dbs=args.val_dbs, seed=args.seed, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
