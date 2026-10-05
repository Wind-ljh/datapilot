"""CSpider 评估 runner：执行准确率 + 消融实验编排。

用法示例：
  # 基线（全量能力）
  python -m eval.runner --split dev --limit 300 --label baseline
  # 消融：去掉 schema linking / few-shot / 自修复
  python -m eval.runner --split dev --limit 300 --no-linking --label no-linking
  python -m eval.runner --split dev --limit 300 --no-fewshot  --label no-fewshot
  python -m eval.runner --split dev --limit 300 --max-repair 0 --label no-repair
  # 对照模型
  python -m eval.runner --split dev --limit 300 --llm llm2 --label qwen7b
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval.cspider import CSpiderExample, extract_order_by, has_order_by, load_dataset, results_equal
from server.agents.pipeline import ChatBIPipeline
from server.core.database import SQLiteAdapter
from server.core.embeddings import EmbeddingClient, build_embedder
from server.core.llm import build_llm
from server.mcp_tools.tools import ToolCore
from server.retrieval.fewshot import FewShotStore, QSLExample
from server.retrieval.linking import SchemaLinker


@dataclass
class EvalReport:
    label: str
    n: int
    ex: float  # 执行准确率
    exec_rate: float  # 预测 SQL 可执行率
    table_recall_at_k: float
    repair_success_rate: float  # 首次失败后自修复成功率
    avg_prompt_tokens: float
    avg_completion_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    details: list[dict] = field(default_factory=list)


def _db_path(data_dir: Path, db_id: str) -> Path:
    return data_dir / "database" / db_id / f"{db_id}.sqlite"


def _build_fewshot_store(
    train_examples: list[CSpiderExample], db_id: str, embedder: EmbeddingClient
) -> FewShotStore | None:
    """同库 train 示例做 few-shot 池（跨库示例会引入不存在的表，反而有害）。"""
    same_db = [ex for ex in train_examples if ex.db_id == db_id]
    if not same_db:
        return None
    pairs = [QSLExample(question=ex.question, sql=ex.gold_sql, tables=ex.gold_tables) for ex in same_db]
    return FewShotStore(pairs, embedder)


def run_evaluation(
    data_dir: str | Path = "data/cspider",
    split: str = "dev",
    limit: int = 0,
    db_filter: str = "",
    llm_key: str = "llm",
    no_linking: bool = False,
    no_fewshot: bool = False,
    max_repair: int = 3,
    label: str = "exp",
    top_k_tables: int = 3,
    out: str = "",
) -> EvalReport:
    data_dir = Path(data_dir)
    tables_path = data_dir / "tables.json"
    split_path = data_dir / f"{split}.json"
    train_path = data_dir / "train_spider.json"

    examples = load_dataset(split_path, tables_path, limit=limit)
    if db_filter:
        examples = [e for e in examples if e.db_id == db_filter]
    train_examples = load_dataset(train_path, tables_path) if train_path.exists() else []
    embedder = build_embedder()
    llm = build_llm(llm_key)

    # 按库缓存流水线（每个 db_id 一套 adapter/linker/fewshot）
    pipelines: dict[str, ChatBIPipeline] = {}

    def get_pipeline(db_id: str) -> ChatBIPipeline:
        if db_id in pipelines:
            return pipelines[db_id]
        adapter = SQLiteAdapter(str(_db_path(data_dir, db_id)))
        store = None if no_fewshot else _build_fewshot_store(train_examples, db_id, embedder)
        pipe = ChatBIPipeline(
            tools=ToolCore(adapter),
            llm=llm,
            linker=None if no_linking else SchemaLinker(adapter, embedder),
            fewshot=store,
            max_repair_rounds=max_repair,
            dialect="SQLite",
        )
        pipelines[db_id] = pipe
        return pipe

    n_ex = exec_ok = recall_hits = repair_used = repair_success = 0
    latencies: list[float] = []
    total_prompt = total_completion = 0
    details: list[dict] = []

    for i, ex in enumerate(examples):
        pipe = get_pipeline(ex.db_id)
        start = time.perf_counter()
        try:
            state = pipe.run(ex.question)
        except Exception as e:  # 单条失败不中断评估
            details.append({"i": i, "db_id": ex.db_id, "question": ex.question, "error": str(e), "ex": 0})
            continue
        wall_ms = (time.perf_counter() - start) * 1000
        latencies.append(wall_ms)
        total_prompt += state.get("prompt_tokens", 0)
        total_completion += state.get("completion_tokens", 0)

        # 表召回
        recall = False
        if ex.gold_tables:
            linked = {t for t in state.get("linked_tables", [])}
            recall = set(ex.gold_tables).issubset(linked)
            recall_hits += int(recall)

        # EX 判定：预测与 gold 各自在同一库上执行后比较结果集
        hit = 0
        executed = bool(state.get("executed"))
        if executed:
            adapter = SQLiteAdapter(str(_db_path(data_dir, ex.db_id)))
            try:
                gold_rows = [list(r) for r in adapter.execute_sql(ex.gold_sql, max_rows=5000).rows]
                pred_rows = [list(r) for r in state["result"]["rows"]]
                pred_sql = state.get("sql", "")
                ordered = has_order_by(ex.gold_sql) and has_order_by(pred_sql)
                gold_keys = extract_order_by(ex.gold_sql) if ordered else None
                pred_keys = extract_order_by(pred_sql) if ordered else None
                hit = int(
                    results_equal(
                        pred_rows, gold_rows, ordered=ordered,
                        order_keys_a=pred_keys, order_keys_b=gold_keys,
                    )
                )
            except Exception:
                hit = 0
            finally:
                adapter.close()
        n_ex += 1
        exec_ok += int(executed)
        ex_score = hit
        if state.get("repair_round", 0) > 0:
            repair_used += 1
            repair_success += int(executed)

        details.append(
            {
                "i": i,
                "db_id": ex.db_id,
                "question": ex.question,
                "gold_sql": ex.gold_sql,
                "pred_sql": state.get("sql", ""),
                "executed": executed,
                "repair_round": state.get("repair_round", 0),
                "ex": ex_score,
                "table_recall": recall,
                "wall_ms": round(wall_ms, 1),
                "tokens": {
                    "prompt": state.get("prompt_tokens", 0),
                    "completion": state.get("completion_tokens", 0),
                },
            }
        )
        if (i + 1) % 20 == 0:
            running_ex = sum(d.get("ex", 0) for d in details) / len(details)
            print(f"  [{i + 1}/{len(examples)}] running EX={running_ex:.3f}")

    latencies.sort()
    report = EvalReport(
        label=label,
        n=n_ex,
        ex=round(sum(d.get("ex", 0) for d in details) / n_ex, 4) if n_ex else 0.0,
        exec_rate=round(exec_ok / n_ex, 4) if n_ex else 0.0,
        table_recall_at_k=round(recall_hits / n_ex, 4) if n_ex else 0.0,
        repair_success_rate=round(repair_success / repair_used, 4) if repair_used else 0.0,
        avg_prompt_tokens=round(total_prompt / n_ex, 1) if n_ex else 0.0,
        avg_completion_tokens=round(total_completion / n_ex, 1) if n_ex else 0.0,
        latency_p50_ms=round(latencies[len(latencies) // 2], 1) if latencies else 0.0,
        latency_p95_ms=round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0.0,
        details=details,
    )

    payload = asdict(report)
    payload["config"] = {
        "split": split,
        "limit": limit,
        "db_filter": db_filter,
        "llm": llm_key,
        "no_linking": no_linking,
        "no_fewshot": no_fewshot,
        "max_repair": max_repair,
        "top_k_tables": top_k_tables,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"\n[{label}] n={report.n} EX={report.ex:.3f} exec_rate={report.exec_rate:.3f} "
        f"recall@k={report.table_recall_at_k:.3f} repair_ok={report.repair_success_rate:.3f} "
        f"p50={report.latency_p50_ms:.0f}ms p95={report.latency_p95_ms:.0f}ms "
        f"avg_tokens={report.avg_prompt_tokens + report.avg_completion_tokens:.0f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CSpider 评估 / 消融实验")
    parser.add_argument("--data-dir", default="data/cspider")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0, help="评估条数（0=全部）")
    parser.add_argument("--db-filter", default="")
    parser.add_argument("--llm", default="llm", choices=["llm", "llm2"])
    parser.add_argument("--no-linking", action="store_true")
    parser.add_argument("--no-fewshot", action="store_true")
    parser.add_argument("--max-repair", type=int, default=3)
    parser.add_argument("--label", default="exp")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        split=args.split,
        limit=args.limit,
        db_filter=args.db_filter,
        llm_key=args.llm,
        no_linking=args.no_linking,
        no_fewshot=args.no_fewshot,
        max_repair=args.max_repair,
        label=args.label,
        out=args.out or f"eval/results/{args.label}.json",
    )


if __name__ == "__main__":
    main()
