"""mini 评测：40 道演示库金标题（data/mini_eval.jsonl），开箱即用。

与 CSpider 评估共用同一套指标定义（EX / 可执行率 / recall@k / 自修复成功率 / 延迟 / Token），
差别只在数据源：mini 评测跑在演示电商库（DuckDB）上，无需下载任何外部数据；
无 API Key 时以 mock 模式运行，用于验证评估链路与工程正确性。

用法：
  python -m eval.mini_runner --label baseline
  python -m eval.mini_runner --no-linking --label no-linking
  python -m eval.mini_runner --no-fewshot --label no-fewshot
  python -m eval.mini_runner --max-repair 0 --label no-repair
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.annotations import apply_corrections, load_corrections
from eval.cspider import extract_order_by, has_order_by, results_equal
from server.agents.pipeline import ChatBIPipeline
from server.core.config import get_settings
from server.core.database import DuckDBAdapter
from server.core.embeddings import build_embedder
from server.core.llm import build_llm
from server.mcp_tools.tools import ToolCore
from server.retrieval.fewshot import FewShotStore
from server.retrieval.linking import SchemaLinker

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_EVAL_PATH = REPO_ROOT / "data" / "mini_eval.jsonl"
FEWSHOT_PATH = REPO_ROOT / "data" / "fewshot_examples.jsonl"


def load_pairs(path: Path, limit: int = 0) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs[:limit] if limit else pairs


def run_mini_evaluation(
    label: str = "mini",
    limit: int = 0,
    llm_key: str = "llm",
    no_linking: bool = False,
    no_fewshot: bool = False,
    max_repair: int | None = None,
    db_path: str | None = None,
    out: str = "",
) -> dict:
    settings = get_settings()
    db_path = db_path or str(settings.db_path)
    adapter = DuckDBAdapter(path=db_path)
    embedder = build_embedder()
    llm = build_llm(llm_key)

    fewshot = (
        None if no_fewshot or not FEWSHOT_PATH.exists() else FewShotStore.from_jsonl(FEWSHOT_PATH, embedder)
    )
    pipeline = ChatBIPipeline(
        tools=ToolCore(adapter),
        llm=llm,
        linker=None if no_linking else SchemaLinker(adapter, embedder),
        fewshot=fewshot,
        max_repair_rounds=max_repair,
        dialect="DuckDB",
    )

    corrections = load_corrections()
    eval_pairs = apply_corrections(load_pairs(MINI_EVAL_PATH, limit), corrections)
    n = ex_hits = exec_ok = repair_used = repair_success = 0
    latencies: list[float] = []
    details: list[dict] = []

    for i, pair in enumerate(eval_pairs):
        start = time.perf_counter()
        state = pipeline.run(pair["question"])
        wall_ms = (time.perf_counter() - start) * 1000
        latencies.append(wall_ms)

        executed = bool(state.get("executed"))
        hit = 0
        if executed:
            try:
                gold = adapter.execute_sql(pair["sql"], max_rows=5000)
                pred_sql = state.get("sql", "")
                ordered = has_order_by(pair["sql"]) and has_order_by(pred_sql)
                gold_keys = extract_order_by(pair["sql"]) if ordered else None
                pred_keys = extract_order_by(pred_sql) if ordered else None
                hit = int(
                    results_equal(
                        [list(r) for r in state["result"]["rows"]],
                        [list(r) for r in gold.rows],
                        ordered=ordered,
                        order_keys_a=pred_keys,
                        order_keys_b=gold_keys,
                    )
                )
            except Exception:
                hit = 0
        n += 1
        ex_hits += hit
        exec_ok += int(executed)
        if state.get("repair_round", 0) > 0:
            repair_used += 1
            repair_success += int(executed)
        corr = pair.get("_correction")
        details.append(
            {
                "i": i,
                "question": pair["question"],
                "gold_sql": pair["sql"],
                "pred_sql": state.get("sql", ""),
                "executed": executed,
                "ex": hit,
                "repair_round": state.get("repair_round", 0),
                "degraded": state.get("degraded", False),
                "wall_ms": round(wall_ms, 1),
                "tokens": state.get("prompt_tokens", 0) + state.get("completion_tokens", 0),
                "gold_corrected": bool(corr),
                "correction_note": corr["issue"] if corr else "",
            }
        )

    latencies.sort()
    report = {
        "label": label,
        "dataset": "mini_eval",
        "db": db_path,
        "n": n,
        "ex": round(ex_hits / n, 4) if n else 0.0,
        "exec_rate": round(exec_ok / n, 4) if n else 0.0,
        "repair_success_rate": round(repair_success / repair_used, 4) if repair_used else None,
        "repair_used": repair_used,
        "latency_p50_ms": round(latencies[n // 2], 1) if latencies else 0.0,
        "latency_p95_ms": round(latencies[int(n * 0.95)], 1) if latencies else 0.0,
        "avg_tokens": round(
            sum(d["tokens"] for d in details) / n, 1
        ) if n else 0.0,
        "config": {
            "llm": llm_key,
            "no_linking": no_linking,
            "no_fewshot": no_fewshot,
            "max_repair": max_repair,
            "mock": settings.mock,
        },
        "details": details,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        f"\n[{label}] n={n} EX={report['ex']:.3f} exec_rate={report['exec_rate']:.3f} "
        f"repair_ok={report['repair_success_rate']} p50={report['latency_p50_ms']:.0f}ms "
        f"p95={report['latency_p95_ms']:.0f}ms avg_tokens={report['avg_tokens']:.0f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="mini 评测（演示库 40 题）")
    parser.add_argument("--label", default="mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--llm", default="llm", choices=["llm", "llm2"])
    parser.add_argument("--no-linking", action="store_true")
    parser.add_argument("--no-fewshot", action="store_true")
    parser.add_argument("--max-repair", type=int, default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_mini_evaluation(
        label=args.label,
        limit=args.limit,
        llm_key=args.llm,
        no_linking=args.no_linking,
        no_fewshot=args.no_fewshot,
        max_repair=args.max_repair,
        db_path=args.db,
        out=args.out or f"eval/results/{args.label}.json",
    )


if __name__ == "__main__":
    main()
