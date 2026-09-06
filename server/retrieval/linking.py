"""Schema Linking：为自然语言问题挑选相关表与列（ChatBI 的非常规 RAG）。

区别于"文档问答 RAG"：这里检索的对象是 schema 元数据（表注释/列注释/类型），
目的是把百级字段压缩成只含相关表列的"迷你 schema"，降低生成幻觉率。
评估指标：表级 recall@k（相对 gold 表集合）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from server.core.database import DuckDBAdapter, TableSchema
from server.core.embeddings import EmbeddingClient
from server.retrieval.hybrid import Doc, HybridRetriever


@dataclass
class LinkedSchema:
    tables: list[TableSchema] = field(default_factory=list)
    hit_ids: list[str] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        return "\n\n".join(t.to_ddl_text() for t in self.tables)


class SchemaLinker:
    """用混合检索做表/列召回，再拼装 LinkedSchema。"""

    def __init__(self, adapter: DuckDBAdapter, embedder: EmbeddingClient):
        self.adapter = adapter
        self._schemas = {s.name: s for s in adapter.get_schema()}
        self._retriever = HybridRetriever(self._build_docs(), embedder)

    def _build_docs(self) -> list[Doc]:
        docs: list[Doc] = []
        for s in self._schemas.values():
            cols = "；".join(
                f"{c.name}({c.dtype}){('：' + c.comment) if c.comment else ''}" for c in s.columns
            )
            docs.append(
                Doc(
                    id=f"table:{s.name}",
                    text=f"表 {s.name} {s.comment} 包含字段 {cols}",
                    metadata={"kind": "table", "table": s.name},
                )
            )
            for c in s.columns:
                docs.append(
                    Doc(
                        id=f"column:{s.name}.{c.name}",
                        text=f"{s.name} 表的 {c.name} 字段 {c.comment} 类型 {c.dtype} 表注释 {s.comment}",
                        metadata={"kind": "column", "table": s.name, "column": c.name},
                    )
                )
        return docs

    def link(self, question: str, top_k_tables: int = 3) -> LinkedSchema:
        """返回与问题最相关的表（含全部列）。"""
        hits = self._retriever.search(question, top_k=top_k_tables * 3)
        # 表级得分聚合：表文档直接命中 + 其列文档命中折半计票
        table_scores: dict[str, float] = {}
        sources: dict[str, dict] = {}
        for hit in hits:
            meta = hit.doc.metadata
            table = meta["table"]
            weight = 1.0 if meta["kind"] == "table" else 0.5
            table_scores[table] = table_scores.get(table, 0.0) + weight
            sources[table] = {**sources.get(table, {}), **hit.sources}
        ranked = sorted(table_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k_tables]
        return LinkedSchema(
            tables=[self._schemas[t] for t, _ in ranked],
            hit_ids=[h.doc.id for h in hits],
            sources={t: sources.get(t, {}) for t, _ in ranked},
        )

    def table_recall(self, question: str, gold_tables: list[str], k: int = 3) -> bool:
        """表级 recall@k：gold 表是否全部（或任一，取交集视角）被召回。

        这里取"全部 gold 表都出现在 top-k"的严格定义，供 eval 汇总。
        """
        linked = self.link(question, top_k_tables=max(k, len(gold_tables)))
        got = {t.name for t in linked.tables}
        return set(gold_tables).issubset(got)
