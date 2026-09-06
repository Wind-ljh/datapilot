"""混合检索：BM25（稀疏）+ 向量（稠密）→ RRF 融合。

面试讲点：为什么 ChatBI 要混合检索——
- 表名/字段名是强精确匹配信号，BM25 对术语命中极敏感（用户说"退款"要命中 status='已退款' 的注释）；
- 向量负责同义改写（"营业额"↔"收入/amount"）；
- 两路召回用 Reciprocal Rank Fusion 融合，无需调权重尺度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi

from server.core.embeddings import EmbeddingClient, cosine

jieba.setLogLevel(60)  # 关闭 jieba 初始化日志

_PUNCT = re.compile(r"[^\w]+", re.UNICODE)  # 中文汉字属 \w，此模式剥离全部中英文标点与空白


def tokenize_zh(text: str) -> list[str]:
    """中文分词 + 小写 + 去标点，同时保留整串（提升表名/字段名整词命中）。"""
    cleaned = _PUNCT.sub(" ", text.lower())
    tokens = [t for t in jieba.lcut(cleaned) if t.strip()]
    if cleaned.strip():
        tokens.append(cleaned.strip())
    return tokens


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    sources: dict  # 例如 {"bm25": 1, "vector": 3}，便于调试两路召回情况


class HybridRetriever:
    """内存级混合检索器：BM25 + 向量 + RRF。千级文档规模下无需独立向量库。"""

    RRF_K = 60  # 经典取值，抑制单路 rank=1 的过度主导

    def __init__(self, docs: list[Doc], embedder: EmbeddingClient):
        self.docs = docs
        self._embedder = embedder
        self._tokenized = [tokenize_zh(d.text) for d in docs]
        self._bm25 = BM25Okapi(self._tokenized) if docs else None
        self._vectors = embedder.embed([d.text for d in docs]) if docs else []

    def _bm25_rank(self, query: str) -> dict[str, int]:
        if not self._bm25:
            return {}
        scores = self._bm25.get_scores(tokenize_zh(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {self.docs[i].id: rank for rank, i in enumerate(order) if scores[i] > 0}

    def _vector_rank(self, query: str) -> dict[str, int]:
        if not self._vectors:
            return {}
        qv = self._embedder.embed([query])[0]
        scores = [cosine(qv, v) for v in self._vectors]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {self.docs[i].id: rank for rank, i in enumerate(order) if scores[i] > 0.05}

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """RRF 融合两路召回，返回 top_k。"""
        bm = self._bm25_rank(query)
        ve = self._vector_rank(query)
        fused: dict[str, float] = {}
        sources: dict[str, dict] = {}
        for doc_id, rank in bm.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            sources.setdefault(doc_id, {})["bm25"] = rank + 1
        for doc_id, rank in ve.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            sources.setdefault(doc_id, {})["vector"] = rank + 1
        doc_map = {d.id: d for d in self.docs}
        hits = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Hit(doc=doc_map[doc_id], score=score, sources=sources[doc_id]) for doc_id, score in hits]
