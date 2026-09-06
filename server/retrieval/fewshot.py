"""Few-shot 示例检索：相似"问题-SQL"对召回，注入生成 prompt。

面试讲点：这是 RAG 在 ChatBI 的第二处应用——
检索的不是知识文档，而是"相似问题的参考 SQL"，
对齐输出风格与 JOIN 惯例，是 zero-shot 到 few-shot 提分的核心手段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from server.core.embeddings import EmbeddingClient
from server.retrieval.hybrid import Doc, HybridRetriever


@dataclass
class QSLExample:
    question: str
    sql: str
    tables: list[str]


class FewShotStore:
    """问题-SQL 对的混合检索库。"""

    def __init__(self, examples: list[QSLExample], embedder: EmbeddingClient):
        self.examples = examples
        docs = [Doc(id=str(i), text=ex.question, metadata={"index": i}) for i, ex in enumerate(examples)]
        self._retriever = HybridRetriever(docs, embedder)

    def top_k(self, question: str, k: int = 3) -> list[QSLExample]:
        hits = self._retriever.search(question, top_k=k)
        return [self.examples[int(h.doc.id)] for h in hits]

    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: EmbeddingClient) -> FewShotStore:
        examples: list[QSLExample] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append(
                    QSLExample(question=obj["question"], sql=obj["sql"], tables=obj.get("tables", []))
                )
        return cls(examples, embedder)
