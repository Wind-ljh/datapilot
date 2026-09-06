"""检索层测试：分词 / 混合检索 / schema linking / few-shot。"""

from __future__ import annotations

from server.core.embeddings import MockEmbedder
from server.retrieval.hybrid import Doc, HybridRetriever, tokenize_zh


def test_tokenize_zh_basic():
    tokens = tokenize_zh("每月的订单量和收入趋势！")
    assert any("订单" in t for t in tokens)
    assert "！" not in tokens


def test_schema_linker_finds_orders(pipeline):
    linked = pipeline.linker.link("每月的退款订单数变化", top_k_tables=3)
    names = [t.name for t in linked.tables]
    assert "orders" in names


def test_schema_linker_finds_products_join(pipeline):
    linked = pipeline.linker.link("销量最高的10个商品", top_k_tables=3)
    names = [t.name for t in linked.tables]
    assert "products" in names


def test_fewshot_top_k(fewshot):
    hits = fewshot.top_k("每个月的订单量是多少", k=3)
    assert len(hits) == 3
    assert all("SELECT" in ex.sql.upper() for ex in hits)


def test_hybrid_retriever_rrf_order():
    docs = [Doc(id=str(i), text=t) for i, t in enumerate(["订单数量统计", "商品库存管理", "用户注册信息"])]
    retriever = HybridRetriever(docs, MockEmbedder())
    hits = retriever.search("订单数量", top_k=2)
    assert hits[0].doc.text == "订单数量统计"
    assert len(hits) == 2
