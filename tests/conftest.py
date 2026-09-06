"""测试共享夹具：mock 模式 + 内存演示库 + 预装配流水线。"""

from __future__ import annotations

import os
from pathlib import Path

# 必须在导入任何 server.* 之前设置，确保全程离线
os.environ["DATAPILOT_MOCK"] = "1"

import pytest

from server.core.database import DuckDBAdapter
from server.core.embeddings import build_embedder
from server.core.llm import build_llm
from server.core.seed_data import generate
from server.mcp_tools.tools import ToolCore
from server.retrieval.fewshot import FewShotStore
from server.retrieval.linking import SchemaLinker

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("db") / "ecommerce.duckdb"
    generate(str(path), n_orders=1200, seed=7)
    return str(path)


@pytest.fixture(scope="session")
def adapter(db_path) -> DuckDBAdapter:
    return DuckDBAdapter(path=db_path)


@pytest.fixture(scope="session")
def tools(adapter) -> ToolCore:
    return ToolCore(adapter)


@pytest.fixture(scope="session")
def fewshot() -> FewShotStore:
    embedder = build_embedder()
    return FewShotStore.from_jsonl(REPO_ROOT / "data" / "fewshot_examples.jsonl", embedder)


@pytest.fixture(scope="session")
def pipeline(adapter, tools, fewshot):
    from server.agents.pipeline import ChatBIPipeline

    return ChatBIPipeline(
        tools=tools,
        llm=build_llm("llm"),
        linker=SchemaLinker(adapter, build_embedder()),
        fewshot=fewshot,
        dialect="DuckDB",
    )
