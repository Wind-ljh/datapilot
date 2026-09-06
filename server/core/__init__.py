"""核心基础设施：配置 / LLM 客户端 / Embedding / DuckDB 数据层 / 种子数据."""

from server.core.config import Settings, get_settings
from server.core.embeddings import (
    EmbeddingClient,
    MockEmbedder,
    SiliconFlowEmbedder,
    build_embedder,
)
from server.core.llm import LLMClient, LLMResponse, MockLLMClient, build_llm

__all__ = [
    "Settings",
    "get_settings",
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "build_llm",
    "EmbeddingClient",
    "MockEmbedder",
    "SiliconFlowEmbedder",
    "build_embedder",
]
