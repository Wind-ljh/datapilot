"""Embedding 客户端：SiliconFlow BGE-M3（免费）+ 离线 Mock。

Mock 向量使用字符 n-gram 哈希，同/近文本相似度可控，
支撑离线环境下检索链路的单测与演示。
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

from server.core.config import EmbedConfig, get_settings


class EmbeddingClient(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SiliconFlowEmbedder:
    """OpenAI 兼容 embeddings 接口（SiliconFlow / 智谱等均适用）。"""

    def __init__(self, config: EmbedConfig):
        self.config = config
        from openai import OpenAI

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.dim = 1024  # bge-m3

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.config.model, input=texts)
        vectors = [item.embedding for item in resp.data]
        self.dim = len(vectors[0]) if vectors else self.dim
        return vectors


class MockEmbedder:
    """确定性哈希向量：unigram+bigram 投射到固定维度并 L2 归一化。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        (value,) = struct.unpack("<I", digest[:4])
        return value % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            cleaned = "".join(text.split())
            grams = [cleaned[i] for i in range(len(cleaned))]
            grams += [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
            for g in grams:
                vec[self._bucket(g)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def build_embedder() -> EmbeddingClient:
    cfg = get_settings().embed
    if cfg.provider == "mock" or not cfg.api_key:
        return MockEmbedder()
    return SiliconFlowEmbedder(cfg)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))  # 向量已归一化时即余弦值
