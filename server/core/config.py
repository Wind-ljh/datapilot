"""全局配置：全部来自环境变量（见 .env.example），无副作用、可测试。

设计要点：所有厂商接口均为 OpenAI 兼容协议，切换厂商只改环境变量，
代码里不出现任何厂商 SDK —— 这是项目"零厂商绑定"的技术决策。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 读取工作目录下的 .env（存在才生效）


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """单个 LLM 端点配置。"""

    provider: str  # zhipu | siliconflow | mock
    api_key: str
    base_url: str
    model: str

    @property
    def usable(self) -> bool:
        """mock 模式或 provider=mock 时不需要真实 key。"""
        return self.provider == "mock" or bool(self.api_key)


@dataclass(frozen=True)
class EmbedConfig:
    provider: str  # siliconflow | mock
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    llm2: LLMConfig  # 对照实验用的第二个端点（消融实验切换）
    embed: EmbedConfig
    db_path: Path
    retrieval_top_k: int
    max_repair_rounds: int
    mock: bool


def load_settings() -> Settings:
    mock = _env_bool("DATAPILOT_MOCK", False)

    llm = LLMConfig(
        provider=_env("DATAPILOT_LLM_PROVIDER", "zhipu"),
        api_key=_env("DATAPILOT_LLM_API_KEY"),
        base_url=_env("DATAPILOT_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        model=_env("DATAPILOT_LLM_MODEL", "glm-4.7-flash"),
    )
    llm2 = LLMConfig(
        provider=_env("DATAPILOT_LLM2_PROVIDER", "siliconflow"),
        api_key=_env("DATAPILOT_LLM2_API_KEY"),
        base_url=_env("DATAPILOT_LLM2_BASE_URL", "https://api.siliconflow.cn/v1"),
        model=_env("DATAPILOT_LLM2_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    embed = EmbedConfig(
        provider="mock" if mock else _env("DATAPILOT_EMBED_PROVIDER", "siliconflow"),
        api_key=_env("DATAPILOT_EMBED_API_KEY"),
        base_url=_env("DATAPILOT_EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
        model=_env("DATAPILOT_EMBED_MODEL", "BAAI/bge-m3"),
    )
    if mock:
        llm = LLMConfig(provider="mock", api_key="", base_url="", model="mock")
        llm2 = LLMConfig(provider="mock", api_key="", base_url="", model="mock")

    return Settings(
        llm=llm,
        llm2=llm2,
        embed=embed,
        db_path=Path(_env("DATAPILOT_DB_PATH", "data/ecommerce.duckdb")),
        retrieval_top_k=_env_int("DATAPILOT_RETRIEVAL_TOP_K", 5),
        max_repair_rounds=_env_int("DATAPILOT_MAX_REPAIR_ROUNDS", 3),
        mock=mock,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例（测试中可用 get_settings.cache_clear() 重置）。"""
    return load_settings()
