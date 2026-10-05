"""LLM 客户端：OpenAI 兼容协议 + 离线 Mock。

要点：
1. 统一 LLMResponse 返回文本与 token 用量，为成本类指标（每查询 Token 消耗）采集铺路；
2. MockLLMClient 让全链路在无 API Key 时可运行（CI/单测/演示兜底），
   mock 的 SQL 与演示库 schema 强绑定，保证确定性。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from server.core.config import LLMConfig, get_settings


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """OpenAI 兼容客户端（zhipu / siliconflow / 自建 vLLM 均适用）。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        from openai import (  # 延迟导入，mock 模式无需安装网络栈
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            OpenAI,
            RateLimitError,
        )

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        # 可重试的瞬时错误：限流 429 + 服务端 5xx（含 502/503/504）+ 网络抖动/超时
        self._retryable = (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.perf_counter()
        # GLM-4.7-Flash 等智谱"混合思考模型"默认开启思考：思维链写入
        # reasoning_content，可能占满 max_tokens 导致 content 为空（澄清/生成节点
        # 拿不到 JSON/SQL）。关闭思考后答案直接落在 content，与旧非思考模型行为一致。
        extra_body = {"thinking": {"type": "disabled"}} if self.config.provider == "zhipu" else None
        # 免费档 GLM-4.7-Flash 有并发/频率限制（错误码 1302/1305）且偶发 5xx 服务错误
        # （错误码 1234），对瞬时错误统一指数退避重试，避免单次抖动打断整轮评估
        # （重试等待计入该次调用延迟，如实反映真实耗时）。
        for attempt in range(8):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                break
            except self._retryable:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 60))  # 1,2,4,8,16,32,60s
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = resp.usage
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


class MockLLMClient:
    """离线确定性客户端：按关键词返回与演示库匹配的可执行 SQL。

    只服务于单测 / CI / 无 key 演示，生产路径不会走到这里。
    """

    _CANNED = [
        # (关键词, SQL)
        (["每月", "月度", "按月"], "SELECT strftime(created_at, '%Y-%m') AS month, COUNT(*) AS order_count, SUM(amount) AS revenue FROM orders GROUP BY month ORDER BY month"),
        (["最畅销", "销量", "卖得", "热门"], "SELECT p.name, SUM(oi.quantity) AS total_qty FROM order_items oi JOIN products p ON p.product_id = oi.product_id GROUP BY p.name ORDER BY total_qty DESC LIMIT 10"),
        (["城市", "地区", "地域"], "SELECT u.city, COUNT(o.order_id) AS order_count, SUM(o.amount) AS revenue FROM orders o JOIN users u ON u.user_id = o.user_id GROUP BY u.city ORDER BY revenue DESC LIMIT 10"),
        (["退款", "退货"], "SELECT COUNT(*) AS refunded_orders, SUM(amount) AS refunded_amount FROM orders WHERE status = '已退款'"),
        (["品类", "类别", "分类"], "SELECT c.name AS category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.product_id = oi.product_id JOIN categories c ON c.category_id = p.category_id GROUP BY c.name ORDER BY revenue DESC"),
    ]

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(provider="mock", api_key="", base_url="", model="mock")
        self.calls = 0

    @staticmethod
    def _canned_sql(question: str) -> str:
        for keywords, sql in MockLLMClient._CANNED:
            if any(k in question for k in keywords):
                return sql
        return "SELECT COUNT(*) AS total_orders FROM orders"

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls += 1
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), ""
        )

        if "请按系统要求输出分析报告" in last_user:
            # 报告节点：mock 一段结论
            text = (
                "从查询结果看，整体订单与收入趋势平稳，头部集中效应明显；"
                "建议结合时间维度进一步下钻定位波动来源。\n图表类型：bar"
            )
        else:
            # 生成/修复节点：只解析真实用户问题（避免被 schema/few-shot 文本干扰）
            m = re.search(r"用户问题[:：]\s*(.+?)\s*$", last_user, re.MULTILINE)
            question = m.group(1) if m else last_user
            text = self._canned_sql(question)

        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion_tokens = len(text) // 4
        return LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, latency_ms=1)


def build_llm(which: str = "llm") -> LLMClient | MockLLMClient:
    """工厂：mock 模式或缺 key 时返回 MockLLMClient。which: 'llm' | 'llm2'。"""
    settings = get_settings()
    cfg = getattr(settings, which)
    if cfg.provider == "mock" or not cfg.api_key:
        return MockLLMClient(cfg)
    return LLMClient(cfg)


def extract_sql(text: str) -> str:
    """从 LLM 输出中提取 SQL（容忍 markdown 代码块与前后缀说明）。"""
    fenced = re.search(r"```(?:sql)?\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    # 无代码块时取第一条以 SELECT/WITH 开头的语句
    match = re.search(r"\b(SELECT|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    if match:
        return text[match.start() :].strip().rstrip("；;")
    return text.strip()
