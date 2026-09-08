"""HTTP API 合同测试（mock 模式，注入会话级流水线，不依赖默认库文件）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(pipeline, monkeypatch):
    import server.api.main as api_main

    monkeypatch.setattr(api_main, "_pipeline", lambda: pipeline)
    return TestClient(api_main.app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mock"] is True


def test_schema_endpoint(client):
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    tables = {t["name"] for t in resp.json()["tables"]}
    assert {"orders", "products", "users"} <= tables


def test_query_end_to_end(client):
    resp = client.post("/api/query", json={"question": "每月的订单量和收入趋势是怎样的"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["need_clarify"] is False
    assert "orders" in body["sql"]
    assert body["rowcount"] >= 0
    assert body["report"]
    assert body["chart"] in {"bar", "line", "pie", "table"}
    assert body["metrics"]["llm_calls"] >= 1


def test_query_rejects_empty_question(client):
    resp = client.post("/api/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic 校验拦截
