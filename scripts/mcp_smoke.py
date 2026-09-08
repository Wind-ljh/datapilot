"""MCP stdio server 冒烟测试：真实拉起子进程 server，走完整 MCP 握手 + 工具调用。

用法：python scripts/mcp_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


async def main() -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.mcp_tools.tools"],
        env={**os.environ, "DATAPILOT_MOCK": "1"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("[smoke] tools:", sorted(t.name for t in tools.tools))

            res = await session.call_tool("list_tables", {})
            tables = json.loads(res.content[0].text)
            print("[smoke] list_tables:", tables)
            assert "orders" in tables, "演示库应包含 orders 表"

            res = await session.call_tool(
                "execute_sql", {"sql": "SELECT COUNT(*) AS n FROM orders"}
            )
            payload = json.loads(res.content[0].text)
            print("[smoke] execute_sql:", payload)
            assert payload["ok"] and payload["rows"][0][0] > 0

            res = await session.call_tool(
                "validate_sql", {"sql": "SELECT bogus_col FROM orders"}
            )
            verdict = json.loads(res.content[0].text)
            print("[smoke] validate_sql(bad):", verdict)
            assert verdict["ok"] is False

    print("[smoke] ALL MCP CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
