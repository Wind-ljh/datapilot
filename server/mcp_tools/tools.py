"""数据库 MCP 工具实现（ToolCore）+ FastMCP server 封装。

工具面固定为 5 个，粒度刻意拉平（面试讲点：工具过少→单次生成信息不足，
过多→选择空间爆炸、token 浪费；5 个是 ChatBI 场景的平衡点）。
"""

from __future__ import annotations

import json
import sys

from server.core.database import DuckDBAdapter, SQLSafetyError


class ToolCore:
    """进程内工具实现，MCP 层与 LangGraph 层共用。"""

    def __init__(self, adapter: DuckDBAdapter):
        self.adapter = adapter

    def list_tables(self) -> list[str]:
        return self.adapter.list_tables()

    def get_schema(self, table: str | None = None) -> list[dict]:
        return [
            {
                "name": s.name,
                "comment": s.comment,
                "row_count": s.row_count,
                "columns": [{"name": c.name, "type": c.dtype, "comment": c.comment} for c in s.columns],
                "ddl_text": s.to_ddl_text(),
            }
            for s in self.adapter.get_schema(table)
        ]

    def sample_rows(self, table: str, n: int = 5) -> list[dict]:
        return self.adapter.sample_rows(table, n=n)

    def validate_sql(self, sql: str) -> dict:
        ok, message = self.adapter.validate_sql(sql)
        return {"ok": ok, "message": message}

    def execute_sql(self, sql: str, max_rows: int = 200) -> dict:
        try:
            result = self.adapter.execute_sql(sql, max_rows=max_rows)
        except SQLSafetyError as e:
            return {"ok": False, "error": str(e), "columns": [], "rows": []}
        except Exception as e:  # 运行时错误（列不存在/类型错误等）→ 作为自修复信号
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "columns": [], "rows": []}
        return {
            "ok": True,
            "columns": result.columns,
            "rows": [list(r) for r in result.rows],
            "rowcount": result.rowcount,
            "elapsed_ms": result.elapsed_ms,
            "truncated": result.truncated,
        }


def build_mcp_server(db_path: str | None = None):
    """构建 FastMCP server 实例（stdio 传输）。"""
    from mcp.server.fastmcp import FastMCP

    adapter = DuckDBAdapter(path=db_path)
    core = ToolCore(adapter)
    mcp = FastMCP("datapilot-db", instructions="中文 ChatBI 数据库只读工具集")

    @mcp.tool()
    def list_tables() -> str:
        """列出分析库中所有表名。"""
        return json.dumps(core.list_tables(), ensure_ascii=False)

    @mcp.tool()
    def get_schema(table: str = "") -> str:
        """获取表结构：列名、类型、中文业务注释、行数。table 为空时返回全部表。"""
        return json.dumps(core.get_schema(table or None), ensure_ascii=False)

    @mcp.tool()
    def sample_rows(table: str, n: int = 5) -> str:
        """采样查看一张表的前 n 行真实数据，用于理解字段取值风格。"""
        return json.dumps(core.sample_rows(table, n=n), ensure_ascii=False, default=str)

    @mcp.tool()
    def validate_sql(sql: str) -> str:
        """只读安全校验 + EXPLAIN 预检 SQL，不执行数据。返回 ok 与错误信息。"""
        return json.dumps(core.validate_sql(sql), ensure_ascii=False)

    @mcp.tool()
    def execute_sql(sql: str, max_rows: int = 200) -> str:
        """执行只读 SELECT 查询并返回结果（最多 max_rows 行）。"""
        return json.dumps(core.execute_sql(sql, max_rows=max_rows), ensure_ascii=False, default=str)

    return mcp


def main() -> None:  # pragma: no cover - 手动运行入口
    server = build_mcp_server()
    print("[mcp_tools] DataPilot MCP server 启动（stdio）", file=sys.stderr)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
