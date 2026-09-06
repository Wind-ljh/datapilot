"""FastMCP 工具层：把数据库能力封装为标准 MCP 工具。

架构要点（面试讲点）：工具逻辑只实现一遍（ToolCore），
- 进程内：LangGraph 各节点直接调用 ToolCore（低延迟）；
- 跨进程：`python -m server.mcp_tools.tools` 以 stdio 运行标准 MCP server，
  任何 MCP 客户端（Claude Desktop / 其他 Agent）都能复用同一套工具。
"""

from server.mcp_tools.tools import ToolCore, build_mcp_server, main

__all__ = ["ToolCore", "build_mcp_server", "main"]
