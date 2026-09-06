"""DuckDB 数据适配层：schema 内省 / 采样 / SQL 校验 / 只读执行。

安全设计（面试常问"Text-to-SQL 如何防注入/防写操作"）：
1. 只允许单条 SELECT/WITH 语句（语句数校验 + 起始关键字校验）；
2. 词边界正则拦截 DML/DDL/ATTACH/COPY 等危险关键字；
3. validate_sql 用 EXPLAIN 预检，不执行即发现语法/列名错误，为自修复回环提供信号。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import duckdb

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|EXPORT|"
    r"IMPORT|INSTALL|LOAD|CALL|PRAGMA|VACUUM|CHECKPOINT|GRANT|REVOKE|SET)\b",
    re.IGNORECASE,
)
_SELECT_PREFIX = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class SQLSafetyError(ValueError):
    """不满足只读安全约束的 SQL。"""


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    comment: str = ""


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    comment: str = ""
    row_count: int = 0

    def to_ddl_text(self) -> str:
        """渲染成 prompt 友好的 schema 文本（含中文列注释）。"""
        lines = [f"TABLE {self.name} -- {self.comment}" if self.comment else f"TABLE {self.name}"]
        for col in self.columns:
            suffix = f"  -- {col.comment}" if col.comment else ""
            lines.append(f"  {col.name} {col.dtype}{suffix}")
        return "\n".join(lines)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    rowcount: int
    elapsed_ms: int
    truncated: bool = False


def assert_readonly_sql(sql: str) -> None:
    """只读安全校验，不通过抛 SQLSafetyError。"""
    cleaned = re.sub(r"--[^\n]*", " ", sql)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
    statements = [s for s in cleaned.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("仅允许单条 SQL 语句")
    single = statements[0]
    if not _SELECT_PREFIX.match(single):
        raise SQLSafetyError("仅允许 SELECT / WITH 查询")
    if _FORBIDDEN.search(single):
        raise SQLSafetyError("包含被禁止的写操作/管理关键字")


class DuckDBAdapter:
    """DuckDB 只读分析适配器。path 传 ':memory:' 可创建内存库（测试用）。"""

    def __init__(self, path: str | None = None, read_only: bool | None = None):
        from server.core.config import get_settings

        resolved = path or str(get_settings().db_path)
        if read_only is None:
            read_only = resolved != ":memory:"
        self.conn = duckdb.connect(resolved, read_only=read_only)

    # ---------- 内省 ----------
    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_schema(self, table: str | None = None) -> list[TableSchema]:
        tables = [table] if table else self.list_tables()
        schemas: list[TableSchema] = []
        for t in tables:
            col_rows = self.conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name = ? ORDER BY ordinal_position",
                [t],
            ).fetchall()
            try:
                table_comment = self.conn.execute(
                    "SELECT comment FROM duckdb_tables() WHERE table_name = ?", [t]
                ).fetchone()
                table_comment = (table_comment[0] if table_comment else "") or ""
            except duckdb.Error:
                table_comment = ""
            columns: list[ColumnInfo] = []
            for name, dtype in col_rows:
                try:
                    col_comment = self.conn.execute(
                        "SELECT comment FROM duckdb_columns() WHERE table_name = ? AND column_name = ?",
                        [t, name],
                    ).fetchone()
                    col_comment = (col_comment[0] if col_comment else "") or ""
                except duckdb.Error:
                    col_comment = ""
                columns.append(ColumnInfo(name=name, dtype=dtype, comment=col_comment))
            row_count = self.conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            schemas.append(TableSchema(name=t, columns=columns, comment=table_comment, row_count=row_count))
        return schemas

    def sample_rows(self, table: str, n: int = 5) -> list[dict]:
        assert re.fullmatch(r"\w+", table), "非法表名"
        rows = self.conn.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}').fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # ---------- SQL 执行 ----------
    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """EXPLAIN 预检：语法/列名错误不执行数据即可发现。返回 (ok, message)。"""
        try:
            assert_readonly_sql(sql)
        except SQLSafetyError as e:
            return False, str(e)
        try:
            self.conn.execute(f"EXPLAIN {sql}")
            return True, "ok"
        except duckdb.Error as e:
            return False, str(e).split("\n")[0]

    def execute_sql(self, sql: str, max_rows: int = 200) -> QueryResult:
        assert_readonly_sql(sql)
        start = time.perf_counter()
        cur = self.conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        columns = [d[0] for d in cur.description] if cur.description else []
        return QueryResult(
            columns=columns,
            rows=rows,
            rowcount=len(rows),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    def close(self) -> None:
        self.conn.close()


class SQLiteAdapter:
    """sqlite3 适配器：与 DuckDBAdapter 同接口，服务于 CSpider 评估（其库为 SQLite 格式）。"""

    def __init__(self, path: str):
        import sqlite3

        self.conn = sqlite3.connect(path, check_same_thread=False)

    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [r[0] for r in rows]

    def get_schema(self, table: str | None = None) -> list[TableSchema]:
        tables = [table] if table else self.list_tables()
        schemas: list[TableSchema] = []
        for t in tables:
            info = self.conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            columns = [ColumnInfo(name=r[1], dtype=r[2] or "TEXT", comment="") for r in info]
            row_count = self.conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            schemas.append(TableSchema(name=t, columns=columns, row_count=row_count))
        return schemas

    def sample_rows(self, table: str, n: int = 5) -> list[dict]:
        assert re.fullmatch(r"\w+", table), "非法表名"
        cur = self.conn.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}')
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        try:
            assert_readonly_sql(sql)
        except SQLSafetyError as e:
            return False, str(e)
        try:
            self.conn.execute(f"EXPLAIN {sql.rstrip(';')}")
            return True, "ok"
        except Exception as e:  # sqlite3.Error
            return False, str(e)

    def execute_sql(self, sql: str, max_rows: int = 200) -> QueryResult:
        assert_readonly_sql(sql)
        start = time.perf_counter()
        cur = self.conn.execute(sql.rstrip(";"))
        rows = cur.fetchmany(max_rows + 1)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        columns = [d[0] for d in cur.description] if cur.description else []
        return QueryResult(
            columns=columns,
            rows=rows[:max_rows],
            rowcount=len(rows[:max_rows]),
            elapsed_ms=elapsed_ms,
            truncated=len(rows) > max_rows,
        )

    def close(self) -> None:
        self.conn.close()
