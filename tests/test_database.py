"""数据层测试：只读安全防护、内省、执行。"""

from __future__ import annotations

import pytest

from server.core.database import SQLSafetyError


class TestReadonlyGuard:
    def test_select_allowed(self):
        from server.core.database import assert_readonly_sql

        assert_readonly_sql("SELECT COUNT(*) FROM orders")
        assert_readonly_sql("WITH t AS (SELECT 1 AS x) SELECT x FROM t")

    def test_insert_blocked(self):
        from server.core.database import assert_readonly_sql

        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("INSERT INTO orders VALUES (1)")

    def test_drop_blocked(self):
        from server.core.database import assert_readonly_sql

        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("DROP TABLE orders")

    def test_update_delete_blocked(self):
        from server.core.database import assert_readonly_sql

        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("UPDATE orders SET amount = 0")
        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("DELETE FROM orders")

    def test_multi_statement_blocked(self):
        from server.core.database import assert_readonly_sql

        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("SELECT 1; DROP TABLE orders")

    def test_comment_strip_not_bypassed(self):
        # 写关键字藏在注释里也不允许通过多语句
        from server.core.database import assert_readonly_sql

        with pytest.raises(SQLSafetyError):
            assert_readonly_sql("SELECT 1; -- SELECT\nDROP TABLE orders")


class TestDuckDBAdapter:
    def test_list_tables(self, adapter):
        tables = adapter.list_tables()
        assert {"categories", "products", "users", "orders", "order_items"} <= set(tables)

    def test_schema_comments_loaded(self, adapter):
        schemas = {s.name: s for s in adapter.get_schema()}
        assert "订单事实表" in schemas["orders"].comment
        col_comments = {c.name: c.comment for c in schemas["orders"].columns}
        assert "下单时间" in col_comments["created_at"]

    def test_execute_count(self, adapter):
        result = adapter.execute_sql("SELECT COUNT(*) AS total FROM orders")
        assert result.rowcount == 1
        assert result.rows[0][0] > 0

    def test_execute_truncation(self, adapter):
        result = adapter.execute_sql("SELECT order_id FROM orders", max_rows=10)
        assert result.rowcount == 10 and result.truncated

    def test_validate_bad_column(self, adapter):
        ok, _ = adapter.validate_sql("SELECT nonexistent_col FROM orders")
        assert ok is False

    def test_validate_good(self, adapter):
        ok, msg = adapter.validate_sql("SELECT COUNT(*) FROM orders WHERE status = '已退款'")
        assert ok, msg

    def test_execute_blocked_sql_raises(self, adapter):
        with pytest.raises(SQLSafetyError):
            adapter.execute_sql("DELETE FROM orders")

    def test_sample_rows(self, adapter):
        rows = adapter.sample_rows("categories", n=3)
        assert len(rows) == 3 and "name" in rows[0]
