"""评估模块测试：EX 判定语义 + mini 评估冒烟。"""

from __future__ import annotations

from eval.cspider import extract_order_by, has_order_by, results_equal


def test_results_equal_set_semantics():
    assert results_equal([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], ordered=False)
    assert not results_equal([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], ordered=True)


def test_results_equal_float_tolerance():
    assert results_equal([[0.1 + 0.2]], [[0.30001]], ordered=False)  # 均舍入到 4 位小数
    assert not results_equal([[1.0]], [[1.001]], ordered=False)


def test_results_equal_mismatch():
    assert not results_equal([[1]], [[1], [2]], ordered=False)


def test_has_order_by():
    assert has_order_by("SELECT x FROM t ORDER BY x")
    assert has_order_by("select x from t order by x desc")
    assert not has_order_by("SELECT x FROM t")


def test_mini_runner_smoke(adapter, db_path, tmp_path):
    from eval.mini_runner import run_mini_evaluation

    report = run_mini_evaluation(
        label="smoke", limit=3, db_path=db_path, out=str(tmp_path / "smoke.json")
    )
    assert report["n"] == 3
    assert {"ex", "exec_rate", "latency_p50_ms", "avg_tokens"} <= set(report.keys())
    assert (tmp_path / "smoke.json").exists()


def test_extract_order_by_alias():
    assert extract_order_by(
        "SELECT city AS 城市, COUNT(*) AS 用户数 FROM users GROUP BY city ORDER BY 用户数 DESC"
    ) == [1]
    assert extract_order_by(
        "SELECT strftime(created_at, '%Y-%m') AS 月份, COUNT(*) AS 订单数 "
        "FROM orders GROUP BY 月份 ORDER BY 月份"
    ) == [0]
    assert extract_order_by(
        "SELECT EXTRACT(year FROM created_at) AS 年份, EXTRACT(quarter FROM created_at) AS 季度, "
        "SUM(amount) AS 收入 FROM orders GROUP BY 年份, 季度 ORDER BY 年份, 季度"
    ) == [0, 1]
    # 裸列名（无别名）也能匹配
    assert extract_order_by(
        "SELECT name AS 商品名, stock AS 库存 FROM products ORDER BY stock ASC LIMIT 10"
    ) == [1]
    assert extract_order_by("SELECT x FROM t") == []


def test_extract_order_by_unparseable():
    # ORDER BY 引用非输出列的表达式（如 COUNT(*)）→ None，回退严格比较
    assert extract_order_by("SELECT COUNT(*) AS c FROM t ORDER BY COUNT(*) DESC") is None


def test_results_equal_tie_group_allows_reorder():
    # i=22 回归：并列组（用户数=258 出现多次）内换序应判等
    gold = [["北京", 258], ["上海", 258], ["广州", 258], ["深圳", 100]]
    pred = [["上海", 258], ["广州", 258], ["北京", 258], ["深圳", 100]]
    assert results_equal(pred, gold, ordered=True, order_keys_a=[1], order_keys_b=[1])
    # 旧语义（不传排序键）仍判不等：本修正是「放宽并列组」，而非「忽略顺序」
    assert not results_equal(pred, gold, ordered=True)


def test_results_equal_partial_tie_allows_reorder():
    # 合法并列：仅并列组内换序，非并列值的相对顺序保持不变
    gold = [["A", 9], ["B", 5], ["C", 5], ["D", 1]]
    pred = [["A", 9], ["C", 5], ["B", 5], ["D", 1]]
    assert results_equal(pred, gold, ordered=True, order_keys_a=[1], order_keys_b=[1])


def test_results_equal_real_order_error_with_partial_tie_fails():
    # 真正的排序错误（非并列值顺序错乱），即便存在并列也应判不等
    gold = [["A", 9], ["B", 5], ["C", 5], ["D", 1]]
    pred = [["B", 5], ["C", 5], ["A", 9], ["D", 1]]
    assert not results_equal(pred, gold, ordered=True, order_keys_a=[1], order_keys_b=[1])


def test_results_equal_tie_fallback_when_keys_missing():
    # 排序键缺失/不一致 → 回退旧语义（严格逐行比较）
    gold = [["北京", 258], ["上海", 258]]
    pred = [["上海", 258], ["北京", 258]]
    assert not results_equal(pred, gold, ordered=True)
    assert not results_equal(pred, gold, ordered=True, order_keys_a=[1], order_keys_b=[0])


def test_corrections_override():
    from eval.annotations import apply_corrections

    pairs = [
        {"question": "q0", "sql": "SELECT 1"},
        {"question": "q1", "sql": "SELECT 2"},
    ]
    corrections = {0: {"gold_sql_override": "SELECT 999", "issue": "退款口径"}}
    out = apply_corrections(pairs, corrections)
    assert out[0]["sql"] == "SELECT 999"
    assert out[0]["_correction"]["issue"] == "退款口径"
    assert out[1]["sql"] == "SELECT 2"
    assert "_correction" not in out[1]
    # 原 pairs 不被修改
    assert pairs[0]["sql"] == "SELECT 1"


def test_load_corrections_real_file():
    from eval.annotations import load_corrections

    corr = load_corrections()
    assert set(corr) == {6, 16, 35}
    for idx in (6, 16, 35):
        assert "已退款" not in corr[idx]["gold_sql_override"]
