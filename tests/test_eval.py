"""评估模块测试：EX 判定语义 + mini 评估冒烟。"""

from __future__ import annotations

from eval.cspider import has_order_by, results_equal


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
