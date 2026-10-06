"""图表 X/Y 推断与渲染前预处理（纯 pandas 层）回归测试。

覆盖连续对话中图表"异变"与 mixed-types 异常的根本修复点：
- 显式推断 x/y，不再依赖 Streamlit 自动猜列；
- 渲染预处理返回副本，绝不原地修改历史 chart_df；
- 日期字符串按时间顺序排序，而非字典序。
"""

from __future__ import annotations

import datetime as dt
import decimal

import pandas as pd

from app.chart_utils import (
    classify_result,
    coerce_datetime,
    format_number,
    infer_xy,
    is_id_like,
    looks_date_like,
    metric_label,
    prepare_chart_df,
)


def test_infer_xy_line_monthly_orders():
    # 趋势类：月份(string) 作为 x，订单量作为 y
    df = pd.DataFrame({"月份": ["2024-01", "2024-02", "2024-03"], "订单量": [327, 512, 799]})
    x, y = infer_xy(df)
    assert x == "月份"
    assert y == "订单量"


def test_infer_xy_bar_category_revenue():
    # 类别对比类：类目名称 x，销售额 y
    df = pd.DataFrame(
        {"类目名称": ["云杉运动鞋", "青竹口红", "白鹭充电宝"], "销售额": [8319858.61, 7817908.25, 7681458.99]}
    )
    x, y = infer_xy(df)
    assert x == "类目名称"
    assert y == "销售额"


def test_infer_xy_multiple_metrics_prefers_amount():
    # 多个数值列：优先金额类作为主要指标（而非订单数）
    df = pd.DataFrame({"类目名称": ["A", "B"], "订单数": [10, 20], "销售额": [1000.0, 2000.0]})
    x, y = infer_xy(df)
    assert x == "类目名称"
    assert y == "销售额"


def test_infer_xy_ignores_id_column():
    # 类目ID 不应作为可读的 X/Y
    df = pd.DataFrame({"类目ID": [1, 2, 3], "类目名称": ["A", "B", "C"], "销售额": [100, 200, 300]})
    x, y = infer_xy(df)
    assert x == "类目名称"
    assert y == "销售额"


def test_id_like_vs_category_name():
    assert is_id_like("类目ID")
    assert is_id_like("user_id")
    assert is_id_like("订单编号")
    assert is_id_like("商品编码")
    assert not is_id_like("类目名称")
    assert not is_id_like("商品名称")
    assert not is_id_like("城市")


def test_infer_xy_no_numeric_returns_none():
    df = pd.DataFrame({"名称": ["A", "B"]})
    x, y = infer_xy(df)
    assert y is None


def test_infer_xy_mixed_types_does_not_raise():
    # 字符串类别 + 多个数值列：不应抛出 mixed-types，且能确定主要指标
    df = pd.DataFrame({"类目名称": ["A", "B"], "销售额": [1.0, 2.0], "订单数": [3, 4]})
    x, y = infer_xy(df)
    assert x == "类目名称"
    assert y == "销售额"


def test_infer_xy_decimal_numeric_column():
    # DuckDB DECIMAL → decimal.Decimal 的 object 列也应识别为数值
    df = pd.DataFrame(
        {"类目名称": ["A", "B"], "销售额": [decimal.Decimal("100.50"), decimal.Decimal("200.25")]}
    )
    x, y = infer_xy(df)
    assert x == "类目名称"
    assert y == "销售额"


def test_infer_xy_datetime_object_column():
    df = pd.DataFrame(
        {"日期": [dt.date(2024, 1, 1), dt.date(2024, 2, 1)], "订单量": [10, 20]}
    )
    x, y = infer_xy(df)
    assert x == "日期"
    assert y == "订单量"


def test_looks_date_like():
    assert looks_date_like(pd.Series(["2024-01", "2024-02"]))
    assert looks_date_like(pd.Series(["2024/01/01", "2024/02/01"]))
    assert not looks_date_like(pd.Series(["类目A", "类目B"]))


def test_coerce_datetime_orders_chronologically():
    # 字典序会把 2024-10 排在 2024-02 之前；转 datetime 后应时间顺序正确
    s = pd.Series(["2024-02", "2024-10", "2024-01"])
    converted = coerce_datetime(s)
    assert pd.api.types.is_datetime64_any_dtype(converted)
    assert converted.min() == pd.Timestamp("2024-01-01")
    assert converted.max() == pd.Timestamp("2024-10-01")


def test_prepare_chart_df_does_not_mutate_input():
    # 历史 chart_df 必须不被后续渲染修改
    df = pd.DataFrame({"月份": ["2024-02", "2024-01"], "订单量": [2, 1]})
    before = df.copy(deep=True)
    out = prepare_chart_df(df)
    assert out is not df
    pd.testing.assert_frame_equal(df, before)  # 原 df 完全未变
    assert not pd.api.types.is_datetime64_any_dtype(df["月份"])  # 原列仍是字符串
    assert pd.api.types.is_datetime64_any_dtype(out["月份"])  # 副本中已转 datetime


def test_prepare_chart_df_stable_across_calls():
    # rerun 场景下多次调用结果一致，且不修改原 df
    df = pd.DataFrame({"类目名称": ["A", "B"], "销售额": [100.0, 200.0]})
    out1 = prepare_chart_df(df)
    out2 = prepare_chart_df(df)
    pd.testing.assert_frame_equal(out1, out2)
    pd.testing.assert_frame_equal(df, pd.DataFrame({"类目名称": ["A", "B"], "销售额": [100.0, 200.0]}))


# ---------------------------------------------------------------------------
# 可视化决策规则（数据规模 + 意图 → KPI / 紧凑 / 正常 / 横向 / 时间序列）
# ---------------------------------------------------------------------------


def test_classify_single_value_uses_metric():
    df = pd.DataFrame({"类目名称": ["服饰鞋包"], "销售额": [127482200.0]})
    spec = classify_result(df, "bar")
    assert spec.mode == "metric"
    assert spec.x == "类目名称"
    assert spec.y == "销售额"


def test_classify_single_scalar_uses_metric():
    # 无类别列的单值结果（如 COUNT(*)）也应走 KPI
    df = pd.DataFrame({"总订单量": [1200]})
    spec = classify_result(df, "bar")
    assert spec.mode == "metric"
    assert spec.x is None
    assert spec.y == "总订单量"


def test_classify_two_three_rows_compact():
    df = pd.DataFrame({"类目名称": ["A", "B", "C"], "销售额": [1000.0, 900.0, 850.0]})
    spec = classify_result(df, "bar")
    assert spec.mode == "bar"  # 短标签、少量类别 → 纵向紧凑
    assert spec.height is not None and spec.height < 320


def test_classify_many_rows_horizontal():
    df = pd.DataFrame({"商品名称": [f"商品{i}" for i in range(10)], "销量": list(range(10))})
    spec = classify_result(df, "bar")
    assert spec.mode == "hbar"
    assert spec.horizontal
    assert spec.height is not None


def test_classify_long_labels_horizontal():
    df = pd.DataFrame(
        {"商品名称": ["高性能运动跑鞋", "便携式移动电源", "无线蓝牙耳机"], "销量": [1, 2, 3]}
    )
    spec = classify_result(df, "bar")
    assert spec.mode == "hbar"


def test_classify_normal_bar_four_rows():
    df = pd.DataFrame({"类目名称": ["A", "B", "C", "D"], "销售额": [1.0, 2.0, 3.0, 4.0]})
    spec = classify_result(df, "bar")
    assert spec.mode == "bar"
    assert spec.height == 320


def test_classify_time_series_line():
    df = pd.DataFrame({"月份": ["2024-01", "2024-02", "2024-03"], "订单量": [100, 200, 300]})
    spec = classify_result(df, "line")
    assert spec.mode == "line"


def test_classify_pie_falls_back_to_table():
    df = pd.DataFrame({"类目名称": ["A", "B"], "销售额": [1.0, 2.0]})
    spec = classify_result(df, "pie")
    assert spec.mode == "table"
    assert spec.y is not None  # y 可识别，但 pie 无图表实现 → 仅表格


def test_classify_no_numeric_falls_back_to_table():
    df = pd.DataFrame({"名称": ["A", "B"]})
    spec = classify_result(df, "bar")
    assert spec.mode == "table"
    assert spec.y is None


def test_classify_more_than_fifteen_rows_scales_height():
    df = pd.DataFrame({"商品名称": [f"商品{i}" for i in range(20)], "销量": list(range(20))})
    spec = classify_result(df, "bar")
    assert spec.mode == "hbar"
    assert spec.height is not None and spec.height > 320


def test_format_number():
    assert format_number(127482200) == "127,482,200"
    assert format_number(12748.22) == "12,748.22"
    assert format_number(100.0) == "100"
    assert format_number(decimal.Decimal("100.5")) == "100.5"
    assert format_number(None) == ""


def test_metric_label_with_category():
    df = pd.DataFrame({"类目名称": ["服饰鞋包"], "销售额": [127482200.0]})
    assert metric_label(df, "类目名称", "销售额") == "销售额 · 服饰鞋包"


def test_metric_label_no_category():
    df = pd.DataFrame({"总订单量": [1200]})
    assert metric_label(df, None, "总订单量") == "总订单量"
