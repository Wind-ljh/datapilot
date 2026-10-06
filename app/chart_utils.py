"""图表 X/Y 轴推断与渲染前预处理（纯 pandas，不依赖 Streamlit，便于单测）。

解决连续对话中 Streamlit 图表"异变"的两个根因：
1. 显式推断并指定 x/y 列，避免 Streamlit 对混合类型 DataFrame 自动猜列；
2. 渲染前返回 df 副本并统一处理日期排序，绝不原地修改历史 chart_df。
"""

from __future__ import annotations

import datetime as dt
import decimal
import numbers
import re
from dataclasses import dataclass

import pandas as pd

# ID / 编号类字段：不作为用户可读的 X/Y 轴
_ID_NAME_RE = re.compile(
    r"(?:^|[\s_])?(id|编号|序号|代码|编码|code|key|guid|uuid)(?:$|[\s_])?",
    re.IGNORECASE,
)

# 金额类指标：作为"主要指标"优先级最高
_AMOUNT_RE = re.compile(
    r"金额|销售额|收入|营收|总额|合计|利润|成本|价格|单价|流水|"
    r"amount|revenue|sales|income|price|cost|profit|gmv|total",
    re.IGNORECASE,
)

# 数量类指标：优先级次之
_COUNT_RE = re.compile(
    r"订单|销量|数量|计数|次数|件数|笔数|退款|成交|"
    r"order|count|quantity|qty|volume|refund",
    re.IGNORECASE,
)

# 时间/日期类字段名
_TIME_NAME_RE = re.compile(
    r"日期|时间|月份|年份|季度|周|月度|年度|"
    r"date|time|month|year|quarter|week|day",
    re.IGNORECASE,
)

# 日期样式字符串值，如 2024-01、2024/01/01、2024年1月
_DATE_VALUE_RE = re.compile(r"^\d{4}[-/.年]\d{1,2}([-/.月]\d{1,2}日?)?$")


def is_id_like(name: object) -> bool:
    """字段名是否像 ID/编号（不作为用户可读的 X/Y 轴）。"""
    return bool(_ID_NAME_RE.search(str(name)))


def _is_time_name(name: object) -> bool:
    return bool(_TIME_NAME_RE.search(str(name)))


def _is_amount(name: object) -> bool:
    return bool(_AMOUNT_RE.search(str(name)))


def _is_count(name: object) -> bool:
    return bool(_COUNT_RE.search(str(name)))


def _is_numeric_column(series: pd.Series) -> bool:
    """数值列判定：兼容 int/float 与 DuckDB DECIMAL→decimal.Decimal 的 object 列。"""
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = [v for v in series.dropna().head(20) if v is not None]
    if not sample:
        return False
    return all(
        isinstance(v, (int, float, decimal.Decimal)) and not isinstance(v, bool)
        for v in sample
    )


def looks_date_like(series: pd.Series) -> bool:
    """列值是否为日期/时间（datetime 类型或日期样式字符串）。"""
    if series is None or series.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = [v for v in series.dropna().head(20) if v is not None]
    if not sample:
        return False
    if all(isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) for v in sample):
        return True
    if all(isinstance(v, str) and _DATE_VALUE_RE.match(v.strip()) for v in sample):
        return True
    return False


def coerce_datetime(series: pd.Series) -> pd.Series:
    """把日期样式列转成 datetime 以保证时间顺序；转换失败则原样返回。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    converted = pd.to_datetime(series, errors="coerce")
    if converted.notna().mean() >= 0.5:
        return converted
    return series


def infer_xy(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """推断图表的 (x, y)。无法可靠确定时对应项为 None。

    规则（只依赖 dtype + 语义类别，不写死具体字段名）：
    - Y：数值列且非 ID；金额类 > 数量类 > 其他数值列，取主要指标。
    - X：日期/时间列 > 类别列（非数值、非 ID）。
    """
    if df is None or df.empty or df.shape[1] == 0:
        return None, None

    numeric = [c for c in df.columns if _is_numeric_column(df[c])]
    metric_cols = [c for c in numeric if not is_id_like(c)]

    y = next((c for c in metric_cols if _is_amount(c)), None)
    if y is None:
        y = next((c for c in metric_cols if _is_count(c)), None)
    if y is None:
        y = metric_cols[0] if metric_cols else None
    if y is None:
        return None, None

    time_cols = [
        c
        for c in df.columns
        if c != y
        and not is_id_like(c)
        and not _is_numeric_column(df[c])
        and (
            pd.api.types.is_datetime64_any_dtype(df[c])
            or _is_time_name(c)
            or looks_date_like(df[c])
        )
    ]
    cat_cols = [
        c
        for c in df.columns
        if c != y and not _is_numeric_column(df[c]) and not is_id_like(c)
    ]
    x = time_cols[0] if time_cols else (cat_cols[0] if cat_cols else None)
    return x, y


def prepare_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    """返回渲染用副本，必要时把日期 X 列转 datetime；绝不修改入参。"""
    out = df.copy()
    x, y = infer_xy(out)
    if x is not None and y is not None and looks_date_like(out[x]):
        out[x] = coerce_datetime(out[x])
    return out


@dataclass(frozen=True)
class VizSpec:
    """一次图表渲染的可视化决策结果。

    mode:
        "metric" —— 单值 / Top 1，用 KPI（st.metric）展示；
        "bar"    —— 纵向柱状图；
        "hbar"   —— 横向柱状图（类别标签较长或类别较多）；
        "line"   —— 折线图（时间序列）；
        "area"   —— 面积图；
        "table"  —— 无法可靠作图 / pie / table，仅展示表格。
    """

    mode: str
    x: str | None = None
    y: str | None = None
    height: int | None = None
    horizontal: bool = False


def format_number(value: object) -> str:
    """把数值格式化为可读字符串：千分位，浮点最多两位小数并去掉尾零。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):  # int / np.integer
        return f"{int(value):,}"
    if isinstance(value, (numbers.Real, decimal.Decimal)):  # float / np.floating / Decimal
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _is_time_column(df: pd.DataFrame, col: str) -> bool:
    """col 是否为时间/日期轴（datetime 类型、时间类字段名或日期样式值）。"""
    return (
        pd.api.types.is_datetime64_any_dtype(df[col])
        or _is_time_name(col)
        or looks_date_like(df[col])
    )


def _should_be_horizontal(df: pd.DataFrame, x: str | None, n: int) -> bool:
    """柱状图是否改用横向：类别名较长或类别较多时，横向避免标签拥挤。"""
    if x is None:
        return False
    if n >= 6:
        return True
    labels = df[x].dropna().astype(str)
    if len(labels) == 0:
        return False
    return int(labels.str.len().max()) >= 6


def _bar_height(n: int, horizontal: bool) -> int:
    """柱状图高度：横向随行数增长（保证标签可读），纵向按规模分级。"""
    if horizontal:
        return min(max(220, 26 * n + 40), 560)
    return 180 if n <= 3 else 320


def metric_label(df: pd.DataFrame, x: str | None, y: str) -> str:
    """单值 KPI 的标签：有类别时拼接「指标 · 类别值」，否则用指标名。"""
    if x is None or _is_time_column(df, x):
        return str(y)
    vals = df[x].dropna()
    if len(vals) == 0:
        return str(y)
    return f"{y} · {vals.iloc[0]}"


def classify_result(df: pd.DataFrame, chart_type: str | None) -> VizSpec:
    """根据数据规模 + 问题意图决定展示方式（决策规则的核心入口）。

    只依赖 dtype + 语义类别 + 行数，不写死任何具体问题/字段名：
    - 1 行          → KPI（单值 / Top 1）；
    - 2～3 行       → 紧凑图表（降高度）；
    - 4～15 行      → 正常图表（类别名长或类别多时横向柱状图）；
    - >15 行        → 正常图表，横向柱状图随行数增加高度。
    - 时间序列 X    → line/area 保持时间顺序。
    """
    if df is None or df.empty or df.shape[1] == 0:
        return VizSpec("table")
    x, y = infer_xy(df)
    if y is None:
        return VizSpec("table", x=x, y=y)

    base = (chart_type or "").strip().lower()
    if base not in {"bar", "line", "area"}:
        # pie / table / 未给类型：沿用旧行为，仅展示表格，不做图表
        return VizSpec("table", x=x, y=y)

    n = len(df)

    if n == 1:
        return VizSpec("metric", x=x, y=y)

    if base == "bar":
        horizontal = _should_be_horizontal(df, x, n)
        mode = "hbar" if horizontal else "bar"
        return VizSpec(mode, x=x, y=y, height=_bar_height(n, horizontal), horizontal=horizontal)

    # line / area
    return VizSpec(base, x=x, y=y, height=180 if n <= 3 else 320)
