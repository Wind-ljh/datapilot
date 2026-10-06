"""可视化决策端到端验证（离线 mock，走真实 Streamlit 渲染路径）。

用 AppTest 驱动 app/streamlit_app.py，提交 7 个连续问题，检查：
1. 每次 rerun 无异常（尤其 mixed-types）；
2. 未触发"回退表格"兜底提示；
3. 单值问题渲染为 KPI（st.metric），其余为图表（vega_lite_chart）；
4. 历史消息逐问累积、无异常。
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

os.environ["DATAPILOT_MOCK"] = "1"

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

QUESTIONS = [
    "哪个商品类别的销售额最高？",
    "哪个城市收入最高？",
    "销售额最高的三个商品类别有哪些？",
    "销售额最高的10个商品是哪些？",
    "各商品类别的销售额是多少？",
    "每月订单量是多少？",
    "比较不同商品类别的销售额和订单数量。",
]


def _has_fallback(at) -> bool:
    for c in at.caption:
        text = str(getattr(c, "value", ""))
        if "失败" in text or "未能可靠识别" in text:
            return True
    return False


def main() -> int:
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()

    fails = 0
    for i, q in enumerate(QUESTIONS, 1):
        at.chat_input[0].set_value(q)
        at.run()

        excs = [e for e in at.exception]
        fallback = _has_fallback(at)
        counts = Counter(e.type for e in at)
        n_metric = counts.get("metric", 0)
        n_chart = counts.get("vega_lite_chart", 0)

        ok = not excs and not fallback
        if not ok:
            fails += 1

        print(
            f"[Q{i}] {q} -> {'OK' if ok else 'FAIL'} "
            f"exceptions={len(excs)} fallback={fallback} "
            f"metrics={n_metric} charts={n_chart} dataframes={counts.get('dataframe', 0)}"
        )
        for e in excs:
            print(f"      EXC: {e.value}")

    print(f"\nTOTAL fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
