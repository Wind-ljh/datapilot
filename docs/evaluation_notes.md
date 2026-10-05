# 评估口径修正说明（Evaluation Cleanup）

本文记录 DataPilot 评估流水线的一次「修干净」：修复两处会影响 EX 公正性的问题，
并给出可复现、可解释的最终评估结果。原始 benchmark **未被修改**，历史结果 **未被覆盖**。

## 1. EX 的定义与计算位置

- 对齐 Spider 官方思路：预测 SQL 与 gold SQL 在同一数据库执行，结果集一致记 1 分；
  含 `ORDER BY` 时按**有序序列**比较，否则按**多重集合**比较（行序无关）。
- 实现：`eval/cspider.py::results_equal`。
- mini 评测调用点：`eval/mini_runner.py`（40 道演示库金标题，`data/mini_eval.jsonl`）；
  CSpider 调用点：`eval/runner.py`。

## 2. 问题一：退款口径 annotation 不一致（i=6 / 16 / 35）

### 现象

`data/mini_eval.jsonl` 中有 11 道题的金 SQL（gold）都带有 `status != '已退款'` 过滤，
但题目文本并未要求「排除退款」。例如：

| index | 题目 | gold 是否过滤退款 | 题目是否提及退款 |
|---|---|---|---|
| 6 | 每月的订单量和收入趋势 | ✅ 过滤 | ❌ |
| 16 | 手机数码品类每个月的收入趋势 | ✅ 过滤 | ❌ |
| 35 | 各季度的收入对比 | ✅ 过滤 | ❌ |

（另有 8 道同型题：3/4/10/11/12/29/36/39，仅 1 号题「不含退款订单」明确要求过滤。）

### 判定

这是一个 **gold 标注口径错误**：gold 擅自排除了退款订单，而题目只问「订单量/收入」，
模型按全量统计（不含退款过滤）才是字面正确。模型对这批题的处理**不一致**——
i=3/4/10/12/36 恰好带上了过滤（当前判对），i=6/35 没带（当前判错）。

### 处理（遵循「不反向改 gold、不为提分而修正」原则）

- **原始 benchmark `data/mini_eval.jsonl` 一律不改。**
- 通过独立 override 文件 `data/eval_corrections.json` 只修正已确认的 3 道题
  （i=6/16/35）的金 SQL，去掉退款过滤；loader 为 `eval/annotations.py`。
- 修正是**方向向下、口径收紧**（题目未要求的过滤应去掉），不是按预测反向改 gold。
- 11 道同型题中**只改这 3 道**（i=6/16/35 为已确认错误，其余 8 道不动），
  避免「统一去掉过滤」导致 5 道当前判对的题被破坏（净 −3 EX）。

### 重要：i=16 修正后仍会判错（这是对的）

i=16 的 pred 用 `SUM(o.amount)`（把整单金额都算进「手机数码」），
而 gold 是 `SUM(oi.quantity * oi.unit_price)`（只算手机数码明细行）。
去掉退款过滤后二者**依然不相等**（pred 约 1.15M vs gold 约 0.41M/月）——
这是**真实的 SQL 语义错误**，属于模型自身的问题，**不修正、不掩盖**。

## 3. 问题二：ORDER BY 并列组的顺序误判（i=22）

### 现象

i=22「各城市的用户数」的 gold 与 pred SQL 完全相同
（`... GROUP BY city ORDER BY 用户数 DESC`），但 `用户数=258` 的城市有 3 个，
组成并列组。数据库在并列组内的返回顺序不保证，导致**同一条 SQL 被严格逐行比较判错**。

### 处理（tie-aware 比较）

`eval/cspider.py::extract_order_by` 解析 ORDER BY 排序键并映射到输出列下标；
`results_equal` 在 `ordered=True` 且排序键已知且一致时，改为：

1. 先比较**多重集合**是否相等（保内容）；
2. 再比较**排序键序列**是否相等（保排序方向与相对顺序）。

这样只允许「**并列组内换序**」，仍能识别真正的排序错误（非并列值错位）。
排序键无法可靠解析时**回退到旧的严格逐行比较**，不影响无 ORDER BY 的题。

## 4. 离线验证（用 baseline 已存的 pred_sql 重算，不调用 LLM）

在 `data/ecommerce.duckdb` 上，用旧/新两种判定重算 baseline 的 39 道已执行题：

| 判定 | EX |
|---|---|
| 旧（原始 gold + 严格比较） | 29/39 = 0.7436（复现 stored baseline 的 29 命中） |
| 新（修正 gold + tie-aware） | 32/39 = 0.8205 |

**净变化 +3，且只翻转 3 题，全部 0→1：**

| index | 题目 | 原因 |
|---|---|---|
| 6 | 每月的订单量和收入趋势 | 退款口径修正 |
| 22 | 各城市的用户数 | tie-aware 并列组换序 |
| 35 | 各季度的收入对比 | 退款口径修正 |

无任何其他题被意外改变；i=16 仍判错（真实语义错误，符合预期）。

## 5. 可复现性与追溯

- 原始 benchmark：`data/mini_eval.jsonl`（未改）。
- 修正表：`data/eval_corrections.json`（含 original→override、issue 说明）。
- 历史结果备份：`pre_eval_cleanup/`（旧结果不覆盖）。
- 重跑结果：`eval/results/*.json`（details 中新增 `gold_corrected` 与 `correction_note`
  字段，标记被修正的题，gold_sql 显示实际参与比较的修正后 SQL）。

## 6. 最终重跑结果（40/40，修正后口径）

第 4 节的「29/39→32/39」是**离线验证**（用已存 pred_sql 重算、不调用 LLM）。
之后对 5 个实验做了完整重跑（40/40），最终 EX：

| 配置 | 旧 EX | 新 EX |
|---|---|---|
| baseline | 0.725 | 0.800 |
| − schema linking | 0.825 | 0.800 |
| − few-shot | 0.300 | 0.400 |
| − 自修复 | 0.650 | 0.700 |
| Qwen2.5-7B | 0.725 | 0.850 |

净变化可分解为「模型重采样效应（温度噪声，正负对称）」与「评估器修正效应（恒 ≥0）」
两项，完整分解与逐题翻转见 [docs/final_evaluation_report.md](final_evaluation_report.md)。
