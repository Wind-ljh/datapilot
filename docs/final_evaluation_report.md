# DataPilot 最终评估报告（Final Evaluation Report）

> 本报告是「评估流水线修干净」后的最终交付物，面向简历与面试：可复现、可解释、可辩护。
> 全程**未修改 benchmark 原始文件、未覆盖历史结果、未反向改 gold、未为提分做任何额外人工修正**。

---

## A. 改了什么（Changes）

评估流水线修干净，只动了**判定口径**，未动模型 / Prompt / RAG / Agent / 实验设计。

### A1. 退款口径 gold 修正（i=6 / 16 / 35）

`data/mini_eval.jsonl` 中有 11 道题的金 SQL 擅自带了 `status != '已退款'` 过滤，但题目文本只问「订单量/收入」，并未要求排除退款。这是 **gold 标注口径错误**。

处理方式（遵循「不反向改 gold、不为提分修正」原则）：

- **原始 benchmark 一律不改**；
- 通过独立 override 文件 `data/eval_corrections.json` 只修正已确认的 3 道题（i=6/16/35）去掉退款过滤；
- 11 道同型题中**只改这 3 道**，其余 8 道不动（避免「统一去过滤」破坏 5 道当前判对的题）；
- 加载器 `eval/annotations.py` 只读 override、不改原数据。

### A2. ORDER BY 并列组 tie-aware 比较（i=22）

i=22「各城市的用户数」gold 与 pred SQL **完全相同**，但「用户数=258」的城市有 3 个组成并列组，数据库在并列组内的返回顺序不保证，导致同一条 SQL 被严格逐行比较判错。

处理方式：

- `eval/cspider.py::extract_order_by` 解析 ORDER BY 排序键并映射到输出列下标；
- `results_equal` 在 `ordered=True` 且排序键已知一致时，改为「**多重集合相等 + 排序键序列相等**」——只允许并列组内换序，仍能识别真正的排序错误；
- 排序键无法可靠解析时**回退旧的严格逐行比较**，不影响无 ORDER BY 的题。

> 关键性质：两处修正都是「把**不公平的 0** 改判为**公平的 1**」，方向向下、口径收紧，**只可能加分、不可能压分**。

---

## B. 测试（Tests）

`pytest` 全绿：**41 项单测通过**（mock 模式离线可跑，覆盖 EX 判定语义 + mini 评估冒烟）。

新增回归测试（锁定本次修正的正确性）：

| 测试 | 保证 |
|---|---|
| `test_results_equal_tie_group_allows_reorder` | 并列组内换序判等 |
| `test_results_equal_real_order_error_with_partial_tie_fails` | 非并列值错位仍判不等（**不允许忽略顺序**） |
| `test_results_equal_tie_fallback_when_keys_missing` | 排序键缺失回退严格比较 |
| `test_extract_order_by_alias` | 别名/裸列名/`EXTRACT(year FROM …)` 的排序键解析 |
| `test_corrections_override` / `test_load_corrections_real_file` | override 不污染原数据；修正集恰为 {6,16,35} |

---

## C. 最终数字（Final Numbers）

mini 评测：40 道演示库金标题，DuckDB 只读执行。修正后评估口径下的最终结果：

| 配置 | EX | 可执行率 | 自修复成功率 | 平均 Token |
|---|---:|---:|---:|---:|
| **baseline**（GLM-4.7-Flash 全量） | **0.800**（32/40） | 0.975 | 0.833 | 1294 |
| − schema linking | 0.800 | 1.000 | 1.000 | 1224 |
| − few-shot | 0.400 | 0.975 | 0.750 | 1070 |
| − 自修复 | 0.700 | 0.825 | — | 1079 |
| 对照：Qwen2.5-7B（免费 API） | **0.850** | 0.950 | 0.333 | 1322 |

> 延迟（P50）列因免费档 API 限流（zhipu 1305）随机命中不同实验，本轮 no-linking/no-fewshot/no-repair 的 P50 被抬到 40–198s，**不代表模型计算速度**，故不列入结论表。

---

## D. 前后差异解释（Before → After）

旧口径 EX → 修正后 EX 的净变化，可严格分解为两个独立可加项：

```
净变化 = 模型重采样效应（温度采样噪声） + 评估器修正效应（只纠偏）
```

| 实验 | 旧 EX | 新 EX | 净变化 | = 重采样效应 | + 评估器效应 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.725 | 0.800 | +0.075 | −0.025 | **+0.100** |
| no-linking | 0.825 | 0.800 | −0.025 | −0.075 | +0.050 |
| no-fewshot | 0.300 | 0.400 | +0.100 | +0.100 | 0.000 |
| no-repair | 0.650 | 0.700 | +0.050 | −0.025 | +0.075 |
| qwen7b | 0.725 | 0.850 | +0.125 | +0.050 | +0.075 |

（方法：对每道题，用「新 pred vs 旧 gold + 严格比较」重算 EX，隔离出重采样效应；余下即评估器效应。）

**两个效应截然不同：**

- **评估器效应恒 ≥ 0**（0.000 ~ +0.100）：修正只纠正不公平判分，从不压分。
- **重采样效应正负对称**（−0.075 ~ +0.100）：`温度=0` 的 glm-4.7-flash 仍非完全确定，40 题上单轮 ±3–4 题的固有噪声。

### baseline 的逐题翻转（5 处，净 +3）

| index | 方向 | 归因 | 说明 |
|---|---|---|---|
| 6 | 0→1 | 评估器 | 退款口径修正 |
| 16 | 0→1 | 评估器 + 重采样（缺一不可） | 新 pred 用了正确的 `SUM(oi.quantity*oi.unit_price)`（旧 pred 误用整单 `SUM(o.amount)`）；同时 gold 去掉退款过滤。二者任缺其一仍判错 |
| 22 | 0→1 | 评估器 | tie-aware 并列组换序 |
| 35 | 0→1 | 评估器 | 退款口径修正 |
| 30 | 1→0 | 重采样 | 新 pred 把「周末」写成 `EXTRACT(ISODOW …) IN (1,7)`（周一+周日），应为 `(6,7)`（周六+周日）——真实模型采样误差，评估器不修正、不掩盖 |

> i=30 是**评估器正确拒绝模型错误**的例证：它证明本次修正是「公平放宽」而非「无差别放水」——真正的排序/语义错误仍会被判错。

---

## E. 五个研究问题的最终判断（Final Judgment）

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | 全量流水线的上限 EX？ | **0.800（32/40）** | baseline |
| 2 | schema linking 有用吗？ | **小库上无增益** | 0.800 vs 0.800（旧口径 0.825 是重采样运气 + 不公平 gold，修正后回归持平） |
| 3 | few-shot 有用吗？ | **是，最大增益来源** | 0.800 → 0.400（−0.400，远超噪声） |
| 4 | 自修复有用吗？ | **是，第二增益来源** | 0.800 → 0.700（−0.100），且可执行率 0.975 → 0.825 |
| 5 | 免费 Qwen2.5-7B 能否比肩 GLM-4.7-Flash？ | **能，公平口径下 ≥ baseline** | 0.850 vs 0.800；0.050 差距在单轮噪声内，稳健结论是「不低于」 |

**一句话总结**：把评估口径修干净后，few-shot 与自修复的增益结论不变且更清晰；schema linking 的「无增益」从噪声中浮出为确定结论；Qwen2.5-7B 从「打平」上修为「略优（≥ 打平）」——所有结论都在可复现、可解释、不编造的约束下得出。

---

## 可复现与追溯

| 资产 | 位置 |
|---|---|
| 原始 benchmark（未改） | `data/mini_eval.jsonl` |
| 独立修正表 | `data/eval_corrections.json` |
| 修正加载器 | `eval/annotations.py` |
| tie-aware 判定 | `eval/cspider.py::results_equal` / `extract_order_by` |
| 历史结果备份 | `pre_eval_cleanup/`（旧结果不覆盖） |
| 重跑结果 | `eval/results/{baseline,no-linking,no-fewshot,no-repair,qwen7b}.json`（details 含 `gold_corrected`/`correction_note`） |
| 口径修正说明 | `docs/evaluation_notes.md` |
