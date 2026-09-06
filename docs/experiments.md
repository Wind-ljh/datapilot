# 实验报告（experiments.md）

> 本文档是简历数字的"弹药库"：每个数字都有可复现的命令支撑。
> 跑完实验后请把结果回填到 README 的结果表，并在此记录环境与观察。

## 1. 评估协议

- **EX（执行准确率）**：预测 SQL 与 gold SQL 在同一库执行，结果集一致记 1 分；
  两者都含 `ORDER BY` 时按有序序列比较，否则按多重集合比较（对齐 Spider 官方评测思路）。
- **可执行率**：预测 SQL 通过只读校验且执行成功的比例。
- **表级 recall@k**：gold 涉及的表是否全部出现在 Schema Linking 的 top-k（k=3）。
- **自修复成功率**：首次校验/执行失败的样本中，经 ≤N 轮修复后最终可执行的比例。
- **成本**：每查询平均 Token（prompt + completion）、P50/P95 端到端延迟。

## 2. 数据集

| 数据集 | 规模 | 用途 |
|---|---|---|
| mini_eval（内置，演示电商库） | 40 题 | 开箱即用回归测试 + 消融初筛 |
| CSpider dev（手动下载） | 官方 dev 集（建议先跑 300 题） | 权威基准，写进简历的数字 |

## 3. 消融矩阵（模板，跑完后回填）

| 实验 | 命令 | EX | 可执行率 | recall@k | 修复成功率 | P50/P95 (ms) | 平均Token |
|---|---|---|---|---|---|---|---|
| baseline | `python -m eval.mini_runner --label baseline` | | | — | | | |
| − schema linking | `... --no-linking --label no-linking` | | | — | | | |
| − few-shot | `... --no-fewshot --label no-fewshot` | | | — | | | |
| − 自修复 | `... --max-repair 0 --label no-repair` | | | — | | | |
| 对照模型 Qwen2.5-7B | `... --llm llm2 --label qwen7b` | | | — | | | |
| QLoRA 微调 3B（CSpider dev） | 见 docs/fine-tuning.md | | | — | | | |

### 观察与分析（示例框架，替换成你的真实发现）

- **去掉 schema linking 后 EX 下降 X pp**，主要失败模式从"选错列"变成"schema 过长导致列名幻觉"，
  且平均 Token 上升 Y%（全量 schema 拼进 prompt 的代价）。
- **few-shot 的贡献集中在 JOIN 类问题**：单表聚合基线已接近上限，多表 JOIN 是 few-shot 收益的来源。
- **自修复的边际收益递减**：第 1 轮修复挽回 a% 的可执行率，第 2、3 轮合计只挽回 b%，
  但 Token 成本线性增长 → 生产配置建议 max_repair=2。
- **3B 微调 vs 7B API**：微调后 3B 在 CSpider dev 上 EX 达到 Z，接近未微调 7B 的 W，
  而单查询成本降低约 N 倍 → "小模型+领域微调"路线的性价比结论。

## 4. 环境记录

- 日期 / API 供应商与模型版本 / 网络情况
- `pip freeze` 关键版本：langgraph、mcp、duckdb、openai
- 温度统一 0.0；每配置跑一遍（LLM 输出不稳定时建议跑 3 遍取均值，并记录方差）

## 5. 演示 GIF 录制

1. `streamlit run app/streamlit_app.py`
2. 使用 OBS / ScreenToGif 录制三问三答（趋势题、TopN 题、占比题）
3. 导出 `docs/demo.gif`（控制在 5MB 内，宽 1000px）
