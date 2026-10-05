# DataPilot 最终发布报告（Final Release Report）

> 本文是「简历级收尾」的最终交付物：说明收尾做了什么、最终状态是什么、哪些数字可辩护。
> 全程遵循四条铁律：**不改 benchmark、不覆盖历史结果、不编造数字、不把未执行代码说成已完成**。

---

## 1. 项目概览

DataPilot 是一个中文 ChatBI / Text-to-SQL 助手：中文自然语言 → 自动选表选列 → 生成并自修复 SQL → 只读执行 → 输出图表与结论。

覆盖六大组件，最终状态如下：

| 组件 | 状态 |
|---|---|
| Text-to-SQL（LangGraph 多阶段 Agent） | ✅ 已完成（真实 LLM 端到端跑通） |
| 混合检索 RAG（BM25 + 向量 + RRF） | ✅ 已完成（Schema Linking + Few-shot，有单测） |
| MCP 工具协议（FastMCP） | ✅ 已完成（5 工具，进程内直调 + stdio，`mcp_smoke.py` 验证） |
| 系统化评估 | ✅ 已完成（EX / 可执行率 / 修复率 / P50/P95 / Token，5 组消融） |
| Streamlit 演示 | ✅ 已完成（真实查询跑通，见 §5 演示） |
| QLoRA 微调 | ⚠️ 仅代码、未训练（无 adapter、无结果） |

---

## 2. 审计结论（收尾前的全面体检）

收尾前对全仓库做了只读审计，结论：

- **5/6 组件真实完成**：代码 + 单测 + 离线可跑；QLoRA 为「代码完整但从未执行」。
- **README 有 3 处展示问题**：① `YOUR_USERNAME` 假 CI badge；② `docs/demo.gif` 断链（文件不存在）；③ 结果表缺 P95 列、QLoRA 行写「待填」。
- **eval/results/ 混杂**：5 组官方结果 + 7 个 scratch 文件（含一个 `label="baseline"` 但 EX=0.65 的混淆文件）。
- **实验数字与 README 一致**：EX 0.800 / 0.800 / 0.400 / 0.700 / 0.850 全部来自真实 JSON。
- **离线验证全绿**：pytest 41 通过、ruff 0 告警、演示库真实（10 万订单）。

---

## 3. README 与展示修复

| 问题 | 修复 |
|---|---|
| `YOUR_USERNAME` 假 CI badge | 删除；改为纯文本说明 CI 工作流已就绪但尚未推送到公开 GitHub 仓库（**不虚构 badge/URL**） |
| `docs/demo.gif` 断链 | 改为真实运行截图 `docs/demo.png` |
| 结果表缺 P95 | 补 P95 列（真实 JSON 值） |
| QLoRA 行「待填」 | 改为「未训练」 |
| 缺项目状态 | 新增「✅ 项目状态」表，明示 QLoRA 仅代码未训练 |
| 延迟脚注 | 扩展为 P50/P95 双重免责（限流噪声，非计算速度） |

---

## 4. 实验结果最终一致性

5 组消融的最终结果（mini 评测 40 题、DuckDB 只读、修正后评估口径），数字直接取自 `eval/results/*.json`：

| 配置 | EX | 可执行率 | 自修复成功率 | P50 | P95 | 平均 Token |
|---|---:|---:|---:|---:|---:|---:|
| baseline（GLM-4.7-Flash 全量） | **0.800**（32/40） | 0.975 | 0.833 | 8.4s | 39.7s | 1294 |
| − schema linking | 0.800 | 1.000 | 1.000 | 40.6s | 266.2s | 1224 |
| − few-shot | 0.400 | 0.975 | 0.750 | 119.8s | 307.9s | 1070 |
| − 自修复 | 0.700 | 0.825 | — | 198.0s | 424.3s | 1079 |
| 对照 Qwen2.5-7B | **0.850** | 0.950 | 0.333 | 6.6s | 14.1s | 1322 |

**五条研究结论**（可辩护，详见 [final_evaluation_report.md](final_evaluation_report.md)）：

1. 全量流水线 EX 上限 = **0.800（32/40）**；
2. schema linking 在小库上无增益（0.800 vs 0.800）；
3. few-shot 是最大增益来源（0.800 → 0.400，−0.400）；
4. 自修复是第二增益来源（0.800 → 0.700，可执行率 0.975 → 0.825）；
5. 免费 Qwen2.5-7B 在公平口径下 ≥ baseline（0.850 vs 0.800，差距在单轮噪声内）。

> **诚实声明**：P50/P95 两列受免费档 API 限流（zhipu 1305）随机影响，no-linking/no-fewshot/no-repair 的延迟被抬到 40–424s，**不代表模型计算速度**，仅作记录、不可横向比较。未限流轮次 P50≈6–8s。

---

## 5. 演示（Demo）

- **演示界面**：`app/streamlit_app.py`，Streamlit，含侧边栏 6 个示例问题、聊天式交互、SQL 展开、图表（bar/line/area）、运行指标。
- **真实查询验证**：在真实 API Key（GLM-4.7-Flash + SiliconFlow BGE-M3）下跑通问题「各品类的销售收入排名」，结果：

  - **SQL**：3 表 JOIN（`order_items ⋈ products ⋈ categories`），`SUM(quantity*unit_price)` 聚合，`ORDER BY … DESC`；
  - **执行**：成功（`executed=true`），返回 10 行（=10 个品类，按收入降序），图表选型 `bar`；
  - **结论**：服饰鞋包以 1.27 亿元居首，报告给出头部集中、梯队分化的分析；
  - **Schema Linking** 正确命中 `products / order_items / categories` 三表。

- **截图**：`docs/demo.png`（真实查询结果的**可视化**：真实 GLM-4.7-Flash 生成 SQL + DuckDB 真实执行行，非 mock、非静态假数据、非伪造 UI）。
  - 说明：`app/streamlit_app.py` 界面已实际启动并通过浏览器验证（侧边栏 6 示例问题、标题、输入框均正常渲染）；由于捕获瞬间免费档 API 限流（zhipu 1305）使界面上实时查询长时间处于「分析中」，最终截图改为**用已捕获的同一次真实查询结果**（`pre_eval_cleanup/scratch_2026-10-06/demo_real_query_2026-10-06.json`，`executed=true`、10 行、bar）渲染成图，与 Streamlit 渲染同一份真实数据。
  - 录制动画 GIF 的方法仍保留在 [docs/experiments.md](experiments.md#5-演示-gif-录制)（Roadmap 中 GIF 项保持未勾选）。

---

## 6. 测试与工程卫生

- **单测**：`pytest` → **41 通过**（mock 模式完全离线，可复现）。
- **Lint**：`ruff check .` → **0 告警**（E/F/I/UP/B，忽略 E501；中文注释拼写不在清理范围）。
- **CI**：`.github/workflows/ci.yml`（ruff + seed_data + pytest，Python 3.10/3.12，mock 模式）——文件真实存在，**因非公开仓库不展示 badge**。
- **结果目录**：`eval/results/` 已清理为**仅 5 组官方结果**；7 个 scratch 文件（smoke*.json / mock-baseline / baseline_clarify-only / rerun.log）与一次 demo 追踪 JSON 已归档到 `pre_eval_cleanup/scratch_2026-10-06/`。
- **路径校验**：README 引用的所有文件/目录全部存在（`.github/workflows/ci.yml`、`docs/*`、`server/*`、`eval/*`、`training/*`、`app/*`、`scripts/*`、`tests/`、`Dockerfile`、`docker-compose.yml`、`.env.example` 等）。
- **占位符扫描**：全仓库无 `YOUR_USERNAME / TODO / FIXME / fake / 待填` 残留（`resume-bullets.md` 保留显式标注的 `【数字】` 模板占位，并新增 QLoRA 未训练的提醒）。

---

## 7. 已知限制与诚实声明

| 项 | 状态 |
|---|---|
| QLoRA 微调 | **未训练**：`training/*` 脚本齐全，但无 adapter、无结果、无微调对比数字。README / resume-bullets 已明确标注，不当作已完成能力 |
| GitHub 仓库 | **尚未推送**：本目录非 git 仓库，未虚构用户名 / repository URL / badge |
| 延迟指标 | P50/P95 受免费档限流污染，不可横向比较（见 §4） |
| 演示动画 | 目前是**静态截图**（docs/demo.png），GIF 录制仍在 Roadmap（未勾选） |
| CSpider 权威基准 | 未下载数据、未跑 dev 300 题（Roadmap 未勾选） |
| 评估规模 | EX 结论基于 40 题 mini 评测；结论稳健性受单轮噪声（±0.10）约束 |

---

## 8. 最终交付清单

- [x] README：删除假 badge、修复断链、补 P95、QLoRA 行改为「未训练」、新增项目状态表、延迟脚注扩展
- [x] 演示：真实 API 跑通一条查询，产出真实结果可视化图 `docs/demo.png`
- [x] 实验：5 组结果数字与 README 完全一致，`eval/results/` 只剩官方结果
- [x] 测试卫生：pytest 41 通过、ruff 0 告警、CI 文件真实、scratch 归档
- [x] 诚实声明：QLoRA 未训练、非公开仓库、延迟噪声、静态截图——均已在 README / 本报告 / resume-bullets 中明示
- [x] 本报告：`docs/final_release_report.md`
