# DataPilot 📊 — 基于 MCP + LangGraph 的中文 ChatBI 数据分析助手

> 用中文自然语言提问 → 自动选择表与字段 → 生成并自修复 SQL → 执行 → 输出图表与结论。
>
> 一个覆盖 **Text-to-SQL、混合检索 RAG、MCP 工具协议、LangGraph 多阶段 Agent、QLoRA 微调、系统化评估** 的完整工程项目。

[![CI](https://github.com/YOUR_USERNAME/datapilot/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

<!-- 演示 GIF：录制方法见 docs/experiments.md 末尾 -->
![demo](docs/demo.gif)

## ✨ 核心设计（也是面试讲点）

| 设计 | 说明 |
|---|---|
| **多阶段 Workflow + 局部 Agent 回环** | 主干流程固定（澄清→检索→生成→执行→报告，可控可测、省 Token），仅在 SQL 校验/执行失败时交还控制权给 LLM 自修复（≤3 轮）——Workflow 与 Agent 的取舍有明确判断 |
| **两处"非常规 RAG"** | 检索对象不是文档，而是 ① schema 元数据（Schema Linking，压缩候选表列）② 相似问-SQL 对（Few-shot 对齐输出风格）；BM25 + 向量双路召回，RRF 融合 |
| **MCP 工具协议** | 数据库能力封装为 5 个标准 MCP 工具（实现一次，进程内直调 + stdio server 两种接入方式），可被任何 MCP 客户端复用 |
| **只读安全防护** | 语句数/起始关键字/词边界危险关键字三重校验 + `EXPLAIN` 预检（不执行数据即可发现语法与列名错误），为自修复回环提供信号 |
| **系统化评估** | 执行准确率 EX、可执行率、表级 recall@k、自修复成功率、P50/P95 延迟、每查询 Token 消耗，全部支持一键消融对比（`--no-linking` / `--no-fewshot` / `--max-repair 0` / `--llm llm2`） |
| **零厂商绑定** | 所有 LLM/Embedding 走 OpenAI 兼容协议，GLM-4-Flash / SiliconFlow / 自建 vLLM 仅靠环境变量切换 |
| **离线可运行** | 内置确定性 Mock LLM 与 Mock Embedding，无 API Key 可跑通全链路、单测与 CI |

## 🏗 架构

```mermaid
flowchart TB
    U[用户中文问题] --> C[澄清 Agent<br/>缺维度/时间则追问]
    C -->|可回答| R[Schema Linking<br/>BM25+向量 RRF 选表选列]
    C -->|需澄清| U
    R --> F[Few-shot 检索<br/>相似问-SQL 对]
    F --> G[SQL 生成<br/>GLM-4-Flash / 微调模型]
    G --> V{EXPLAIN + 只读校验}
    V -->|失败| P[自修复 ≤3 轮<br/>错误信息回填] --> G
    V -->|通过| E[执行 SQL<br/>DuckDB 只读]
    E -->|报错| P
    E -->|成功| B[报告 Agent<br/>结论 + 图表选型]
    B --> O[结论 + 图表 + 指标]

    subgraph MCP 工具层（实现一次，两种接入）
        T1[list_tables] T2[get_schema] T3[sample_rows] T4[validate_sql] T5[execute_sql]
    end
    R -.->|进程内直调| T2
    E -.->|进程内直调| T5
```

## 🚀 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 生成中文电商演示库（约 10 秒，10 万行订单）
python -m server.core.seed_data --rows 100000

# 3a. 无 API Key？直接以 mock 模式体验（确定性演示 SQL）
set DATAPILOT_MOCK=1 && streamlit run app/streamlit_app.py

# 3b. 有 API Key（GLM-4-Flash 完全免费）？复制配置后体验完整链路
copy .env.example .env   # 填入 DATAPILOT_LLM_API_KEY
streamlit run app/streamlit_app.py

# HTTP API
uvicorn server.api.main:app --port 8000   # POST /api/query {"question": "每月订单量"}
```

## 📊 评估与消融

```bash
# 开箱即用的 mini 评测（40 道演示库金标题）
python -m eval.mini_runner --label baseline

# 消融实验（每条命令产出一个可对比的指标行）
python -m eval.mini_runner --no-linking   --label no-linking
python -m eval.mini_runner --no-fewshot   --label no-fewshot
python -m eval.mini_runner --max-repair 0 --label no-repair
python -m eval.mini_runner --llm llm2     --label qwen7b      # 对照模型

# CSpider 权威基准（中文 text-to-SQL，需手动下载数据）
python -m eval.download_cspider    # 镜像失效时打印官网下载指引
python -m eval.runner --split dev --limit 300 --label baseline
```

结果报告写入 `eval/results/*.json`，汇总方法见 [docs/experiments.md](docs/experiments.md)。

## 🔧 QLoRA 微调（免费算力可复现）

```bash
# ① 构造 SFT 数据（CSpider train → ChatBI 对话格式，按库划分 train/val 防泄漏）
python -m training.build_sft_data

# ② ModelScope 免费 GPU / Colab T4 上执行
python -m training.train_qlora --model Qwen/Qwen2.5-Coder-3B-Instruct

# ③ 合并导出 + vLLM 部署脚本
python -m training.merge_export
```

完整复现步骤（含免费算力申请）见 [docs/fine-tuning.md](docs/fine-tuning.md)。

## 📁 项目结构

```
datapilot/
├── server/
│   ├── core/          # config / LLM 客户端 / Embedding / DuckDB+SQLite 适配 / 种子数据
│   ├── mcp_tools/     # FastMCP 工具层（ToolCore 单一实现）
│   ├── retrieval/     # 混合检索（BM25+向量 RRF）/ Schema Linking / Few-shot
│   ├── agents/        # LangGraph 流水线：澄清→linking→生成→自修复回环→报告
│   └── api/           # FastAPI 入口
├── eval/              # mini 评测 / CSpider runner（EX+recall+修复率+延迟+Token）/ 数据下载
├── training/          # SFT 数据构造 / QLoRA 训练 / 合并导出
├── app/               # Streamlit 演示
├── tests/             # pytest（mock 模式离线全绿）
└── docs/              # 实验报告 / 简历话术 / 微调指南
```

## 📈 结果

> 跑通后用 `python -m eval.mini_runner --label <exp>` 的输出更新此表（方法见 docs/experiments.md）。

| 配置 | mini 评测 EX | 可执行率 | P50 延迟 | 平均 Token |
|---|---|---|---|---|
| baseline（GLM-4-Flash + 全量能力） | 待填 | 待填 | 待填 | 待填 |
| − schema linking | 待填 | 待填 | 待填 | 待填 |
| − few-shot | 待填 | 待填 | 待填 | 待填 |
| − 自修复 | 待填 | 待填 | 待填 | 待填 |
| 对照：Qwen2.5-7B（免费 API） | 待填 | 待填 | 待填 | 待填 |
| QLoRA 微调 Qwen2.5-Coder-3B（CSpider dev） | 待填 | — | — | — |

## 🗺 Roadmap

- [ ] Streamlit 演示 GIF
- [ ] CSpider dev 300 题完整消融数据
- [ ] LoRA adapter 发布到 ModelScope
- [ ] 幻觉检测：结果为空时的归因（schema 误选 vs 数据本身为空）
- [ ] 多轮上下文记忆（指代消解："那 6 月呢？"）

## License

MIT
