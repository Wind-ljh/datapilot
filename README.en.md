# DataPilot 📊 — Chinese ChatBI Data Analysis Agent with MCP + LangGraph

> [中文文档](README.md) ｜ Ask questions in natural language → automatic schema linking → SQL generation with self-repair → execution → charts & conclusions.

A complete engineering project covering **Text-to-SQL, hybrid-retrieval RAG, the MCP tool protocol, LangGraph multi-stage agents, QLoRA fine-tuning, and systematic evaluation**.

## Highlights

- **Workflow backbone + local agent loop**: a fixed 5-stage pipeline (clarify → link → generate → execute → report); control is handed back to the LLM only when SQL validation/execution fails (self-repair, ≤3 rounds).
- **Two unconventional RAG uses**: retrieval targets are ① schema metadata (schema linking via BM25 + vector search, RRF fusion) and ② similar question–SQL pairs (few-shot), not documents.
- **MCP tool layer**: 5 database tools implemented once (`ToolCore`), exposed both in-process and as a standard FastMCP stdio server.
- **Read-only safety**: statement count / keyword checks + `EXPLAIN` pre-flight — errors surface without touching data, feeding the repair loop.
- **Systematic evaluation**: execution accuracy (EX), exec rate, table recall@k, repair success rate, P50/P95 latency, tokens/query — each with one-flag ablation (`--no-linking`, `--no-fewshot`, `--max-repair 0`, `--llm llm2`).
- **Vendor-neutral & offline-friendly**: OpenAI-compatible endpoints (GLM-4-Flash / SiliconFlow / self-hosted vLLM) switched by env vars; deterministic mock LLM/embedder makes CI run with zero API keys.

## Quick Start

```bash
pip install -e .
python -m server.core.seed_data --rows 100000   # generate the demo e-commerce DuckDB

# Without any API key (mock mode)
DATAPILOT_MOCK=1 streamlit run app/streamlit_app.py

# With a free GLM-4-Flash key
cp .env.example .env   # fill DATAPILOT_LLM_API_KEY
streamlit run app/streamlit_app.py
```

## Evaluation & Ablation

```bash
python -m eval.mini_runner --label baseline       # 40-question built-in benchmark
python -m eval.mini_runner --no-linking --label no-linking
python -m eval.runner --split dev --limit 300     # CSpider (manual download, see eval/download_cspider.py)
```

## Fine-tuning

```bash
python -m training.build_sft_data                 # CSpider train → chat SFT data (db-level split)
python -m training.train_qlora                    # QLoRA on free GPU (ModelScope / Colab)
python -m training.merge_export                   # merge adapter → vLLM-ready
```

See [docs/fine-tuning.md](docs/fine-tuning.md) (Chinese) for the full free-tier recipe.

## Tests

```bash
pip install -e ".[dev]"
pytest        # 29 tests, fully offline via mock mode
ruff check server eval training app tests
```

MIT License.
