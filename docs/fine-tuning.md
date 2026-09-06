# QLoRA 微调复现指南（fine-tuning.md）

目标：在**零成本**（ModelScope 免费 GPU 或 Colab T4）完成 Qwen2.5-Coder-3B 的 QLoRA SFT，
得到可在 CSpider dev 上量化对比的 adapter，并走通 vLLM 部署路径。

## 0. 前置准备（本地）

```bash
# CSpider 数据（本地完成）
python -m eval.download_cspider      # 或按打印的指引手动下载到 data/cspider/

# 构造 SFT 数据（本地完成，无需 GPU）
python -m training.build_sft_data --data-dir data/cspider --out-dir training/data
# 输出 training/data/sft_train.jsonl 与 sft_val.jsonl（按 db_id 划分，防泄漏）
```

## 1. ModelScope 免费算力（推荐，国内直连）

1. 注册 modelscope.cn 并完成实名认证 → 进入「我的 Notebook」→ 选择免费 GPU 实例
   （A10/24G 级别，每日有时长额度，本任务单次训练约 1-2 小时）。
2. 上传整个项目目录（或 git clone 你的仓库），安装训练依赖：
   ```bash
   pip install -e ".[training]"
   ```
3. 训练（首次运行会自动从 ModelScope 魔搭拉取 Qwen2.5-Coder-3B-Instruct，国内无需科学上网）：
   ```bash
   python -m training.train_qlora \
       --model Qwen/Qwen2.5-Coder-3B-Instruct \
       --data training/data/sft_train.jsonl \
       --val  training/data/sft_val.jsonl \
       --out  training/output/datapilot-qlora \
       --epochs 2 --lora-r 16
   ```
   显存预算：3B 4bit + LoRA r=16 + batch 4 × grad-accum 4 ≈ 10-14GB。
4. 合并导出（同机执行）：
   ```bash
   python -m training.merge_export \
       --model Qwen/Qwen2.5-Coder-3B-Instruct \
       --adapter training/output/datapilot-qlora \
       --out training/output/datapilot-3b-merged
   ```
5. 回传 `training/output/datapilot-qlora/`（adapter，约几十 MB）到本地。

## 2. 备选：Google Colab T4

免费 T4（16GB）跑 3B 需要收紧参数：`--batch 1 --grad-accum 16`，其余同上。
注意 Colab 访问 HuggingFace/ModelScope 的网络问题，建议先在 ModelScope 侧尝试。

## 3. 微调效果评测（回到本地/任意可推理环境）

方式 A（免部署，推荐）：把 adapter 挂到 Ollama / transformers 推理脚本，抽 CSpider dev 100-300 题
用 `eval.runner` 的 EX 逻辑对比「基座 3B vs 微调 3B vs 7B API」。

方式 B（vLLM，面试加分）：
```bash
pip install vllm
vllm serve training/output/datapilot-3b-merged --port 8100
# 然后把 .env 的 DATAPILOT_LLM_BASE_URL 指向 http://127.0.0.1:8100/v1
# 即可复用同一套评估 runner 对比 —— 这就是"零厂商绑定"设计的价值
```

## 4. 发布

- adapter 上传 ModelScope Models（建一个 model repo，附训练配置与评测数字）；
- 在 README 结果表填入微调前后 EX 对比；
- docs/experiments.md 记录训练超参与 loss 曲线截图。

## 5. 常见坑

| 现象 | 原因 / 处理 |
|---|---|
| OOM | batch 降为 1、`--max-len 768`；确认 bitsandbytes 已装且是 4bit 加载 |
| loss 不降 | 检查 chat template 是否被 tokenizer.apply_chat_template 正确应用（本仓库已处理） |
| 评测 EX 与训练 val loss 脱节 | 正常现象；EX 才是目标指标，val loss 只看趋势 |
| LoRA 合并后输出退化 | 确认 merge 时 dtype=bf16 与训练一致，避免 fp16 上溢 |
