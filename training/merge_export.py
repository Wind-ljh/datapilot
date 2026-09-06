"""LoRA adapter 与基座合并导出（供 vLLM/ollama 部署）。

用法（训练机上执行）：
  python -m training.merge_export \
      --model Qwen/Qwen2.5-Coder-3B-Instruct \
      --adapter training/output/datapilot-qlora \
      --out training/output/datapilot-3b-merged
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    print(f"[merge_export] 已合并导出到 {args.out}")


if __name__ == "__main__":
    main()
