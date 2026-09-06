"""QLoRA SFT 训练脚本（transformers + peft + trl）。

适用环境：ModelScope 免费 GPU Notebook（A10/24G 级别即可）/ Colab T4。
用法：
  python -m training.train_qlora \
      --model Qwen/Qwen2.5-Coder-3B-Instruct \
      --data training/data/sft_train.jsonl \
      --val training/data/sft_val.jsonl \
      --out training/output/datapilot-qlora

显存预算（3B，4bit + LoRA r=16，batch 4 + grad-accum 4，seq 1024）：约 10-14GB。
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--data", default="training/data/sft_train.jsonl")
    parser.add_argument("--val", default="training/data/sft_val.jsonl")
    parser.add_argument("--out", default="training/output/datapilot-qlora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    data_files = {"train": args.data}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)

    def _to_text(example):
        # sharegpt conversations → 单条训练文本（ChatML 由 tokenizer 模板完成）
        msgs = []
        for turn in example["conversations"]:
            role = {"system": "system", "human": "user", "gpt": "assistant"}[turn["from"]]
            msgs.append({"role": role, "content": turn["value"]})
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(_to_text, remove_columns=[c for c in ds["train"].column_names if c != "text"])

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if args.val else "no",
        bf16=True,
        max_length=args.max_len,
        dataset_text_field="text",
        packing=False,
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds["train"], peft_config=lora)
    if args.val:
        trainer.eval_dataset = ds["validation"]
    trainer.train()
    trainer.save_model(args.out)  # 保存 adapter
    tokenizer.save_pretrained(args.out)
    print(f"[train_qlora] LoRA adapter 已保存到 {args.out}")


if __name__ == "__main__":
    main()
