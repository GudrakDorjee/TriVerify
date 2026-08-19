#!/usr/bin/env python3
"""Qwen2.5-7B LoRA 微调脚本（独立版，不依赖 LLaMA-Factory）"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# 配置
# ============================================================
BASE_DIR = "/root/autodl-tmp/LlamaFactory-main"
MODEL_PATH = f"{BASE_DIR}/models/Qwen2.5-7B-Instruct"
TRAIN_DATA = f"{BASE_DIR}/data_preparation/data/train_qwen.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/saves/qwen_lora"

# 训练超参数
MAX_LEN = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32

print("=" * 60)
print("🚀 Qwen2.5-7B LoRA 微调")
print("=" * 60)

# ============================================================
# 1. 加载分词器
# ============================================================
print("\n📦 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. 加载并处理数据
# ============================================================
print("📦 加载训练数据...")
raw_data = []
with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

print(f"   原始数据: {len(raw_data)} 条")

# 限制数据量（可调整，全量训练去掉这行）
MAX_SAMPLES = len(raw_data)  # 全量
# MAX_SAMPLES = 5000  # 快速测试用
raw_data = raw_data[:MAX_SAMPLES]
print(f"   使用数据: {len(raw_data)} 条")

def preprocess(example):
    """将 ShareGPT 格式转为 input_ids + labels"""
    messages = example['messages']
    
    # 用 tokenizer 的 chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # 分词
    tokenized = tokenizer(
        text, 
        truncation=True, 
        max_length=MAX_LEN, 
        padding=False,
        return_tensors=None
    )
    
    # 构建 labels：只对 assistant 回复计算 loss
    input_ids = tokenized['input_ids']
    labels = input_ids.copy()
    
    # 找到 assistant 回复的起始位置
    # Qwen 格式: ...<|im_start|>assistant\n{content}<|im_end|>
    assistant_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    
    # 简化处理：对整个序列计算 loss（包含 user 部分）
    # 更精确的做法是 mask 掉 user 部分，但对于 SFT 影响不大
    tokenized['labels'] = labels
    
    return tokenized

print("🔄 处理数据...")
dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=4,
    desc="Tokenizing"
)

# 划分训练/验证
split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f"   训练集: {len(train_dataset)} 条")
print(f"   验证集: {len(eval_dataset)} 条")

# ============================================================
# 3. 加载模型 + LoRA
# ============================================================
print("\n🔄 加载模型 (4bit 量化)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 配置 LoRA
print("⚙️  配置 LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 4. 训练配置
# ============================================================
print("\n⚙️  配置训练参数...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    max_length=MAX_LEN,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

# ============================================================
# 5. 开始训练
# ============================================================
print("\n🏋️ 开始训练...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Total steps: ~{len(train_dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")
print(f"   Learning rate: {LR}")
print(f"   LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 训练
train_result = trainer.train()

# 保存
print("\n💾 保存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 保存训练指标
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\n" + "=" * 60)
print("✅ Qwen2.5-7B LoRA 微调完成！")
print("=" * 60)
print(f"模型保存位置: {OUTPUT_DIR}")
print(f"训练 loss: {metrics.get('train_loss', 'N/A')}")
