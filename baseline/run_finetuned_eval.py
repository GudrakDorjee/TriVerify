#!/usr/bin/env python3
"""微调后模型评估推理"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
MODEL_PATH = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
LORA_PATH = f'{BASE_DIR}/saves/qwen_lora_v2'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/finetuned'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

# 加载数据
with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f][:50]

with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
    terminology = json.load(f)

print('=' * 60)
print('🚀 微调后模型评估 (Qwen2.5-7B + LoRA)')
print('=' * 60)
print(f'测试集: {len(test_data)} 条')

# 加载微调模型
print('🔄 加载基座模型 + LoRA...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print(f'✓ 加载完成, 显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

def generate(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def get_terms(text):
    relevant = []
    for sk_term, info in terminology.items():
        if sk_term in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
            relevant.append(f"{sk_term}={info.get('chinese', '?')}/{info.get('tibetan', '?')}")
    return '；'.join(relevant[:4]) if relevant else ''

def build_prompt(entry, source_lang, target_lang, pivot_lang=None):
    source_text = entry[LANG_FIELDS[source_lang]]
    source_name = LANG_NAMES[source_lang]
    target_name = LANG_NAMES[target_lang]
    terms = get_terms(source_text)

    if pivot_lang:
        pivot_text = entry[LANG_FIELDS[pivot_lang]]
        pivot_name = LANG_NAMES[pivot_lang]
        user_msg = f"请将以下{source_name}翻译为{target_name}，参考{pivot_name}辅助理解。只输出翻译。"
        if terms:
            user_msg += f"\n术语：{terms}"
        user_msg += f"\n\n{source_name}：{source_text}\n{pivot_name}参考：{pivot_text}"
    else:
        user_msg = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
        if terms:
            user_msg += f"\n术语：{terms}"
        user_msg += f"\n\n{source_name}：{source_text}"

    messages = [
        {"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
        {"role": "user", "content": user_msg}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def post_process(text):
    for marker in ['**解释', '**注释', '\n\n---', '解释：', '注释：']:
        if marker in text:
            text = text[:text.index(marker)]
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return text.strip()

# 实验计划：六方向直接 + 六方向互证
experiments = [
    ('sk', 'cn', None, 'finetuned_direct'),
    ('tb', 'cn', None, 'finetuned_direct'),
    ('sk', 'tb', None, 'finetuned_direct'),
    ('cn', 'tb', None, 'finetuned_direct'),
    ('cn', 'sk', None, 'finetuned_direct'),
    ('tb', 'sk', None, 'finetuned_direct'),
    ('sk', 'cn', 'tb', 'finetuned_mutual'),
    ('tb', 'cn', 'sk', 'finetuned_mutual'),
    ('sk', 'tb', 'cn', 'finetuned_mutual'),
    ('cn', 'tb', 'sk', 'finetuned_mutual'),
    ('cn', 'sk', 'tb', 'finetuned_mutual'),
    ('tb', 'sk', 'cn', 'finetuned_mutual'),
]

for source, target, pivot, method in experiments:
    direction = f"{source}→{target}" + (f"(via {pivot})" if pivot else "")
    print(f"\n--- {direction} [{method}] ---")

    results = []
    for entry in tqdm(test_data, desc=direction, leave=False):
        prompt = build_prompt(entry, source, target, pivot)
        prediction = generate(prompt)
        prediction = post_process(prediction)

        results.append({
            'id': entry['id'],
            'source_lang': source,
            'target_lang': target,
            'source_text': entry[LANG_FIELDS[source]],
            'reference': entry[LANG_FIELDS[target]],
            'prediction': prediction,
            'model': 'qwen_lora_v2',
            'method': method
        })

    # 保存
    suffix = "mutual" if pivot else "direct"
    filename = f"qwen_lora_{source}_to_{target}_{suffix}.jsonl"
    output_path = f"{RESULTS_DIR}/{filename}"
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 快速预览
    print(f"  预测: {results[0]['prediction'][:80]}")
    print(f"  参考: {results[0]['reference'][:80]}")
    print(f"  💾 {filename}")

del model, base_model, tokenizer
torch.cuda.empty_cache()
print('\n✅ 微调后评估推理全部完成')
print(f'结果目录: {RESULTS_DIR}')
