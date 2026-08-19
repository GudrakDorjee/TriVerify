#!/usr/bin/env python3
"""Step 2: 三模型基线推理脚本"""
import torch
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# 路径配置
BASE_DIR = "/root/autodl-tmp/LlamaFactory-main"
DATA_DIR = f"{BASE_DIR}/data_preparation/data"
RESULTS_DIR = f"{BASE_DIR}/baseline/results"

MODEL_PATHS = {
    'qwen': f"{BASE_DIR}/models/Qwen2.5-7B-Instruct",
    'gemma_mitra': f"{BASE_DIR}/models/gemma-2-mitra-e",
    'gemma_4b': f"{BASE_DIR}/models/googletranslategemma-4b-it",
}

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

def load_model(model_key: str):
    """加载模型，返回 (model, tokenizer, model_type)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_path = MODEL_PATHS[model_key]
    print(f"🔄 加载模型: {model_key} ({model_path})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    if model_key == 'gemma_4b':
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    
    model.eval()
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"✓ 加载完成, 显存: {mem:.1f} GB")
    
    return model, tokenizer

def format_prompt(model_key: str, tokenizer, user_message: str) -> str:
    """根据模型类型格式化prompt"""
    if model_key == 'qwen':
        messages = [
            {"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
            {"role": "user", "content": user_message}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    elif model_key == 'gemma_mitra':
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    
    elif model_key == 'gemma_4b':
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

def post_process(text: str, target_lang: str) -> str:
    """后处理：去除解释、只保留翻译"""
    # 去除 "解释：" "注释：" 等后续内容
    for marker in ['**解释', '**注释', '**考量', '\n\n---', '解释：', '注释：']:
        if marker in text:
            text = text[:text.index(marker)]
    
    # 去除 "译文：" 前缀
    for prefix in ['**译文：**', '译文：', '**译文:**', '翻译：']:
        if prefix in text:
            text = text[text.index(prefix) + len(prefix):]
    
    # 去除 markdown 格式
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    return text.strip()

# ============================================================
# Prompt 构建
# ============================================================

def build_zero_shot_prompt(entry: Dict, source_lang: str, target_lang: str,
                           terminology: Dict = None) -> str:
    """零样本prompt"""
    source_text = entry[LANG_FIELDS[source_lang]]
    source_name = LANG_NAMES[source_lang]
    target_name = LANG_NAMES[target_lang]
    
    # 术语提示
    term_hint = ""
    if terminology:
        relevant = []
        for sk_term, info in terminology.items():
            if sk_term in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                relevant.append(f"{sk_term}={info.get('chinese', '?')}")
        if relevant:
            term_hint = f"\n术语参考：{'；'.join(relevant[:5])}\n"
    
    prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体风格。只输出翻译结果，不要解释。{term_hint}\n{source_name}原文：\n{source_text}\n\n{target_name}翻译："
    return prompt

def build_few_shot_prompt(entry: Dict, source_lang: str, target_lang: str,
                          examples: List[Dict], terminology: Dict = None) -> str:
    """少样本prompt"""
    source_name = LANG_NAMES[source_lang]
    target_name = LANG_NAMES[target_lang]
    source_field = LANG_FIELDS[source_lang]
    target_field = LANG_FIELDS[target_lang]
    source_text = entry[source_field]
    
    # 术语提示
    term_hint = ""
    if terminology:
        relevant = []
        for sk_term, info in terminology.items():
            if sk_term in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                relevant.append(f"{sk_term}={info.get('chinese', '?')}")
        if relevant:
            term_hint = f"\n术语参考：{'；'.join(relevant[:5])}\n"
    
    # 构建示例
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n示例{i}：\n{source_name}：{ex[source_field]}\n{target_name}：{ex[target_field]}\n"
    
    prompt = (f"请将{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译结果。"
              f"{term_hint}\n{examples_text}\n"
              f"现在请翻译：\n{source_name}：{source_text}\n{target_name}：")
    return prompt

def build_mutual_prompt(entry: Dict, source_lang: str, target_lang: str,
                        pivot_lang: str, terminology: Dict = None) -> str:
    """互证prompt（引入第三语言辅助）"""
    source_text = entry[LANG_FIELDS[source_lang]]
    pivot_text = entry[LANG_FIELDS[pivot_lang]]
    source_name = LANG_NAMES[source_lang]
    target_name = LANG_NAMES[target_lang]
    pivot_name = LANG_NAMES[pivot_lang]
    
    term_hint = ""
    if terminology:
        relevant = []
        for sk_term, info in terminology.items():
            if sk_term in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                relevant.append(f"{sk_term}={info.get('chinese', '?')}")
        if relevant:
            term_hint = f"\n术语参考：{'；'.join(relevant[:5])}\n"
    
    prompt = (f"请将以下{source_name}翻译为{target_name}。参考{pivot_name}译本辅助理解语义。"
              f"只输出翻译结果，不要解释。{term_hint}\n"
              f"{source_name}原文：\n{source_text}\n\n"
              f"{pivot_name}参考：\n{pivot_text}\n\n"
              f"{target_name}翻译：")
    return prompt

# ============================================================
# 实验运行
# ============================================================

def run_experiment(model_key: str, test_data: List[Dict], 
                   source_lang: str, target_lang: str,
                   method: str = "zero_shot",
                   terminology: Dict = None,
                   train_data: List[Dict] = None,
                   pivot_lang: str = None,
                   n_shots: int = 3,
                   max_samples: int = None) -> List[Dict]:
    """运行单个实验"""
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"🧪 实验: {model_key} | {source_lang}→{target_lang} | {method}")
    print(f"   样本数: {len(test_data)}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer = load_model(model_key)
    
    results = []
    start_time = time.time()
    
    for entry in tqdm(test_data, desc=f"{model_key} {method}"):
        # 构建prompt
        if method == "zero_shot":
            user_msg = build_zero_shot_prompt(entry, source_lang, target_lang, terminology)
        elif method == "few_shot":
            import random
            examples = random.sample([e for e in train_data if e['id'] != entry['id']], 
                                     min(n_shots, len(train_data)))
            user_msg = build_few_shot_prompt(entry, source_lang, target_lang, examples, terminology)
        elif method == "mutual":
            user_msg = build_mutual_prompt(entry, source_lang, target_lang, pivot_lang, terminology)
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 格式化并生成
        prompt = format_prompt(model_key, tokenizer, user_msg)
        prediction = generate(model, tokenizer, prompt)
        prediction = post_process(prediction, target_lang)
        
        results.append({
            'id': entry['id'],
            'source_lang': source_lang,
            'target_lang': target_lang,
            'source_text': entry[LANG_FIELDS[source_lang]],
            'reference': entry[LANG_FIELDS[target_lang]],
            'prediction': prediction,
            'model': model_key,
            'method': method
        })
    
    elapsed = time.time() - start_time
    print(f"⏱️  耗时: {elapsed:.1f}s ({elapsed/len(test_data):.2f}s/条)")
    
    # 保存结果
    output_dir = Path(RESULTS_DIR) / method
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_key}_{source_lang}_to_{target_lang}.jsonl"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"💾 保存: {output_path}")
    
    # 卸载模型
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"🗑️  模型已卸载")
    
    return results

# ============================================================
# 主入口
# ============================================================

def quick_test(n_samples: int = 5):
    """快速测试"""
    print("=" * 60)
    print(f"⚡ 快速测试 ({n_samples} 条)")
    print("=" * 60)
    
    # 加载数据
    with open(f"{DATA_DIR}/splits/test.jsonl", 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:n_samples]
    
    with open(f"{DATA_DIR}/terminology_final.json", 'r', encoding='utf-8') as f:
        terminology = json.load(f)
    
    # 测试 Qwen（最稳定）
    results = run_experiment(
        model_key='qwen',
        test_data=test_data,
        source_lang='sk',
        target_lang='cn',
        method='zero_shot',
        terminology=terminology,
        max_samples=n_samples
    )
    
    # 展示结果
    print("\n📋 翻译结果:")
    for r in results:
        print(f"\n[{r['id']}]")
        print(f"  预测: {r['prediction'][:100]}")
        print(f"  参考: {r['reference'][:100]}")
    
    print("\n✅ 快速测试完成")

def run_full_baseline(max_samples: int = None):
    """运行完整基线实验"""
    print("=" * 60)
    print("🚀 Step 2: 完整基线实验")
    print("=" * 60)
    
    # 加载数据
    with open(f"{DATA_DIR}/splits/test.jsonl", 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    with open(f"{DATA_DIR}/splits/train.jsonl", 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    with open(f"{DATA_DIR}/terminology_final.json", 'r', encoding='utf-8') as f:
        terminology = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"⚠️  限制测试样本数: {max_samples}")
    
    print(f"测试集: {len(test_data)} 条")
    print(f"训练集: {len(train_data)} 条 (用于few-shot)")
    
    # 实验计划
    experiments = [
        # 零样本
        ('qwen', 'sk', 'cn', 'zero_shot', None),
        ('qwen', 'tb', 'cn', 'zero_shot', None),
        ('gemma_mitra', 'sk', 'cn', 'zero_shot', None),
        ('gemma_4b', 'sk', 'cn', 'zero_shot', None),
        
        # 少样本 (3-shot)
        ('qwen', 'sk', 'cn', 'few_shot', None),
        ('qwen', 'tb', 'cn', 'few_shot', None),
        
        # 互证 (引入第三语言)
        ('qwen', 'sk', 'cn', 'mutual', 'tb'),
        ('qwen', 'tb', 'cn', 'mutual', 'sk'),
    ]
    
    all_results = {}
    
    for model_key, source, target, method, pivot in experiments:
        try:
            results = run_experiment(
                model_key=model_key,
                test_data=test_data,
                source_lang=source,
                target_lang=target,
                method=method,
                terminology=terminology,
                train_data=train_data,
                pivot_lang=pivot,
                max_samples=max_samples
            )
            key = f"{method}_{model_key}_{source}_{target}"
            all_results[key] = len(results)
        except Exception as e:
            print(f"❌ 实验失败: {model_key} {source}→{target} {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 基线实验全部完成")
    print("=" * 60)
    for key, count in all_results.items():
        print(f"  {key}: {count} 条")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test(n_samples=5)
        elif sys.argv[1] == "full":
            run_full_baseline()
        elif sys.argv[1].isdigit():
            # 限制样本数的完整实验
            run_full_baseline(max_samples=int(sys.argv[1]))
        else:
            print("用法: python3 model_inference.py [test|full|<n_samples>]")
    else:
        quick_test(n_samples=3)
