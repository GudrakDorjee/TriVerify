#!/usr/bin/env python3
"""Step 5: 增强前后对比评估"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu.metrics import CHRF

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
LORA_V2 = f'{BASE_DIR}/saves/qwen_lora_v2'
LORA_V3 = f'{BASE_DIR}/saves/qwen_lora_v3_augmented'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/step5_augmentation'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

def load_model(lora_path):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE, quantization_config=quant_config,
        device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.9, do_sample=True, repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    gen = outputs[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    for m in ['**解释', '解释：', '注释：', '\n\n---']:
        if m in text:
            text = text[:text.index(m)]
    return re.sub(r'\*\*(.+?)\*\*', r'\1', text).strip()

def evaluate_model(model, tokenizer, test_data, terminology, directions):
    """评估模型在多个方向上的表现"""
    chrf_scorer = CHRF(word_order=2)
    results = {}

    for sl, tl in directions:
        preds, refs = [], []
        term_total, term_correct = 0, 0
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[tl]

        for entry in tqdm(test_data, desc=f"{sl}->{tl}", leave=False):
            sn, tn = LANG_NAMES[sl], LANG_NAMES[tl]
            user_msg = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。\n\n{sn}：{entry[LANG_FIELDS[sl]]}"
            messages = [{"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
                       {"role": "user", "content": user_msg}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pred = generate(model, tokenizer, prompt)
            preds.append(pred)
            refs.append(entry[LANG_FIELDS[tl]])

            for sk, info in terminology.items():
                if sk not in entry[LANG_FIELDS[sl]]:
                    continue
                exp = sk if tl == 'sk' else info.get(target_key, '')
                if not exp or exp not in entry[LANG_FIELDS[tl]]:
                    continue
                term_total += 1
                if exp in pred:
                    term_correct += 1

        chrf = chrf_scorer.corpus_score(preds, [refs]).score
        ta = term_correct / term_total if term_total > 0 else 0
        results[f"{sl}->{tl}"] = {'chrf': chrf, 'term_acc': ta}

    return results

def main():
    print("=" * 70)
    print("📊 Step 5: 增强前后对比评估")
    print("=" * 70)

    # 加载数据
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:50]

    with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
        terminology = json.load(f)

    directions = [('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb'), ('cn', 'tb'), ('cn', 'sk'), ('tb', 'sk')]

    # 评估 v2（增强前）
    print("\n🔄 评估 v2 (增强前)...")
    model_v2, tok_v2 = load_model(LORA_V2)
    results_v2 = evaluate_model(model_v2, tok_v2, test_data, terminology, directions)
    del model_v2, tok_v2
    torch.cuda.empty_cache()

    # 评估 v3（增强后）
    if Path(LORA_V3).exists():
        print("\n🔄 评估 v3 (增强后)...")
        model_v3, tok_v3 = load_model(LORA_V3)
        results_v3 = evaluate_model(model_v3, tok_v3, test_data, terminology, directions)
        del model_v3, tok_v3
        torch.cuda.empty_cache()
    else:
        print(f"\n⚠️  v3 模型不存在: {LORA_V3}")
        results_v3 = None

    # 打印对比
    print("\n" + "=" * 70)
    print("📊 增强前后对比")
    print("=" * 70)
    print(f"\n{'方向':<8} {'v2 chrF++':>10} {'v3 chrF++':>10} {'增益':>7} | {'v2 TermAcc':>10} {'v3 TermAcc':>10} {'增益':>7}")
    print("-" * 75)

    for d in [f"{sl}->{tl}" for sl, tl in directions]:
        v2 = results_v2.get(d, {})
        v3 = results_v3.get(d, {}) if results_v3 else {}

        v2c = v2.get('chrf', 0)
        v3c = v3.get('chrf', 0)
        v2t = v2.get('term_acc', 0)
        v3t = v3.get('term_acc', 0)

        cg = f"+{v3c-v2c:.2f}" if v3c >= v2c else f"{v3c-v2c:.2f}"
        tg = f"+{v3t-v2t:.4f}" if v3t >= v2t else f"{v3t-v2t:.4f}"

        if results_v3:
            print(f"{d:<8} {v2c:>10.2f} {v3c:>10.2f} {cg:>7} | {v2t:>10.4f} {v3t:>10.4f} {tg:>7}")
        else:
            print(f"{d:<8} {v2c:>10.2f} {'N/A':>10} {'N/A':>7} | {v2t:>10.4f} {'N/A':>10} {'N/A':>7}")

    # 保存结果
    compare_results = {'v2': results_v2, 'v3': results_v3}
    with open(f'{RESULTS_DIR}/augmentation_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(compare_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 对比结果: {RESULTS_DIR}/augmentation_comparison.json")
    print("\n✅ Step 5 对比评估完成")

if __name__ == "__main__":
    main()
