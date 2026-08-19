#!/usr/bin/env python3
"""
Step 6: 综合评估
1. 完整测试集 (1855条) 评估
2. 案例分析（成功/失败案例）
3. 生成论文用表格和图表
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA_V2 = f'{BASE_DIR}/saves/qwen_lora_v2'
QWEN_LORA_V3 = f'{BASE_DIR}/saves/qwen_lora_v3_augmented'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/step6_full'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class FullEvaluator:
    def __init__(self, lora_path):
        with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
            self.terminology = json.load(f)
        self.bleu_scorer = BLEU(effective_order=True)
        self.chrf_scorer = CHRF(word_order=2)

        print(f"Loading model with LoRA: {lora_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()
        print(f"Done, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, top_p=0.9, do_sample=True, repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        gen = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        for m in ['**解释', '解释：', '注释：', '\n\n---']:
            if m in text:
                text = text[:text.index(m)]
        return re.sub(r'\*\*(.+?)\*\*', r'\1', text).strip()

    def _format_prompt(self, user_message):
        messages = [{"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
                    {"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def translate_direct(self, text, source_lang, target_lang):
        sn, tn = LANG_NAMES[source_lang], LANG_NAMES[target_lang]
        terms = self._get_terms(text)
        prompt = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt += f"\n术语：{terms}"
        prompt += f"\n\n{sn}：{text}"
        return self.generate(self._format_prompt(prompt))

    def translate_mutual(self, text, source_lang, target_lang, pivot_text, pivot_lang):
        sn, tn, pn = LANG_NAMES[source_lang], LANG_NAMES[target_lang], LANG_NAMES[pivot_lang]
        terms = self._get_terms(text)
        prompt = f"请将以下{sn}翻译为{tn}，参考{pn}辅助理解。只输出翻译。"
        if terms:
            prompt += f"\n术语：{terms}"
        prompt += f"\n\n{sn}：{text}\n{pn}参考：{pivot_text}"
        return self.generate(self._format_prompt(prompt))

    def _get_terms(self, text):
        relevant = []
        for sk, info in self.terminology.items():
            if sk in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
                relevant.append(f"{sk}={info.get('chinese','?')}/{info.get('tibetan','?')}")
        return '；'.join(relevant[:5]) if relevant else ''

    def compute_term_accuracy(self, predictions, references, sources, target_lang):
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
        total, correct = 0, 0
        for pred, ref, src in zip(predictions, references, sources):
            for sk, info in self.terminology.items():
                if sk not in src and info.get('tibetan', '') not in src and info.get('chinese', '') not in src:
                    continue
                exp = sk if target_lang == 'sk' else info.get(target_key, '')
                if not exp or exp not in ref:
                    continue
                total += 1
                if exp in pred:
                    correct += 1
        return correct / total if total > 0 else 0

    def evaluate_direction(self, test_data, source_lang, target_lang, method='direct'):
        """评估单个方向"""
        pivot_lang = list({'sk', 'tb', 'cn'} - {source_lang, target_lang})[0]
        preds, refs, srcs = [], [], []

        for entry in tqdm(test_data, desc=f"{source_lang}->{target_lang} ({method})", leave=False):
            src = entry[LANG_FIELDS[source_lang]]
            ref = entry[LANG_FIELDS[target_lang]]

            if method == 'direct':
                pred = self.translate_direct(src, source_lang, target_lang)
            elif method == 'mutual':
                pivot_text = entry[LANG_FIELDS[pivot_lang]]
                pred = self.translate_mutual(src, source_lang, target_lang, pivot_text, pivot_lang)
            else:
                pred = self.translate_direct(src, source_lang, target_lang)

            preds.append(pred)
            refs.append(ref)
            srcs.append(src)

        # 计算指标
        bleu = self.bleu_scorer.corpus_score(preds, [refs]).score
        chrf = self.chrf_scorer.corpus_score(preds, [refs]).score
        term_acc = self.compute_term_accuracy(preds, refs, srcs, target_lang)

        return {
            'bleu': bleu, 'chrf': chrf, 'term_acc': term_acc,
            'n_samples': len(preds),
            'predictions': preds, 'references': refs, 'sources': srcs
        }

    def find_case_studies(self, preds, refs, srcs, entries, source_lang, target_lang, n=5):
        """找出成功和失败案例"""
        cases = []
        for i, (pred, ref, src) in enumerate(zip(preds, refs, srcs)):
            # 句级 chrF++
            sent_chrf = self.chrf_scorer.sentence_score(pred, [ref]).score
            cases.append({
                'id': entries[i]['id'],
                'source': src[:100],
                'prediction': pred[:150],
                'reference': ref[:150],
                'chrf': sent_chrf,
            })

        cases.sort(key=lambda x: x['chrf'], reverse=True)

        return {
            'best': cases[:n],
            'worst': cases[-n:],
        }

def run_full_evaluation(max_samples=None):
    """运行完整评估"""
    # 加载完整测试集
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    print("=" * 70)
    print(f"🚀 Step 6: 综合评估 ({len(test_data)} 条测试集)")
    print("=" * 70)

    # 使用增强后模型 v3
    lora_path = QWEN_LORA_V3 if Path(QWEN_LORA_V3).exists() else QWEN_LORA_V2
    print(f"使用模型: {lora_path}")

    evaluator = FullEvaluator(lora_path)

    directions = [
        ('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb'),
        ('cn', 'tb'), ('cn', 'sk'), ('tb', 'sk'),
    ]

    all_results = {}
    case_studies = {}

    for sl, tl in directions:
        d = f"{sl}->{tl}"
        print(f"\n{'='*60}")
        print(f"📌 {d}")

        # 直接翻译
        res_direct = evaluator.evaluate_direction(test_data, sl, tl, 'direct')
        all_results[f"{d}_direct"] = {
            'bleu': res_direct['bleu'], 'chrf': res_direct['chrf'], 'term_acc': res_direct['term_acc']
        }

        # 互证翻译
        res_mutual = evaluator.evaluate_direction(test_data, sl, tl, 'mutual')
        all_results[f"{d}_mutual"] = {
            'bleu': res_mutual['bleu'], 'chrf': res_mutual['chrf'], 'term_acc': res_mutual['term_acc']
        }

        # 打印结果
        print(f"  直接: BLEU={res_direct['bleu']:.2f}, chrF++={res_direct['chrf']:.2f}, TermAcc={res_direct['term_acc']:.4f}")
        print(f"  互证: BLEU={res_mutual['bleu']:.2f}, chrF++={res_mutual['chrf']:.2f}, TermAcc={res_mutual['term_acc']:.4f}")

        # 案例分析（用互证结果）
        cases = evaluator.find_case_studies(
            res_mutual['predictions'], res_mutual['references'],
            res_mutual['sources'], test_data, sl, tl
        )
        case_studies[d] = cases

        # 保存详细结果
        detail_path = f"{RESULTS_DIR}/{sl}_to_{tl}_results.jsonl"
        with open(detail_path, 'w', encoding='utf-8') as f:
            for i in range(len(test_data)):
                f.write(json.dumps({
                    'id': test_data[i]['id'],
                    'source_lang': sl, 'target_lang': tl,
                    'source': res_mutual['sources'][i],
                    'reference': res_mutual['references'][i],
                    'pred_direct': res_direct['predictions'][i],
                    'pred_mutual': res_mutual['predictions'][i],
                }, ensure_ascii=False) + '\n')

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 综合评估汇总")
    print("=" * 70)

    print(f"\n{'方向':<10} {'方法':<8} {'BLEU':>6} {'chrF++':>7} {'TermAcc':>8}")
    print("-" * 45)
    for d in [f"{sl}->{tl}" for sl, tl in directions]:
        for method in ['direct', 'mutual']:
            k = f"{d}_{method}"
            r = all_results[k]
            print(f"{d:<10} {method:<8} {r['bleu']:>6.2f} {r['chrf']:>7.2f} {r['term_acc']:>8.4f}")
        print("-" * 45)

    # 平均值
    direct_chrf = np.mean([all_results[f"{sl}->{tl}_direct"]['chrf'] for sl, tl in directions])
    mutual_chrf = np.mean([all_results[f"{sl}->{tl}_mutual"]['chrf'] for sl, tl in directions])
    direct_term = np.mean([all_results[f"{sl}->{tl}_direct"]['term_acc'] for sl, tl in directions])
    mutual_term = np.mean([all_results[f"{sl}->{tl}_mutual"]['term_acc'] for sl, tl in directions])

    print(f"\n平均值:")
    print(f"  直接: chrF++={direct_chrf:.2f}, TermAcc={direct_term:.4f}")
    print(f"  互证: chrF++={mutual_chrf:.2f}, TermAcc={mutual_term:.4f}")
    print(f"  增益: chrF++={mutual_chrf-direct_chrf:+.2f}, TermAcc={mutual_term-direct_term:+.4f}")

    # ============================================================
    # 案例分析
    # ============================================================
    print("\n" + "=" * 70)
    print("📋 案例分析")
    print("=" * 70)

    for d in ['sk->cn', 'tb->cn']:
        if d not in case_studies:
            continue
        cases = case_studies[d]

        print(f"\n--- {d} 最佳翻译 (Top 3) ---")
        for c in cases['best'][:3]:
            print(f"  [{c['id']}] chrF++={c['chrf']:.1f}")
            print(f"    源: {c['source'][:80]}")
            print(f"    译: {c['prediction'][:80]}")
            print(f"    参: {c['reference'][:80]}")
            print()

        print(f"--- {d} 最差翻译 (Bottom 3) ---")
        for c in cases['worst'][:3]:
            print(f"  [{c['id']}] chrF++={c['chrf']:.1f}")
            print(f"    源: {c['source'][:80]}")
            print(f"    译: {c['prediction'][:80]}")
            print(f"    参: {c['reference'][:80]}")
            print()

    # 保存所有结果
    with open(f'{RESULTS_DIR}/full_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(f'{RESULTS_DIR}/case_studies.json', 'w', encoding='utf-8') as f:
        json.dump(case_studies, f, ensure_ascii=False, indent=2)

    del evaluator
    torch.cuda.empty_cache()

    print(f"\n💾 结果保存: {RESULTS_DIR}/")
    print("✅ Step 6 综合评估完成")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_full_evaluation(max_samples=n)
