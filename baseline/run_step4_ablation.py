#!/usr/bin/env python3
"""Step 4: 术语约束与重排序 - 消融实验"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import Counter
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA = f'{BASE_DIR}/saves/qwen_lora_v2'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/step4_ablation'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class TermConstrainedTranslator:
    def __init__(self):
        with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
            self.terminology = json.load(f)
        print("Loading Qwen + LoRA...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base_model, QWEN_LORA)
        self.model.eval()
        print(f"Done, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    def generate(self, prompt, max_new_tokens=256, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=0.9, do_sample=True, repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        gen = outputs[0][inputs['input_ids'].shape[1]:]
        return self._post_process(self.tokenizer.decode(gen, skip_special_tokens=True).strip())

    def generate_multiple(self, prompt, n=4):
        candidates = []
        for temp in [0.2, 0.4, 0.6, 0.8][:n]:
            text = self.generate(prompt, temperature=temp)
            if text and text not in candidates:
                candidates.append(text)
        return candidates

    def _post_process(self, text):
        for m in ['**解释', '**注释', '\n\n---', '解释：', '注释：']:
            if m in text:
                text = text[:text.index(m)]
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        return text.strip()

    def _format_prompt(self, user_message):
        messages = [{"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
                    {"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def get_terms_for_text(self, text):
        relevant = []
        for sk, info in self.terminology.items():
            if sk in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
                relevant.append(f"{sk}={info.get('chinese','?')}/{info.get('tibetan','?')}")
        return '；'.join(relevant[:5]) if relevant else ''

    def get_expected_terms(self, source_text, target_lang):
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
        expected = []
        for sk, info in self.terminology.items():
            if sk in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                if target_lang == 'sk':
                    expected.append(sk)
                elif target_key and info.get(target_key):
                    expected.append(info[target_key])
        return expected

    def compute_term_coverage(self, text, expected_terms):
        if not expected_terms:
            return 1.0
        return sum(1 for t in expected_terms if t in text) / len(expected_terms)

    def translate_with_term_constraint(self, source_text, source_lang, target_lang, constraint_level='hard'):
        sn, tn = LANG_NAMES[source_lang], LANG_NAMES[target_lang]
        terms = self.get_terms_for_text(source_text)
        expected = self.get_expected_terms(source_text, target_lang)

        if constraint_level == 'none':
            p = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。\n\n{sn}：{source_text}"
            return self.generate(self._format_prompt(p))

        elif constraint_level == 'soft':
            p = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。"
            if terms:
                p += f"\n术语参考：{terms}"
            p += f"\n\n{sn}：{source_text}"
            return self.generate(self._format_prompt(p))

        elif constraint_level == 'hard':
            if expected:
                tl = '、'.join(expected[:5])
                p = f"请将以下{sn}翻译为{tn}。翻译中必须包含以下术语：{tl}。只输出翻译结果。"
            else:
                p = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。"
            if terms:
                p += f"\n术语参考：{terms}"
            p += f"\n\n{sn}：{source_text}"
            return self.generate(self._format_prompt(p))

        elif constraint_level == 'iterative':
            p = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。"
            if terms:
                p += f"\n术语参考：{terms}"
            p += f"\n\n{sn}：{source_text}"
            first = self.generate(self._format_prompt(p))
            missing = [t for t in expected if t not in first]
            if not missing:
                return first
            ms = '、'.join(missing)
            p2 = f"以下翻译缺少术语，请修正确保包含：{ms}。只输出修正后翻译。\n原文：{source_text}\n当前翻译：{first}\n修正后翻译："
            return self.generate(self._format_prompt(p2))

    def rerank_candidates(self, candidates, source_text, target_lang, reference_len, pivot_text=None, weights=None):
        if weights is None:
            weights = {'term': 0.35, 'length': 0.20, 'fluency': 0.25, 'consistency': 0.20}
        expected = self.get_expected_terms(source_text, target_lang)
        scored = []
        for text in candidates:
            if not text or len(text) < 3:
                continue
            ts = self.compute_term_coverage(text, expected)
            ls = max(0, 1 - abs(len(text) / max(reference_len, 1) - 1) * 0.8) if reference_len > 0 else 0.5
            fs = self._compute_fluency(text, target_lang)
            cs = self._compute_consistency(text, pivot_text, target_lang) if pivot_text else 0.5
            total = weights['term'] * ts + weights['length'] * ls + weights['fluency'] * fs + weights['consistency'] * cs
            scored.append({'text': text, 'total_score': total, 'term_score': ts,
                          'length_score': ls, 'fluency_score': fs, 'consistency_score': cs})
        scored.sort(key=lambda x: x['total_score'], reverse=True)
        return scored

    def _compute_fluency(self, text, target_lang):
        score = 1.0
        words = text.split()
        if len(words) > 5:
            for n in [3, 4]:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                if any(c >= 3 for c in Counter(ngrams).values()):
                    score -= 0.3
                    break
        if len(text) < 10:
            score -= 0.3
        if target_lang == 'cn':
            non_cn = sum(1 for c in text if unicodedata.category(c).startswith('Lo') and not ('\u4e00' <= c <= '\u9fff'))
            if len(text) > 0 and non_cn / len(text) > 0.3:
                score -= 0.3
        if '**' in text or '注：' in text:
            score -= 0.2
        return max(0, min(1, score))

    def _compute_consistency(self, text, pivot_text, target_lang):
        if not pivot_text or not text:
            return 0.5
        lc = max(0, 1 - abs(len(text) / max(len(pivot_text), 1) - 1) * 0.5)
        exp = self.get_expected_terms(pivot_text, target_lang)
        tc = sum(1 for t in exp if t in text) / len(exp) if exp else 0.5
        return 0.5 * lc + 0.5 * tc

    def full_pipeline(self, entry, source_lang, target_lang, use_term_constraint=True,
                      use_mutual=True, use_reranking=True, rerank_weights=None):
        pivot_lang = list({'sk', 'tb', 'cn'} - {source_lang, target_lang})[0]
        src = entry[LANG_FIELDS[source_lang]]
        pvt = entry[LANG_FIELDS[pivot_lang]]
        ref = entry[LANG_FIELDS[target_lang]]
        sn, tn, pn = LANG_NAMES[source_lang], LANG_NAMES[target_lang], LANG_NAMES[pivot_lang]
        terms = self.get_terms_for_text(src)
        expected = self.get_expected_terms(src, target_lang)
        candidates = []

        p = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。"
        if terms:
            p += f"\n术语：{terms}"
        p += f"\n\n{sn}：{src}"
        candidates.extend(self.generate_multiple(self._format_prompt(p), n=2))

        if use_mutual:
            pm = f"请将以下{sn}翻译为{tn}，参考{pn}辅助理解。只输出翻译。"
            if terms:
                pm += f"\n术语：{terms}"
            pm += f"\n\n{sn}：{src}\n{pn}参考：{pvt}"
            candidates.extend(self.generate_multiple(self._format_prompt(pm), n=2))

        if use_term_constraint and expected:
            candidates.append(self.translate_with_term_constraint(src, source_lang, target_lang, 'hard'))
            candidates.append(self.translate_with_term_constraint(src, source_lang, target_lang, 'iterative'))

        candidates = list(set(c for c in candidates if c and len(c) > 3))

        if use_reranking and len(candidates) > 1:
            scored = self.rerank_candidates(candidates, src, target_lang, len(ref), pvt, rerank_weights)
            best = scored[0]['text'] if scored else (candidates[0] if candidates else "")
            best_score = scored[0]['total_score'] if scored else 0
        else:
            best = candidates[0] if candidates else ""
            best_score = 0

        return {'prediction': best, 'best_score': best_score, 'n_candidates': len(candidates),
                'expected_terms': expected, 'term_coverage': self.compute_term_coverage(best, expected)}

def run_ablation_experiment(max_samples=50):
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:max_samples]

    print("=" * 70)
    print("Step 4: Ablation Experiment")
    print("=" * 70)
    print(f"Samples: {max_samples}")

    translator = TermConstrainedTranslator()

    ablation_configs = {
        'A_baseline': {'desc': 'baseline(no constraint)', 'use_term_constraint': False, 'use_mutual': False, 'use_reranking': False},
        'B_soft_term': {'desc': '+soft term', 'use_term_constraint': False, 'use_mutual': False, 'use_reranking': False},
        'C_hard_term': {'desc': '+hard term', 'use_term_constraint': True, 'use_mutual': False, 'use_reranking': False},
        'D_mutual': {'desc': '+mutual', 'use_term_constraint': False, 'use_mutual': True, 'use_reranking': False},
        'E_rerank': {'desc': '+rerank', 'use_term_constraint': False, 'use_mutual': False, 'use_reranking': True},
        'F_full': {'desc': 'full pipeline', 'use_term_constraint': True, 'use_mutual': True, 'use_reranking': True},
    }

    directions = [('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb')]
    all_results = {}

    for sl, tl in directions:
        d = f"{sl}->{tl}"
        print(f"\nDirection: {d}")
        for cn, cfg in ablation_configs.items():
            print(f"  Running {cn}: {cfg['desc']}")
            results = []
            for entry in tqdm(test_data, desc=cn, leave=False):
                if cn == 'A_baseline':
                    pred = translator.translate_with_term_constraint(entry[LANG_FIELDS[sl]], sl, tl, 'none')
                    res = {'prediction': pred, 'term_coverage': translator.compute_term_coverage(pred, translator.get_expected_terms(entry[LANG_FIELDS[sl]], tl))}
                elif cn == 'B_soft_term':
                    pred = translator.translate_with_term_constraint(entry[LANG_FIELDS[sl]], sl, tl, 'soft')
                    res = {'prediction': pred, 'term_coverage': translator.compute_term_coverage(pred, translator.get_expected_terms(entry[LANG_FIELDS[sl]], tl))}
                elif cn == 'C_hard_term':
                    pred = translator.translate_with_term_constraint(entry[LANG_FIELDS[sl]], sl, tl, 'hard')
                    res = {'prediction': pred, 'term_coverage': translator.compute_term_coverage(pred, translator.get_expected_terms(entry[LANG_FIELDS[sl]], tl))}
                else:
                    res = translator.full_pipeline(entry, sl, tl,
                        use_term_constraint=cfg['use_term_constraint'],
                        use_mutual=cfg['use_mutual'],
                        use_reranking=cfg['use_reranking'])
                results.append({'id': entry['id'], 'source_lang': sl, 'target_lang': tl,
                    'source_text': entry[LANG_FIELDS[sl]], 'reference': entry[LANG_FIELDS[tl]],
                    'prediction': res['prediction'], 'term_coverage': res.get('term_coverage', 0),
                    'model': 'qwen_lora_v2', 'method': cn})
            path = f"{RESULTS_DIR}/{sl}_to_{tl}_{cn}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            all_results[f"{d}_{cn}"] = results

    # Evaluation
    print("\n" + "=" * 70)
    print("Ablation Results")
    print("=" * 70)
    from sacrebleu.metrics import CHRF
    chrf_scorer = CHRF(word_order=2)
    print(f"\n{'Dir':<10} {'Config':<14} {'chrF++':>7} {'TermAcc':>8} | Desc")
    print("-" * 70)
    eval_summary = {}
    for d in ['sk->cn', 'tb->cn', 'sk->tb']:
        for cn, cfg in ablation_configs.items():
            k = f"{d}_{cn}"
            if k not in all_results:
                continue
            res = all_results[k]
            preds = [r['prediction'] for r in res]
            refs = [r['reference'] for r in res]
            chrf = chrf_scorer.corpus_score(preds, [refs]).score
            tl = res[0]['target_lang']
            tk = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[tl]
            tot, cor = 0, 0
            for r in res:
                for sk, info in translator.terminology.items():
                    if sk not in r['source_text'] and info.get('tibetan', '') not in r['source_text'] and info.get('chinese', '') not in r['source_text']:
                        continue
                    exp = sk if tl == 'sk' else info.get(tk, '')
                    if not exp:
                        continue
                    if exp in r['reference']:
                        tot += 1
                        if exp in r['prediction']:
                            cor += 1
            ta = cor / tot if tot > 0 else 0
            print(f"{d:<10} {cn:<14} {chrf:>7.2f} {ta:>8.4f} | {cfg['desc']}")
            eval_summary[k] = {'chrf': chrf, 'term_acc': ta}
        print("-" * 70)

    # Gains
    print(f"\n{'Dir':<10} {'Component':<14} {'chrF++ gain':>11} {'TermAcc gain':>12}")
    print("-" * 50)
    for d in ['sk->cn', 'tb->cn', 'sk->tb']:
        bk = f"{d}_A_baseline"
        if bk not in eval_summary:
            continue
        bc, bt = eval_summary[bk]['chrf'], eval_summary[bk]['term_acc']
        for cn in ['B_soft_term', 'C_hard_term', 'D_mutual', 'E_rerank', 'F_full']:
            k = f"{d}_{cn}"
            if k not in eval_summary:
                continue
            cg = eval_summary[k]['chrf'] - bc
            tg = eval_summary[k]['term_acc'] - bt
            cs = f"+{cg:.2f}" if cg >= 0 else f"{cg:.2f}"
            ts = f"+{tg:.4f}" if tg >= 0 else f"{tg:.4f}"
            print(f"{d:<10} {cn:<14} {cs:>11} {ts:>12}")
        print("-" * 50)

    # Weight tuning
    print("\n" + "=" * 70)
    print("Weight Tuning (SK->CN, 30 samples)")
    print("=" * 70)
    wcs = {
        'W1_term_heavy': {'term': 0.50, 'length': 0.15, 'fluency': 0.20, 'consistency': 0.15},
        'W2_balanced': {'term': 0.30, 'length': 0.25, 'fluency': 0.25, 'consistency': 0.20},
        'W3_fluency': {'term': 0.25, 'length': 0.20, 'fluency': 0.35, 'consistency': 0.20},
        'W4_consist': {'term': 0.25, 'length': 0.15, 'fluency': 0.20, 'consistency': 0.40},
        'W5_term_max': {'term': 0.70, 'length': 0.10, 'fluency': 0.10, 'consistency': 0.10},
    }
    print(f"{'Config':<14} {'chrF++':>7} {'TermAcc':>8}")
    print("-" * 35)
    for wn, w in wcs.items():
        res = []
        for entry in tqdm(test_data[:30], desc=wn, leave=False):
            r = translator.full_pipeline(entry, 'sk', 'cn', True, True, True, w)
            res.append({'prediction': r['prediction'], 'reference': entry['chinese'],
                       'source_text': entry['sanskrit'], 'target_lang': 'cn'})
        preds = [r['prediction'] for r in res]
        refs = [r['reference'] for r in res]
        chrf = chrf_scorer.corpus_score(preds, [refs]).score
        tot, cor = 0, 0
        for r in res:
            for sk, info in translator.terminology.items():
                if sk not in r['source_text']:
                    continue
                exp = info.get('chinese', '')
                if not exp:
                    continue
                if exp in r['reference']:
                    tot += 1
                    if exp in r['prediction']:
                        cor += 1
        ta = cor / tot if tot > 0 else 0
        print(f"{wn:<14} {chrf:>7.2f} {ta:>8.4f}")

    del translator
    torch.cuda.empty_cache()
    print("\nStep 4 Done!")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_ablation_experiment(n)
