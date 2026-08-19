#!/usr/bin/env python3
"""评估所有实验结果：基线 vs 微调"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from sacrebleu.metrics import BLEU, CHRF
    SACREBLEU = True
except ImportError:
    SACREBLEU = False
    print("安装 sacrebleu: pip install sacrebleu")

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
RESULTS_BASE = f'{BASE_DIR}/baseline/results'

# 加载术语表
with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
    terminology = json.load(f)

LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

def compute_term_accuracy(results):
    """计算术语准确率"""
    total = 0
    correct = 0
    target_key_map = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}
    
    for r in results:
        target_lang = r['target_lang']
        target_key = target_key_map.get(target_lang)
        source_text = r['source_text']
        pred = r['prediction']
        ref = r['reference']
        
        for sk_term, info in terminology.items():
            if sk_term not in source_text and info.get('tibetan', '') not in source_text and info.get('chinese', '') not in source_text:
                continue
            
            if target_lang == 'sk':
                expected = sk_term
            else:
                expected = info.get(target_key, '')
            
            if not expected:
                continue
            
            if expected in ref:
                total += 1
                if expected in pred:
                    correct += 1
    
    return correct / total if total > 0 else 0.0

def evaluate_file(filepath):
    """评估单个结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    if not results:
        return None
    
    predictions = [r['prediction'] for r in results]
    references = [r['reference'] for r in results]
    
    scores = {'n': len(results)}
    
    if SACREBLEU:
        bleu = BLEU(effective_order=True)
        chrf = CHRF(word_order=2)
        scores['bleu'] = bleu.corpus_score(predictions, [references]).score
        scores['chrf'] = chrf.corpus_score(predictions, [references]).score
    
    scores['term_acc'] = compute_term_accuracy(results)
    
    # 平均长度比
    avg_pred_len = np.mean([len(p) for p in predictions])
    avg_ref_len = np.mean([len(r) for r in references])
    scores['len_ratio'] = avg_pred_len / avg_ref_len if avg_ref_len > 0 else 0
    
    return scores

def main():
    print("=" * 90)
    print("📊 Step 2 完整评估报告：基线 vs 微调")
    print("=" * 90)
    
    all_results = []
    
    # 扫描所有结果目录
    for subdir in ['zero_shot', 'few_shot', 'mutual', 'finetuned']:
        result_dir = Path(RESULTS_BASE) / subdir
        if not result_dir.exists():
            continue
        
        for filepath in sorted(result_dir.glob("*.jsonl")):
            scores = evaluate_file(str(filepath))
            if scores is None:
                continue
            
            # 从文件名解析信息
            name = filepath.stem
            method = subdir
            
            # 解析方向
            with open(filepath, 'r', encoding='utf-8') as f:
                first = json.loads(f.readline())
            
            source_lang = first.get('source_lang', '?')
            target_lang = first.get('target_lang', '?')
            model = first.get('model', '?')
            
            all_results.append({
                'file': name,
                'model': model,
                'method': method,
                'direction': f"{source_lang}→{target_lang}",
                'source_lang': source_lang,
                'target_lang': target_lang,
                **scores
            })
    
    # 按方向分组打印
    by_direction = defaultdict(list)
    for r in all_results:
        by_direction[r['direction']].append(r)
    
    print(f"\n{'方向':<8} {'模型':<16} {'方法':<18} {'BLEU':>7} {'chrF++':>8} {'TermAcc':>8} {'LenRatio':>9}")
    print("─" * 90)
    
    for direction in sorted(by_direction.keys()):
        items = sorted(by_direction[direction], key=lambda x: x.get('chrf', 0), reverse=True)
        for r in items:
            bleu_str = f"{r.get('bleu', 0):.2f}" if SACREBLEU else "N/A"
            chrf_str = f"{r.get('chrf', 0):.2f}" if SACREBLEU else "N/A"
            print(f"{r['direction']:<8} {r['model']:<16} {r['method']:<18} "
                  f"{bleu_str:>7} {chrf_str:>8} {r['term_acc']:>8.4f} {r['len_ratio']:>9.2f}")
        print("─" * 90)
    
    # 微调增益分析
    print("\n" + "=" * 90)
    print("📈 微调增益分析 (Qwen: 零样本 vs 微调直接 vs 微调互证)")
    print("=" * 90)
    
    print(f"\n{'方向':<8} {'零样本chrF':>10} {'微调直接chrF':>12} {'微调互证chrF':>12} {'直接增益':>8} {'互证增益':>8}")
    print("─" * 70)
    
    for direction in sorted(by_direction.keys()):
        items = by_direction[direction]
        
        # 找 qwen 零样本
        zs = [r for r in items if 'qwen' in r['model'] and r['method'] == 'zero_shot']
        ft_direct = [r for r in items if r['method'] == 'finetuned' and 'direct' in r['file']]
        ft_mutual = [r for r in items if r['method'] == 'finetuned' and 'mutual' in r['file']]
        
        zs_chrf = zs[0]['chrf'] if zs else 0
        ft_d_chrf = ft_direct[0]['chrf'] if ft_direct else 0
        ft_m_chrf = ft_mutual[0]['chrf'] if ft_mutual else 0
        
        gain_d = ft_d_chrf - zs_chrf
        gain_m = ft_m_chrf - zs_chrf
        
        gain_d_str = f"+{gain_d:.2f}" if gain_d >= 0 else f"{gain_d:.2f}"
        gain_m_str = f"+{gain_m:.2f}" if gain_m >= 0 else f"{gain_m:.2f}"
        
        print(f"{direction:<8} {zs_chrf:>10.2f} {ft_d_chrf:>12.2f} {ft_m_chrf:>12.2f} "
              f"{gain_d_str:>8} {gain_m_str:>8}")
    
    # 术语准确率对比
    print("\n" + "=" * 90)
    print("📈 术语准确率对比")
    print("=" * 90)
    
    print(f"\n{'方向':<8} {'零样本':>8} {'few_shot':>8} {'互证(基线)':>10} {'微调直接':>8} {'微调互证':>8}")
    print("─" * 60)
    
    for direction in sorted(by_direction.keys()):
        items = by_direction[direction]
        
        def get_term(method_filter, file_filter=None):
            filtered = [r for r in items if r['method'] == method_filter]
            if file_filter:
                filtered = [r for r in filtered if file_filter in r['file']]
            if filtered:
                return filtered[0]['term_acc']
            return 0
        
        zs = get_term('zero_shot')
        fs = get_term('few_shot')
        mt = get_term('mutual')
        ft_d = get_term('finetuned', 'direct')
        ft_m = get_term('finetuned', 'mutual')
        
        print(f"{direction:<8} {zs:>8.4f} {fs:>8.4f} {mt:>10.4f} {ft_d:>8.4f} {ft_m:>8.4f}")
    
    print("\n✅ 评估完成")

if __name__ == "__main__":
    main()
