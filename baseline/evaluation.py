#!/usr/bin/env python3
"""评估脚本 - BLEU/chrF++/术语准确率"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

try:
    from sacrebleu.metrics import BLEU, CHRF
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("⚠️  sacrebleu 未安装，运行: pip install sacrebleu")

class TranslationEvaluator:
    """翻译评估器"""
    
    def __init__(self, terminology_path: str = None):
        if SACREBLEU_AVAILABLE:
            self.bleu = BLEU(effective_order=True)
            self.chrf = CHRF(word_order=2)
        
        self.terminology = {}
        if terminology_path and Path(terminology_path).exists():
            with open(terminology_path, 'r', encoding='utf-8') as f:
                self.terminology = json.load(f)
    
    def evaluate_file(self, result_path: str):
        """评估单个结果文件"""
        with open(result_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        predictions = [r['prediction'] for r in results]
        references = [r['reference'] for r in results]
        
        print(f"\n{'='*60}")
        print(f"📊 评估: {Path(result_path).name}")
        print(f"{'='*60}")
        print(f"样本数: {len(results)}")
        
        # BLEU
        if SACREBLEU_AVAILABLE:
            bleu_score = self.bleu.corpus_score(predictions, [references]).score
            print(f"BLEU:   {bleu_score:.2f}")
            
            # chrF++
            chrf_score = self.chrf.corpus_score(predictions, [references]).score
            print(f"chrF++: {chrf_score:.2f}")
        else:
            print("BLEU/chrF++: 未安装 sacrebleu")
        
        # 术语准确率
        if self.terminology:
            term_acc = self._compute_term_accuracy(
                predictions, references, 
                [r['source_text'] for r in results],
                results[0]['target_lang']
            )
            print(f"术语准确率: {term_acc:.4f}")
        
        # 平均长度
        avg_pred_len = np.mean([len(p) for p in predictions])
        avg_ref_len = np.mean([len(r) for r in references])
        print(f"平均长度: 预测={avg_pred_len:.1f}, 参考={avg_ref_len:.1f}")
        
        return {
            'bleu': bleu_score if SACREBLEU_AVAILABLE else 0,
            'chrf': chrf_score if SACREBLEU_AVAILABLE else 0,
            'term_acc': term_acc if self.terminology else 0,
        }
    
    def _compute_term_accuracy(self, predictions: List[str], 
                                references: List[str],
                                sources: List[str],
                                target_lang: str) -> float:
        """计算术语准确率"""
        total = 0
        correct = 0
        
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
        
        for pred, ref, src in zip(predictions, references, sources):
            for sk_term, info in self.terminology.items():
                # 源文本包含该术语
                if sk_term not in src:
                    continue
                
                if target_lang == 'sk':
                    expected = sk_term
                else:
                    expected = info.get(target_key, '')
                
                if not expected:
                    continue
                
                total += 1
                if expected in ref and expected in pred:
                    correct += 1
        
        return correct / total if total > 0 else 0.0

def evaluate_all_results():
    """评估所有结果"""
    results_dir = Path("/root/autodl-tmp/LlamaFactory-main/baseline/results/baseline")
    term_path = "/root/autodl-tmp/LlamaFactory-main/data_preparation/data/terminology_final.json"
    
    evaluator = TranslationEvaluator(terminology_path=term_path)
    
    result_files = list(results_dir.glob("*.jsonl"))
    
    if not result_files:
        print(f"⚠️  未找到结果文件: {results_dir}")
        return
    
    print("=" * 60)
    print("📊 评估所有基线实验结果")
    print("=" * 60)
    
    all_scores = {}
    for filepath in sorted(result_files):
        scores = evaluator.evaluate_file(str(filepath))
        all_scores[filepath.stem] = scores
    
    # 汇总
    print(f"\n{'='*60}")
    print("📋 评估汇总")
    print(f"{'='*60}")
    for name, scores in all_scores.items():
        print(f"{name}:")
        print(f"  BLEU={scores['bleu']:.2f}, chrF++={scores['chrf']:.2f}, "
              f"TermAcc={scores['term_acc']:.4f}")

if __name__ == "__main__":
    evaluate_all_results()
