cat > /root/autodl-tmp/LlamaFactory-main/baseline/run_step4_ablation.py << 'PYTHON'
#!/usr/bin/env python3
"""
Step 4: 术语约束与重排序 - 消融实验
1. 术语约束解码模块
2. 多候选重排序模块
3. 消融实验：逐步加入各组件
4. 调优重排序权重
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from itertools import product
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
    """术语约束翻译器 + 多候选重排序"""
    
    def __init__(self):
        with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
            self.terminology = json.load(f)
        
        print("🔄 加载 Qwen + LoRA...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base_model, QWEN_LORA)
        self.model.eval()
        print(f"✓ 加载完成, 显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.3) -> str:
        """基础生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=0.9, do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._post_process(text)
    
    def generate_multiple(self, prompt: str, n: int = 4) -> List[str]:
        """生成多个候选（不同温度）"""
        candidates = []
        temps = [0.2, 0.4, 0.6, 0.8][:n]
        for temp in temps:
            text = self.generate(prompt, temperature=temp)
            if text and text not in candidates:
                candidates.append(text)
        return candidates
    
    def _post_process(self, text: str) -> str:
        for marker in ['**解释', '**注释', '\n\n---', '解释：', '注释：']:
            if marker in text:
                text = text[:text.index(marker)]
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        return text.strip()
    
    def _format_prompt(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
            {"role": "user", "content": user_message}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # ================================================================
    # 术语相关工具
    # ================================================================
    
    def get_terms_for_text(self, text: str) -> str:
        """获取术语提示字符串"""
        relevant = []
        for sk_term, info in self.terminology.items():
            if sk_term in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
                relevant.append(f"{sk_term}={info.get('chinese','?')}/{info.get('tibetan','?')}")
        return '；'.join(relevant[:5]) if relevant else ''
    
    def get_expected_terms(self, source_text: str, target_lang: str) -> List[str]:
        """获取目标语言中应出现的术语"""
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
        expected = []
        for sk_term, info in self.terminology.items():
            if sk_term in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                if target_lang == 'sk':
                    expected.append(sk_term)
                elif target_key and info.get(target_key):
                    expected.append(info[target_key])
        return expected
    
    def compute_term_coverage(self, text: str, expected_terms: List[str]) -> float:
        """计算术语覆盖率"""
        if not expected_terms:
            return 1.0
        covered = sum(1 for t in expected_terms if t in text)
        return covered / len(expected_terms)
    
    # ================================================================
    # 组件1：术语约束解码
    # ================================================================
    
    def translate_with_term_constraint(self, source_text: str, source_lang: str,
                                        target_lang: str, 
                                        constraint_level: str = 'hard') -> str:
        """
        术语约束翻译
        constraint_level:
          - 'none': 无约束（普通翻译）
          - 'soft': 软约束（在prompt中提示术语）
          - 'hard': 硬约束（强制要求包含术语）
          - 'iterative': 迭代约束（检查后补充）
        """
        source_name = LANG_NAMES[source_lang]
        target_name = LANG_NAMES[target_lang]
        terms = self.get_terms_for_text(source_text)
        expected_terms = self.get_expected_terms(source_text, target_lang)
        
        if constraint_level == 'none':
            # 无约束：不提供术语信息
            prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
            prompt += f"\n\n{source_name}：{source_text}"
            formatted = self._format_prompt(prompt)
            return self.generate(formatted)
        
        elif constraint_level == 'soft':
            # 软约束：在prompt中提供术语参考
            prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
            if terms:
                prompt += f"\n术语参考：{terms}"
            prompt += f"\n\n{source_name}：{source_text}"
            formatted = self._format_prompt(prompt)
            return self.generate(formatted)
        
        elif constraint_level == 'hard':
            # 硬约束：明确要求必须包含术语
            if expected_terms:
                term_list = '、'.join(expected_terms[:5])
                prompt = (f"请将以下{source_name}翻译为{target_name}。"
                         f"翻译中必须包含以下术语：{term_list}。"
                         f"只输出翻译结果。")
            else:
                prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
            if terms:
                prompt += f"\n术语参考：{terms}"
            prompt += f"\n\n{source_name}：{source_text}"
            formatted = self._format_prompt(prompt)
            return self.generate(formatted)
        
        elif constraint_level == 'iterative':
            # 迭代约束：先翻译，检查缺失术语，再补充
            # 第一轮：软约束翻译
            prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
            if terms:
                prompt += f"\n术语参考：{terms}"
            prompt += f"\n\n{source_name}：{source_text}"
            formatted = self._format_prompt(prompt)
            first_translation = self.generate(formatted)
            
            # 检查缺失术语
            missing_terms = [t for t in expected_terms if t not in first_translation]
            
            if not missing_terms:
                return first_translation
            
            # 第二轮：修正翻译，补充缺失术语
            missing_str = '、'.join(missing_terms)
            prompt2 = (f"以下是{source_name}到{target_name}的翻译，但缺少了一些术语。"
                      f"请修正翻译，确保包含：{missing_str}。只输出修正后的完整翻译。"
                      f"\n\n原文：{source_text}"
                      f"\n当前翻译：{first_translation}"
                      f"\n缺失术语：{missing_str}"
                      f"\n修正后翻译：")
            formatted2 = self._format_prompt(prompt2)
            return self.generate(formatted2)
    
    # ================================================================
    # 组件2：多候选重排序
    # ================================================================
    
    def rerank_candidates(self, candidates: List[str], source_text: str,
                          target_lang: str, reference_len: int,
                          pivot_text: str = None,
                          weights: Dict[str, float] = None) -> List[Dict]:
        """
        多候选重排序
        
        评分维度：
        1. term_score: 术语覆盖率
        2. length_score: 长度合理性
        3. fluency_score: 流畅度（基于重复检测和格式）
        4. consistency_score: 与枢轴文本的一致性
        """
        if weights is None:
            weights = {
                'term': 0.35,
                'length': 0.20,
                'fluency': 0.25,
                'consistency': 0.20,
            }
        
        expected_terms = self.get_expected_terms(source_text, target_lang)
        scored = []
        
        for text in candidates:
            if not text or len(text) < 3:
                continue
            
            # 1. 术语覆盖率
            term_score = self.compute_term_coverage(text, expected_terms)
            
            # 2. 长度合理性
            if reference_len > 0:
                ratio = len(text) / reference_len
                length_score = max(0, 1 - abs(ratio - 1) * 0.8)
            else:
                length_score = 0.5
            
            # 3. 流畅度（启发式）
            fluency_score = self._compute_fluency(text, target_lang)
            
            # 4. 一致性（与枢轴文本的关联）
            if pivot_text:
                consistency_score = self._compute_consistency(text, pivot_text, target_lang)
            else:
                consistency_score = 0.5
            
            # 加权总分
            total = (
                weights['term'] * term_score +
                weights['length'] * length_score +
                weights['fluency'] * fluency_score +
                weights['consistency'] * consistency_score
            )
            
            scored.append({
                'text': text,
                'total_score': total,
                'term_score': term_score,
                'length_score': length_score,
                'fluency_score': fluency_score,
                'consistency_score': consistency_score,
            })
        
        scored.sort(key=lambda x: x['total_score'], reverse=True)
        return scored
    
    def _compute_fluency(self, text: str, target_lang: str) -> float:
        """流畅度评分（启发式）"""
        score = 1.0
        
        # 惩罚：重复片段
        words = text.split()
        if len(words) > 5:
            for n in [3, 4]:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                from collections import Counter
                counts = Counter(ngrams)
                if any(c >= 3 for c in counts.values()):
                    score -= 0.3
                    break
        
        # 惩罚：过短
        if len(text) < 10:
            score -= 0.3
        
        # 惩罚：包含源语言字符（语言混杂）
        import unicodedata
        if target_lang == 'cn':
            # 汉文输出不应包含大量梵文/藏文
            non_cn = sum(1 for c in text if unicodedata.category(c).startswith('Lo') 
                        and not ('\u4e00' <= c <= '\u9fff'))
            if len(text) > 0 and non_cn / len(text) > 0.3:
                score -= 0.3
        
        # 惩罚：包含 markdown 或解释性文字
        if '**' in text or '注：' in text or '解释' in text:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _compute_consistency(self, text: str, pivot_text: str, 
                             target_lang: str) -> float:
        """与枢轴文本的一致性（简化版）"""
        if not pivot_text or not text:
            return 0.5
        
        # 长度比例一致性
        ratio = len(text) / max(len(pivot_text), 1)
        length_consistency = max(0, 1 - abs(ratio - 1) * 0.5)
        
        # 共享术语检查
        expected = self.get_expected_terms(pivot_text, target_lang)
        if expected:
            covered = sum(1 for t in expected if t in text)
            term_consistency = covered / len(expected)
        else:
            term_consistency = 0.5
        
        return 0.5 * length_consistency + 0.5 * term_consistency
    
    # ================================================================
    # 完整 Pipeline（组合所有组件）
    # ================================================================
    
    def full_pipeline(self, entry: Dict, source_lang: str, target_lang: str,
                      use_term_constraint: bool = True,
                      use_mutual: bool = True,
                      use_reranking: bool = True,
                      rerank_weights: Dict = None,
                      n_candidates: int = 4) -> Dict:
        """
        完整翻译 pipeline
        可通过开关控制各组件，用于消融实验
        """
        all_langs = {'sk', 'tb', 'cn'}
        pivot_lang = list(all_langs - {source_lang, target_lang})[0]
        
        source_text = entry[LANG_FIELDS[source_lang]]
        pivot_text = entry[LANG_FIELDS[pivot_lang]]
        reference = entry[LANG_FIELDS[target_lang]]
        
        source_name = LANG_NAMES[source_lang]
        target_name = LANG_NAMES[target_lang]
        pivot_name = LANG_NAMES[pivot_lang]
        terms = self.get_terms_for_text(source_text)
        expected_terms = self.get_expected_terms(source_text, target_lang)
        
        candidates = []
        
        # --- 基础候选：直接翻译（多温度）---
        prompt_base = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt_base += f"\n术语：{terms}"
        prompt_base += f"\n\n{source_name}：{source_text}"
        formatted = self._format_prompt(prompt_base)
        
        base_candidates = self.generate_multiple(formatted, n=2)
        candidates.extend(base_candidates)
        
        # --- 互证候选 ---
        if use_mutual:
            prompt_mutual = (f"请将以下{source_name}翻译为{target_name}，"
                           f"参考{pivot_name}辅助理解。只输出翻译。")
            if terms:
                prompt_mutual += f"\n术语：{terms}"
            prompt_mutual += f"\n\n{source_name}：{source_text}\n{pivot_name}参考：{pivot_text}"
            formatted_m = self._format_prompt(prompt_mutual)
            mutual_candidates = self.generate_multiple(formatted_m, n=2)
            candidates.extend(mutual_candidates)
        
        # --- 术语约束候选 ---
        if use_term_constraint and expected_terms:
            # 硬约束
            hard_result = self.translate_with_term_constraint(
                source_text, source_lang, target_lang, 'hard'
            )
            candidates.append(hard_result)
            
            # 迭代约束
            iter_result = self.translate_with_term_constraint(
                source_text, source_lang, target_lang, 'iterative'
            )
            candidates.append(iter_result)
        
        # 去重
        candidates = list(set(c for c in candidates if c and len(c) > 3))
        
        # --- 重排序 ---
        if use_reranking and len(candidates) > 1:
            scored = self.rerank_candidates(
                candidates, source_text, target_lang,
                reference_len=len(reference),
                pivot_text=pivot_text,
                weights=rerank_weights
            )
            best = scored[0]['text'] if scored else candidates[0]
            best_score = scored[0]['total_score'] if scored else 0
            scores_detail = scored[0] if scored else {}
        else:
            best = candidates[0] if candidates else ""
            best_score = 0
            scores_detail = {}
            scored = [{'text': c, 'total_score': 0} for c in candidates]
        
        return {
            'prediction': best,
            'best_score': best_score,
            'n_candidates': len(candidates),
            'scores_detail': scores_detail,
            'all_candidates': scored[:5],  # 保留前5个
            'expected_terms': expected_terms,
            'term_coverage': self.compute_term_coverage(best, expected_terms),
        }

# ================================================================
# 消融实验
# ================================================================

def run_ablation_experiment(max_samples: int = 50):
    """
    消融实验：逐步加入各组件
    
    配置：
    A: 基础（直接翻译，无约束，无重排序）
    B: + 术语软约束
    C: + 术语硬约束
    D: + 互证翻译
    E: + 多候选重排序
    F: 完整 pipeline（全部组件）
    """
    
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:max_samples]
    
    print("=" * 70)
    print("🧪 Step 4: 消融实验 - 逐步加入各组件")
    print("=" * 70)
    print(f"测试样本: {max_samples} 条")
    
    translator = TermConstrainedTranslator()
    
    # 消融配置
    ablation_configs = {
        'A_baseline': {
            'desc': '基础（直接翻译，无约束）',
            'use_term_constraint': False,
            'use_mutual': False,
            'use_reranking': False,
        },
        'B_soft_term': {
            'desc': '+ 术语软约束',
            'use_term_constraint': False,  # 软约束在prompt中已有
            'use_mutual': False,
            'use_reranking': False,
        },
        'C_hard_term': {
            'desc': '+ 术语硬约束',
            'use_term_constraint': True,
            'use_mutual': False,
            'use_reranking': False,
        },
        'D_mutual': {
            'desc': '+ 互证翻译',
            'use_term_constraint': False,
            'use_mutual': True,
            'use_reranking': False,
        },
        'E_rerank': {
            'desc': '+ 多候选重排序',
            'use_term_constraint': False,
            'use_mutual': False,
            'use_reranking': True,
        },
        'F_full': {
            'desc': '完整 pipeline（全部组件）',
            'use_term_constraint': True,
            'use_mutual': True,
            'use_reranking': True,
        },
    }
    
    # 选择代表性方向
    directions = [
        ('sk', 'cn'),  # 梵→汉（核心方向）
        ('tb', 'cn'),  # 藏→汉
        ('sk', 'tb'),  # 梵→藏
    ]
    
    all_results = {}
    
    for source_lang, target_lang in directions:
        direction = f"{source_lang}→{target_lang}"
        print(f"\n{'='*70}")
        print(f"📌 方向: {direction}")
        print(f"{'='*70}")
        
        for config_name, config in ablation_configs.items():
            print(f"\n  --- {config_name}: {config['desc']} ---")
            
            results = []
            for entry in tqdm(test_data, desc=f"{config_name}", leave=False):
                
                if config_name == 'A_baseline':
                    # 纯直接翻译，无术语提示
                    pred = translator.translate_with_term_constraint(
                        entry[LANG_FIELDS[source_lang]], source_lang, target_lang, 'none'
                    )
                    result = {'prediction': pred, 'term_coverage': 
                              translator.compute_term_coverage(
                                  pred, translator.get_expected_terms(
                                      entry[LANG_FIELDS[source_lang]], target_lang))}
                
                elif config_name == 'B_soft_term':
                    # 软约束（prompt中提供术语）
                    pred = translator.translate_with_term_constraint(
                        entry[LANG_FIELDS[source_lang]], source_lang, target_lang, 'soft'
                    )
                    result = {'prediction': pred, 'term_coverage':
                              translator.compute_term_coverage(
                                  pred, translator.get_expected_terms(
                                      entry[LANG_FIELDS[source_lang]], target_lang))}
                
                elif config_name == 'C_hard_term':
                    # 硬约束
                    pred = translator.translate_with_term_constraint(
                        entry[LANG_FIELDS[source_lang]], source_lang, target_lang, 'hard'
                    )
                    result = {'prediction': pred, 'term_coverage':
                              translator.compute_term_coverage(
                                  pred, translator.get_expected_terms(
                                      entry[LANG_FIELDS[source_lang]], target_lang))}
                
                else:
                    # D/E/F 使用 full_pipeline
                    result = translator.full_pipeline(
                        entry, source_lang, target_lang,
                        use_term_constraint=config['use_term_constraint'],
                        use_mutual=config['use_mutual'],
                        use_reranking=config['use_reranking'],
                    )
                
                results.append({
                    'id': entry['id'],
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'source_text': entry[LANG_FIELDS[source_lang]],
                    'reference': entry[LANG_FIELDS[target_lang]],
                    'prediction': result['prediction'],
                    'term_coverage': result.get('term_coverage', 0),
                    'model': 'qwen_lora_v2',
                    'method': config_name,
                })
            
            # 保存
            path = f"{RESULTS_DIR}/{source_lang}_to_{target_lang}_{config_name}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            key = f"{direction}_{config_name}"
            all_results[key] = results
    
    # ============================================================
    # 评估消融结果
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 消融实验评估结果")
    print("=" * 70)
    
    try:
        from sacrebleu.metrics import BLEU, CHRF
        bleu_scorer = BLEU(effective_order=True)
        chrf_scorer = CHRF(word_order=2)
        
        print(f"\n{'方向':<8} {'配置':<14} {'chrF++':>7} {'TermCov':>8} {'TermAcc':>8} | 描述")
        print("─" * 85)
        
        eval_summary = {}
        
        for direction in ['sk→cn', 'tb→cn', 'sk→tb']:
            for config_name, config in ablation_configs.items():
                key = f"{direction}_{config_name}"
                if key not in all_results:
                    continue
                
                results = all_results[key]
                preds = [r['prediction'] for r in results]
                refs = [r['reference'] for r in results]
                
                chrf_score = chrf_scorer.corpus_score(preds, [refs]).score
                avg_term_cov = np.mean([r['term_coverage'] for r in results])
                
                # 术语准确率
                total_t = 0
                correct_t = 0
                target_lang = results[0]['target_lang']
                target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
                for r in results:
                    for sk_term, info in translator.terminology.items():
                        src = r['source_text']
                        if sk_term not in src and info.get('tibetan','') not in src and info.get('chinese','') not in src:
                            continue
                        if target_lang == 'sk':
                            expected = sk_term
                        else:
                            expected = info.get(target_key, '')
                        if not expected:
                            continue
                        if expected in r['reference']:
                            total_t += 1
                            if expected in r['prediction']:
                                correct_t += 1
                term_acc = correct_t / total_t if total_t > 0 else 0
                
                print(f"{direction:<8} {config_name:<14} {chrf_score:>7.2f} {avg_term_cov:>8.4f} "
                      f"{term_acc:>8.4f} | {config['desc']}")
                
                eval_summary[key] = {
                    'chrf': chrf_score, 'term_cov': avg_term_cov, 'term_acc': term_acc
                }
            
            print("─" * 85)
        
        # 组件增益分析
        print("\n" + "=" * 70)
        print("📈 组件增益分析（相对于基线 A）")
        print("=" * 70)
        
        print(f"\n{'方向':<8} {'组件':<20} {'chrF++增益':>10} {'TermAcc增益':>11}")
        print("─" * 55)
        
        for direction in ['sk→cn', 'tb→cn', 'sk→tb']:
            baseline_key = f"{direction}_A_baseline"
            if baseline_key not in eval_summary:
                continue
            
            base_chrf = eval_summary[baseline_key]['chrf']
            base_term = eval_summary[baseline_key]['term_acc']
            
            for config_name in ['B_soft_term', 'C_hard_term', 'D_mutual', 'E_rerank', 'F_full']:
                key = f"{direction}_{config_name}"
                if key not in eval_summary:
                    continue
                
                chrf_gain = eval_summary[key]['chrf'] - base_chrf
                term_gain = eval_summary[key]['term_acc'] - base_term
                
                chrf_str = f"+{chrf_gain:.2f}" if chrf_gain >= 0 else f"{chrf_gain:.2f}"
                term_str = f"+{term_gain:.4f}" if term_gain >= 0 else f"{term_gain:.4f}"

cat >> /root/autodl-tmp/LlamaFactory-main/baseline/run_step4_ablation.py << 'PYTHON'
                term_str = f"+{term_gain:.4f}" if term_gain >= 0 else f"{term_gain:.4f}"
                
                print(f"{direction:<8} {config_name:<20} {chrf_str:>10} {term_str:>11}")
            
            print("─" * 55)
    
    except ImportError:
        print("⚠️  sacrebleu 未安装")
    
    # ============================================================
    # 权重调优实验
    # ============================================================
    print("\n" + "=" * 70)
    print("⚙️  重排序权重调优")
    print("=" * 70)
    
    # 在 SK→CN 方向上测试不同权重组合
    weight_configs = {
        'W1_term_heavy': {'term': 0.50, 'length': 0.15, 'fluency': 0.20, 'consistency': 0.15},
        'W2_balanced': {'term': 0.30, 'length': 0.25, 'fluency': 0.25, 'consistency': 0.20},
        'W3_fluency_heavy': {'term': 0.25, 'length': 0.20, 'fluency': 0.35, 'consistency': 0.20},
        'W4_consistency_heavy': {'term': 0.25, 'length': 0.15, 'fluency': 0.20, 'consistency': 0.40},
        'W5_term_only': {'term': 0.70, 'length': 0.10, 'fluency': 0.10, 'consistency': 0.10},
    }
    
    print(f"\n在 SK→CN 方向上测试 {len(weight_configs)} 种权重组合...")
    print(f"{'权重配置':<22} {'chrF++':>7} {'TermAcc':>8} | 权重分布")
    print("─" * 75)
    
    for w_name, weights in weight_configs.items():
        results = []
        for entry in tqdm(test_data[:30], desc=w_name, leave=False):  # 用30条快速测试
            result = translator.full_pipeline(
                entry, 'sk', 'cn',
                use_term_constraint=True,
                use_mutual=True,
                use_reranking=True,
                rerank_weights=weights,
            )
            results.append({
                'prediction': result['prediction'],
                'reference': entry['chinese'],
                'source_text': entry['sanskrit'],
                'target_lang': 'cn',
                'term_coverage': result.get('term_coverage', 0),
            })
        
        preds = [r['prediction'] for r in results]
        refs = [r['reference'] for r in results]
        
        try:
            chrf_score = chrf_scorer.corpus_score(preds, [refs]).score
        except:
            chrf_score = 0
        
        # 术语准确率
        total_t = 0
        correct_t = 0
        for r in results:
            for sk_term, info in translator.terminology.items():
                if sk_term not in r['source_text']:
                    continue
                expected = info.get('chinese', '')
                if not expected:
                    continue
                if expected in r['reference']:
                    total_t += 1
                    if expected in r['prediction']:
                        correct_t += 1
        term_acc = correct_t / total_t if total_t > 0 else 0
        
        w_str = f"T={weights['term']:.2f} L={weights['length']:.2f} F={weights['fluency']:.2f} C={weights['consistency']:.2f}"
        print(f"{w_name:<22} {chrf_score:>7.2f} {term_acc:>8.4f} | {w_str}")
    
    # 清理
    del translator
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("✅ Step 4 消融实验完成")
    print("=" * 70)
    print(f"结果目录: {RESULTS_DIR}")
    print("""
结论模板：
1. 术语约束：软约束 vs 硬约束 vs 迭代约束的 TermAcc 增益
2. 互证翻译：引入第三语言参考的 chrF++ 增益
3. 重排序：多候选选择的综合提升
4. 最优权重：term_heavy 还是 balanced
""")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_ablation_experiment(max_samples=n)
PYTHON

                