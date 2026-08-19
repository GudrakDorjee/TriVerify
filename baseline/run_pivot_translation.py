#!/usr/bin/env python3
"""
枢轴翻译实验：SK → TB → CN（两步翻译）
对比：直接翻译 vs 枢轴翻译 vs 互证翻译
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA = f'{BASE_DIR}/saves/qwen_lora_v2'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/pivot'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class PivotTranslator:
    """枢轴翻译器"""
    
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
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, top_p=0.9, do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._post_process(text)
    
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
    
    def _get_terms(self, text: str) -> str:
        relevant = []
        for sk_term, info in self.terminology.items():
            if sk_term in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
                relevant.append(f"{sk_term}={info.get('chinese','?')}/{info.get('tibetan','?')}")
        return '；'.join(relevant[:5]) if relevant else ''
    
    # ================================================================
    # 三种翻译方式
    # ================================================================
    
    def translate_direct(self, source_text: str, source_lang: str, 
                         target_lang: str) -> str:
        """直接翻译：source → target"""
        terms = self._get_terms(source_text)
        source_name = LANG_NAMES[source_lang]
        target_name = LANG_NAMES[target_lang]
        
        prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt += f"\n术语：{terms}"
        prompt += f"\n\n{source_name}：{source_text}"
        
        formatted = self._format_prompt(prompt)
        return self.generate(formatted)
    
    def translate_pivot(self, source_text: str, source_lang: str,
                        target_lang: str, pivot_lang: str) -> Dict:
        """枢轴翻译：source → pivot → target（两步）"""
        source_name = LANG_NAMES[source_lang]
        pivot_name = LANG_NAMES[pivot_lang]
        target_name = LANG_NAMES[target_lang]
        terms = self._get_terms(source_text)
        
        # 第一步：source → pivot
        prompt1 = f"请将以下{source_name}翻译为{pivot_name}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt1 += f"\n术语：{terms}"
        prompt1 += f"\n\n{source_name}：{source_text}"
        
        formatted1 = self._format_prompt(prompt1)
        pivot_translation = self.generate(formatted1)
        
        # 第二步：pivot → target
        prompt2 = f"请将以下{pivot_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt2 += f"\n术语：{terms}"
        prompt2 += f"\n\n{pivot_name}：{pivot_translation}"
        
        formatted2 = self._format_prompt(prompt2)
        final_translation = self.generate(formatted2)
        
        return {
            'pivot_text': pivot_translation,
            'final_text': final_translation
        }
    
    def translate_mutual(self, source_text: str, pivot_text: str,
                         source_lang: str, target_lang: str, 
                         pivot_lang: str) -> str:
        """互证翻译：source + pivot参考 → target"""
        source_name = LANG_NAMES[source_lang]
        target_name = LANG_NAMES[target_lang]
        pivot_name = LANG_NAMES[pivot_lang]
        terms = self._get_terms(source_text)
        
        prompt = (f"请将以下{source_name}翻译为{target_name}，参考{pivot_name}辅助理解语义。"
                 f"只输出翻译。")
        if terms:
            prompt += f"\n术语：{terms}"
        prompt += f"\n\n{source_name}：{source_text}\n{pivot_name}参考：{pivot_text}"
        
        formatted = self._format_prompt(prompt)
        return self.generate(formatted)

def run_comparison_experiment(max_samples: int = 50):
    """运行三种方法对比实验"""
    
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:max_samples]
    
    print("=" * 70)
    print("🧪 直接翻译 vs 枢轴翻译 vs 互证翻译 对比实验")
    print("=" * 70)
    print(f"测试样本: {max_samples} 条")
    
    translator = PivotTranslator()
    
    # 六个方向，每个方向三种方法
    directions = [
        ('sk', 'cn', 'tb'),  # 梵→汉，枢轴=藏
        ('tb', 'cn', 'sk'),  # 藏→汉，枢轴=梵
        ('sk', 'tb', 'cn'),  # 梵→藏，枢轴=汉
        ('cn', 'tb', 'sk'),  # 汉→藏，枢轴=梵
        ('cn', 'sk', 'tb'),  # 汉→梵，枢轴=藏
        ('tb', 'sk', 'cn'),  # 藏→梵，枢轴=汉
    ]
    
    all_results = {}
    
    for source_lang, target_lang, pivot_lang in directions:
        direction = f"{source_lang}→{target_lang}"
        pivot_path = f"{source_lang}→{pivot_lang}→{target_lang}"
        
        print(f"\n{'='*70}")
        print(f"📌 方向: {direction} (枢轴路径: {pivot_path})")
        print(f"{'='*70}")
        
        results_direct = []
        results_pivot = []
        results_mutual = []
        
        for entry in tqdm(test_data, desc=direction):
            source_text = entry[LANG_FIELDS[source_lang]]
            pivot_text = entry[LANG_FIELDS[pivot_lang]]  # 真实的第三语言文本
            reference = entry[LANG_FIELDS[target_lang]]
            
            # 1. 直接翻译
            pred_direct = translator.translate_direct(source_text, source_lang, target_lang)
            results_direct.append({
                'id': entry['id'],
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': source_text,
                'reference': reference,
                'prediction': pred_direct,
                'model': 'qwen_lora_v2',
                'method': 'direct',
            })
            
            # 2. 枢轴翻译（两步）
            pivot_result = translator.translate_pivot(
                source_text, source_lang, target_lang, pivot_lang
            )
            results_pivot.append({
                'id': entry['id'],
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': source_text,
                'reference': reference,
                'prediction': pivot_result['final_text'],
                'pivot_translation': pivot_result['pivot_text'],
                'model': 'qwen_lora_v2',
                'method': 'pivot',
                'pivot_lang': pivot_lang,
            })
            
            # 3. 互证翻译（使用真实的第三语言参考）
            pred_mutual = translator.translate_mutual(
                source_text, pivot_text, source_lang, target_lang, pivot_lang
            )
            results_mutual.append({
                'id': entry['id'],
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': source_text,
                'reference': reference,
                'prediction': pred_mutual,
                'model': 'qwen_lora_v2',
                'method': 'mutual',
                'pivot_lang': pivot_lang,
            })
        
        # 保存三种方法的结果
        for method, results in [('direct', results_direct), 
                                 ('pivot', results_pivot), 
                                 ('mutual', results_mutual)]:
            path = f"{RESULTS_DIR}/{source_lang}_to_{target_lang}_{method}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        # 快速预览
        print(f"  直接: {results_direct[0]['prediction'][:60]}")
        print(f"  枢轴: {results_pivot[0]['prediction'][:60]}")
        print(f"  互证: {results_mutual[0]['prediction'][:60]}")
        print(f"  参考: {results_direct[0]['reference'][:60]}")
        
        all_results[direction] = {
            'direct': results_direct,
            'pivot': results_pivot,
            'mutual': results_mutual,
        }
    
    # ============================================================
    # 即时评估
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 三种方法对比评估")
    print("=" * 70)
    
    try:
        from sacrebleu.metrics import BLEU, CHRF
        bleu_scorer = BLEU(effective_order=True)
        chrf_scorer = CHRF(word_order=2)
        
        print(f"\n{'方向':<10} {'方法':<10} {'BLEU':>7} {'chrF++':>8} {'TermAcc':>8}")
        print("─" * 50)
        
        summary = {}
        
        for direction, methods in all_results.items():
            for method_name, results in methods.items():
                preds = [r['prediction'] for r in results]
                refs = [r['reference'] for r in results]
                
                bleu_score = bleu_scorer.corpus_score(preds, [refs]).score
                chrf_score = chrf_scorer.corpus_score(preds, [refs]).score
                
                # 术语准确率
                total_terms = 0
                correct_terms = 0
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
                            total_terms += 1
                            if expected in r['prediction']:
                                correct_terms += 1
                
                term_acc = correct_terms / total_terms if total_terms > 0 else 0
                
                print(f"{direction:<10} {method_name:<10} {bleu_score:>7.2f} {chrf_score:>8.2f} {term_acc:>8.4f}")
                
                key = f"{direction}_{method_name}"
                summary[key] = {'bleu': bleu_score, 'chrf': chrf_score, 'term_acc': term_acc}
            
            print("─" * 50)
        
        # 方法平均
        print(f"\n{'方法':<10} {'平均BLEU':>9} {'平均chrF++':>10} {'平均TermAcc':>11}")
        print("─" * 45)
        for method in ['direct', 'pivot', 'mutual']:
            method_scores = [v for k, v in summary.items() if method in k]
            if method_scores:
                avg_bleu = np.mean([s['bleu'] for s in method_scores])
                avg_chrf = np.mean([s['chrf'] for s in method_scores])
                avg_term = np.mean([s['term_acc'] for s in method_scores])
                print(f"{method:<10} {avg_bleu:>9.2f} {avg_chrf:>10.2f} {avg_term:>11.4f}")
        
    except ImportError:
        print("⚠️  sacrebleu 未安装，跳过自动评估")
    
    del translator
    torch.cuda.empty_cache()
    
    print(f"\n✅ 对比实验完成")
    print(f"结果目录: {RESULTS_DIR}")

if __name__ == "__main__":
    import sys
    import numpy as np
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_comparison_experiment(max_samples=n)
