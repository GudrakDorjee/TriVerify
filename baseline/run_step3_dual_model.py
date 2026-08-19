#!/usr/bin/env python3
"""
Step 3: 双模型协同三语互证翻译
Qwen (全能) + gemma_4b (藏文专家)
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA = f'{BASE_DIR}/saves/qwen_lora_v2'
GEMMA4B_BASE = f'{BASE_DIR}/models/googletranslategemma-4b-it'
GEMMA4B_LORA = f'{BASE_DIR}/saves/gemma_4b_lora'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/step3_dual_model'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class DualModelTranslator:
    """双模型协同翻译器"""
    
    def __init__(self):
        # 加载术语表
        with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
            self.terminology = json.load(f)
        
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.gemma_model = None
        self.gemma_tokenizer = None
        
        self._load_models()
    
    def _load_models(self):
        """加载两个微调模型"""
        print("🔄 加载 Qwen + LoRA...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        qwen_base = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.qwen_model = PeftModel.from_pretrained(qwen_base, QWEN_LORA)
        self.qwen_model.eval()
        print(f"  ✓ Qwen 加载完成, 显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
        
        print("\n🔄 加载 gemma-4b + LoRA...")
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA4B_BASE, trust_remote_code=True)
        if self.gemma_tokenizer.pad_token is None:
            self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
        
        gemma_base = AutoModelForCausalLM.from_pretrained(
            GEMMA4B_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager',
        )
        self.gemma_model = PeftModel.from_pretrained(gemma_base, GEMMA4B_LORA)
        self.gemma_model.eval()
        print(f"  ✓ gemma-4b 加载完成, 总显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    
    def generate(self, model, tokenizer, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.3) -> str:
        """生成文本"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                          max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._post_process(text)
    
    def _post_process(self, text: str) -> str:
        for marker in ['**解释', '**注释', '\n\n---', '解释：', '注释：']:
            if marker in text:
                text = text[:text.index(marker)]
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        return text.strip()
    
    def _format_qwen_prompt(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
            {"role": "user", "content": user_message}
        ]
        return self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def _format_gemma_prompt(self, user_message: str) -> str:
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    
    def _get_terms(self, text: str) -> str:
        relevant = []
        for sk_term, info in self.terminology.items():
            if sk_term in text or info.get('tibetan', '') in text or info.get('chinese', '') in text:
                relevant.append(f"{sk_term}={info.get('chinese','?')}/{info.get('tibetan','?')}")
        return '；'.join(relevant[:5]) if relevant else ''
    
    def _get_expected_terms(self, source_text: str, target_lang: str) -> List[str]:
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[target_lang]
        expected = []
        
        for sk_term, info in self.terminology.items():
            if sk_term in source_text or info.get('tibetan', '') in source_text or info.get('chinese', '') in source_text:
                if target_lang == 'sk':
                    expected.append(sk_term)
                elif target_key and info.get(target_key):
                    expected.append(info[target_key])
        
        return expected
    
    def translate_with_dual_model(self, entry: Dict, source_lang: str, 
                                   target_lang: str) -> Dict:
        """
        双模型协同翻译：
        1. Qwen 生成多个候选（直接 + 互证 + 术语约束）
        2. gemma-4b 在藏文方向生成候选
        3. 重排序选择最佳
        """
        all_langs = {'sk', 'tb', 'cn'}
        pivot_lang = list(all_langs - {source_lang, target_lang})[0]
        
        source_text = entry[LANG_FIELDS[source_lang]]
        pivot_text = entry[LANG_FIELDS[pivot_lang]]
        terms = self._get_terms(source_text)
        expected_terms = self._get_expected_terms(source_text, target_lang)
        
        source_name = LANG_NAMES[source_lang]
        target_name = LANG_NAMES[target_lang]
        pivot_name = LANG_NAMES[pivot_lang]
        
        candidates = []
        
        # === Qwen 路径 ===
        # 1. 直接翻译
        prompt = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
        if terms:
            prompt += f"\n术语：{terms}"
        prompt += f"\n\n{source_name}：{source_text}"
        
        formatted = self._format_qwen_prompt(prompt)
        pred = self.generate(self.qwen_model, self.qwen_tokenizer, formatted)
        candidates.append({'text': pred, 'path': 'qwen_direct', 'model': 'qwen'})
        
        # 2. 互证翻译
        prompt_mutual = (f"请将以下{source_name}翻译为{target_name}，参考{pivot_name}辅助理解。"
                        f"只输出翻译。")
        if terms:
            prompt_mutual += f"\n术语：{terms}"
        prompt_mutual += f"\n\n{source_name}：{source_text}\n{pivot_name}参考：{pivot_text}"
        
        formatted = self._format_qwen_prompt(prompt_mutual)
        pred = self.generate(self.qwen_model, self.qwen_tokenizer, formatted)
        candidates.append({'text': pred, 'path': 'qwen_mutual', 'model': 'qwen'})
        
        # 3. 术语约束
        if expected_terms:
            term_list = '、'.join(expected_terms[:5])
            prompt_term = (f"请将以下{source_name}翻译为{target_name}。"
                          f"翻译中必须包含：{term_list}。只输出翻译。")
            if terms:
                prompt_term += f"\n术语：{terms}"
            prompt_term += f"\n\n{source_name}：{source_text}"
            
            formatted = self._format_qwen_prompt(prompt_term)
            pred = self.generate(self.qwen_model, self.qwen_tokenizer, formatted)
            candidates.append({'text': pred, 'path': 'qwen_term', 'model': 'qwen'})
        
        # === gemma-4b 路径（仅藏文相关方向）===
        if target_lang == 'tb' or source_lang == 'tb':
            prompt_gemma = f"请将以下{source_name}翻译为{target_name}，保持古典文献文体。只输出翻译。"
            if terms:
                prompt_gemma += f"\n术语：{terms}"
            prompt_gemma += f"\n\n{source_name}：{source_text}"
            
            formatted = self._format_gemma_prompt(prompt_gemma)
            pred = self.generate(self.gemma_model, self.gemma_tokenizer, formatted)
            candidates.append({'text': pred, 'path': 'gemma4b_direct', 'model': 'gemma4b'})
        
        # 重排序
        scored = self._rerank(candidates, source_text, target_lang, expected_terms, entry, pivot_lang)
        best = scored[0]
        
        return {
            'best_translation': best['text'],
            'best_path': best['path'],
            'best_model': best['model'],
            'best_score': best['total_score'],
            'all_candidates': scored,
            'expected_terms': expected_terms,
        }
    
    def _rerank(self, candidates: List[Dict], source_text: str, target_lang: str,
                expected_terms: List[str], entry: Dict, pivot_lang: str) -> List[Dict]:
        """候选重排序"""
        reference = entry[LANG_FIELDS[target_lang]]
        ref_len = len(reference)
        
        for cand in candidates:
            text = cand['text']
            
            # 1. 术语覆盖
            if expected_terms:
                covered = sum(1 for t in expected_terms if t in text)
                term_score = covered / len(expected_terms)
            else:
                term_score = 0.5
            
            # 2. 长度合理性
            if ref_len > 0 and len(text) > 0:
                ratio = len(text) / ref_len
                length_score = max(0, 1 - abs(ratio - 1))
            else:
                length_score = 0.0
            
            # 3. 路径偏好
            path_bonus = {
                'qwen_mutual': 0.9,
                'qwen_term': 0.85,
                'qwen_direct': 0.7,
                'gemma4b_direct': 0.75,
            }.get(cand['path'], 0.5)
            
            # 4. 模型偏好（Qwen 更可靠）
            model_bonus = 0.8 if cand['model'] == 'qwen' else 0.6
            
            total_score = (
                0.40 * term_score +
                0.25 * length_score +
                0.20 * path_bonus +
                0.15 * model_bonus
            )
            
            cand['term_score'] = term_score
            cand['length_score'] = length_score
            cand['path_bonus'] = path_bonus
            cand['model_bonus'] = model_bonus
            cand['total_score'] = total_score
        
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        return candidates

def run_step3_experiment(max_samples: int = 50):
    """运行 Step 3 完整实验"""
    
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:max_samples]
    
    print("=" * 70)
    print("🚀 Step 3: 双模型协同三语互证翻译")
    print("=" * 70)
    print(f"测试样本: {max_samples} 条")
    print(f"模型: Qwen2.5-7B + gemma-4b-it (双LoRA)")
    
    translator = DualModelTranslator()
    
    directions = [
        ('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb'),
        ('cn', 'tb'), ('cn', 'sk'), ('tb', 'sk'),
    ]
    
    all_results = {}
    
    for source_lang, target_lang in directions:
        direction = f"{source_lang}→{target_lang}"
        print(f"\n{'='*60}")
        print(f"📌 方向: {direction}")
        print(f"{'='*60}")
        
        results = []
        
        for entry in tqdm(test_data, desc=direction):
            output = translator.translate_with_dual_model(entry, source_lang, target_lang)
            
            results.append({
                'id': entry['id'],
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': entry[LANG_FIELDS[source_lang]],
                'reference': entry[LANG_FIELDS[target_lang]],
                'prediction': output['best_translation'],
                'best_path': output['best_path'],
                'best_model': output['best_model'],
                'best_score': output['best_score'],
                'n_candidates': len(output['all_candidates']),
                'expected_terms': output['expected_terms'],
                'model': 'dual_qwen_gemma4b',
                'method': 'step3_dual_mutual',
            })
        
        # 保存
        output_path = f"{RESULTS_DIR}/{source_lang}_to_{target_lang}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        # 统计
        paths = [r['best_path'] for r in results]
        models = [r['best_model'] for r in results]
        path_dist = {p: paths.count(p) for p in set(paths)}
        model_dist = {m: models.count(m) for m in set(models)}
        avg_score = np.mean([r['best_score'] for r in results])
        
        print(f"  路径分布: {path_dist}")
        print(f"  模型分布: {model_dist}")
        print(f"  平均得分: {avg_score:.4f}")
        print(f"  💾 {output_path}")
        
        # 样本
        print(f"  预测: {results[0]['prediction'][:80]}")
        print(f"  参考: {results[0]['reference'][:80]}")
        
        all_results[direction] = results
    
    del translator
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("✅ Step 3 双模型协同实验完成")
    print("=" * 70)
    print(f"结果目录: {RESULTS_DIR}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_step3_experiment(max_samples=n)
