#!/usr/bin/env python3
"""
带 Redis 缓存的完整推理流程
首次运行：正常推理并缓存结果
后续运行：直接从缓存读取，跳过已有结果
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 导入缓存模块
from redis_cache import TranslationCache, CachedTranslator

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA = f'{BASE_DIR}/saves/qwen_lora_v3_augmented'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/cached_eval'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class CachedFullEvaluator:
    """带缓存的完整评估器"""

    def __init__(self):
        # 初始化缓存
        self.cache = TranslationCache(prefix='ramayana', ttl=86400*90)  # 90天过期

        # 加载术语表
        with open(f'{DATA_DIR}/terminology_final.json', 'r', encoding='utf-8') as f:
            self.terminology = json.load(f)

        # 加载模型
        print("🔄 加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE, quantization_config=quant_config,
            device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)

        lora_path = QWEN_LORA if Path(QWEN_LORA).exists() else f'{BASE_DIR}/saves/qwen_lora_v2'
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()
        print(f"✓ 模型加载完成, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

        # 包装为带缓存的翻译器
        self.cached_translator = CachedTranslator(self, self.cache)

    def generate(self, prompt, max_new_tokens=256, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=0.9, do_sample=True, repetition_penalty=1.2,
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

    def run_cached_evaluation(self, test_data: List[Dict], directions: List[tuple]):
        """带缓存的评估"""
        from sacrebleu.metrics import CHRF
        chrf_scorer = CHRF(word_order=2)

        print(f"\n{'方向':<10} {'方法':<8} {'chrF++':>7} {'TermAcc':>8} {'缓存命中':>8} {'耗时':>8}")
        print("-" * 60)

        all_results = {}
        total_start = time.time()

        for sl, tl in directions:
            pivot_lang = list({'sk', 'tb', 'cn'} - {sl, tl})[0]

            for method in ['direct', 'mutual']:
                start = time.time()
                preds = []
                cache_hits = 0

                for entry in tqdm(test_data, desc=f"{sl}->{tl} {method}", leave=False):
                    src = entry[LANG_FIELDS[sl]]
                    pivot = entry[LANG_FIELDS[pivot_lang]] if method == 'mutual' else None

                    pred = self.cached_translator.translate(
                        src, sl, tl, method=method, pivot_text=pivot
                    )
                    preds.append(pred)

                elapsed = time.time() - start
                refs = [entry[LANG_FIELDS[tl]] for entry in test_data]
                srcs = [entry[LANG_FIELDS[sl]] for entry in test_data]

                chrf = chrf_scorer.corpus_score(preds, [refs]).score

                # 术语准确率
                target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[tl]
                tot, cor = 0, 0
                for pred, ref, src in zip(preds, refs, srcs):
                    for sk, info in self.terminology.items():
                        if sk not in src and info.get('tibetan', '') not in src and info.get('chinese', '') not in src:
                            continue
                        exp = sk if tl == 'sk' else info.get(target_key, '')
                        if not exp or exp not in ref:
                            continue
                        tot += 1
                        if exp in pred:
                            cor += 1
                ta = cor / tot if tot > 0 else 0

                # 缓存统计
                stats = self.cache.get_stats()
                hit_rate = stats['hit_rate']

                print(f"{sl}->{tl:<5} {method:<8} {chrf:>7.2f} {ta:>8.4f} {hit_rate:>8} {elapsed:>7.1f}s")

                all_results[f"{sl}->{tl}_{method}"] = {
                    'chrf': chrf, 'term_acc': ta, 'time': elapsed
                }

                # 保存详细结果
                detail_path = f"{RESULTS_DIR}/{sl}_to_{tl}_{method}.jsonl"
                with open(detail_path, 'w', encoding='utf-8') as f:
                    for i, (pred, ref, src) in enumerate(zip(preds, refs, srcs)):
                        f.write(json.dumps({
                            'id': test_data[i]['id'],
                            'source_lang': sl, 'target_lang': tl,
                            'method': method,
                            'source': src, 'reference': ref, 'prediction': pred,
                        }, ensure_ascii=False) + '\n')

        total_time = time.time() - total_start
        print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

        # 最终缓存统计
        self.cache.print_stats()

        # 保存汇总
        with open(f'{RESULTS_DIR}/evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return all_results

def main():
    print("=" * 70)
    print("⚡ Redis 缓存加速评估")
    print("=" * 70)

    # 加载测试数据
    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    import sys
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    test_data = test_data[:max_samples]
    print(f"测试集: {len(test_data)} 条")

    directions = [
        ('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb'),
        ('cn', 'tb'), ('cn', 'sk'), ('tb', 'sk'),
    ]

    evaluator = CachedFullEvaluator()
    results = evaluator.run_cached_evaluation(test_data, directions)

    # 第二次运行演示缓存加速
    print("\n" + "=" * 70)
    print("⚡ 第二次运行（全部从缓存读取）")
    print("=" * 70)
    results2 = evaluator.run_cached_evaluation(test_data, directions)

    del evaluator
    torch.cuda.empty_cache()
    print("\n✅ 完成")

if __name__ == "__main__":
    main()
