#!/usr/bin/env python3
"""
Step 5: 自迭代三角数据增强实验
核心思路：
1. 用当前模型对单语/双语数据生成伪平行译文
2. 通过三角一致性过滤高质量伪平行对
3. 加入训练集重新微调
4. 对比增强前后的翻译质量
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import json
import re
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = '/root/autodl-tmp/LlamaFactory-main'
DATA_DIR = f'{BASE_DIR}/data_preparation/data'
QWEN_BASE = f'{BASE_DIR}/models/Qwen2.5-7B-Instruct'
QWEN_LORA = f'{BASE_DIR}/saves/qwen_lora_v2'
RESULTS_DIR = f'{BASE_DIR}/baseline/results/step5_augmentation'
AUGMENT_DIR = f'{BASE_DIR}/data_preparation/data/augmented'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(AUGMENT_DIR).mkdir(parents=True, exist_ok=True)

LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

class TriangleAugmentor:
    """三角一致性数据增强器"""

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
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        for m in ['**解释', '解释：', '注释：', '\n\n---']:
            if m in text:
                text = text[:text.index(m)]
        return re.sub(r'\*\*(.+?)\*\*', r'\1', text).strip()

    def _format_prompt(self, user_message):
        messages = [{"role": "system", "content": "你是梵藏汉古典文献翻译专家。"},
                    {"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def translate(self, text, source_lang, target_lang):
        """单方向翻译"""
        sn, tn = LANG_NAMES[source_lang], LANG_NAMES[target_lang]
        prompt = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。\n\n{sn}：{text}"
        return self.generate(self._format_prompt(prompt))

    def compute_triangle_consistency(self, entry: Dict) -> Dict:
        """
        计算三角一致性得分
        对于三元组 (SK, TB, CN)，验证：
        1. SK→CN 与 SK→TB→CN 的一致性
        2. SK→TB 与 SK→CN→TB 的一致性
        3. TB→CN 与 TB→SK→CN 的一致性
        """
        sk = entry.get('sanskrit', '')
        tb = entry.get('tibetan', '')
        cn = entry.get('chinese', '')

        scores = {}

        # 路径1: SK→CN vs SK→TB→CN
        if sk and tb:
            direct_cn = self.translate(sk, 'sk', 'cn')
            pivot_cn = self.translate(tb, 'tb', 'cn')  # 用已有TB翻译
            scores['sk_cn_consistency'] = self._text_similarity(direct_cn, pivot_cn)
            scores['direct_cn'] = direct_cn
            scores['pivot_cn'] = pivot_cn

        # 路径2: SK→TB vs SK→CN→TB
        if sk and cn:
            direct_tb = self.translate(sk, 'sk', 'tb')
            pivot_tb = self.translate(cn, 'cn', 'tb')
            scores['sk_tb_consistency'] = self._text_similarity(direct_tb, pivot_tb)
            scores['direct_tb'] = direct_tb
            scores['pivot_tb'] = pivot_tb

        # 路径3: 回译一致性 SK→CN→SK
        if sk:
            forward_cn = self.translate(sk, 'sk', 'cn')
            back_sk = self.translate(forward_cn, 'cn', 'sk')
            scores['backtrans_consistency'] = self._text_similarity(sk, back_sk)
            scores['back_sk'] = back_sk

        # 综合得分
        valid_scores = [v for k, v in scores.items() if k.endswith('_consistency') and isinstance(v, float)]
        scores['overall'] = np.mean(valid_scores) if valid_scores else 0.0

        return scores

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度（字符级 Jaccard + 长度比）"""
        if not text1 or not text2:
            return 0.0

        # 字符级 bigram 重叠
        def get_bigrams(text):
            text = text.replace(' ', '').replace('།', '').replace('。', '').replace('，', '')
            return set(text[i:i+2] for i in range(len(text)-1)) if len(text) > 1 else set()

        bg1 = get_bigrams(text1)
        bg2 = get_bigrams(text2)

        if not bg1 or not bg2:
            return 0.0

        jaccard = len(bg1 & bg2) / len(bg1 | bg2)

        # 长度比
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))

        return 0.7 * jaccard + 0.3 * len_ratio

    def generate_pseudo_parallel(self, entries: List[Dict], n_augment: int = 500) -> List[Dict]:
        """
        生成伪平行数据：
        1. 从训练集中选取条目
        2. 用模型生成缺失方向的翻译
        3. 通过三角一致性过滤
        """
        print(f"\n{'='*60}")
        print(f"📦 生成伪平行数据 (目标: {n_augment} 条)")
        print(f"{'='*60}")

        # 选取源数据
        source_entries = random.sample(entries, min(n_augment * 2, len(entries)))

        augmented = []
        filtered_out = 0

        for entry in tqdm(source_entries, desc="Generating pseudo-parallel"):
            if len(augmented) >= n_augment:
                break

            # 策略1: 用SK生成新的CN和TB翻译
            sk = entry['sanskrit']
            new_cn = self.translate(sk, 'sk', 'cn')
            new_tb = self.translate(sk, 'sk', 'tb')

            # 三角一致性验证
            # 验证: new_cn 与 通过TB翻译得到的CN 是否一致
            verify_cn = self.translate(entry['tibetan'], 'tb', 'cn')
            consistency = self._text_similarity(new_cn, verify_cn)

            if consistency >= 0.15:  # 阈值
                augmented.append({
                    'id': f"aug_{entry['id']}",
                    'sanskrit': sk,
                    'tibetan': new_tb,
                    'chinese': new_cn,
                    'consistency_score': consistency,
                    'source': 'triangle_augment',
                    'original_tb': entry['tibetan'],
                    'original_cn': entry['chinese'],
                })
            else:
                filtered_out += 1

        print(f"\n✓ 生成: {len(augmented)} 条伪平行数据")
        print(f"  过滤: {filtered_out} 条 (一致性 < 0.15)")
        print(f"  通过率: {len(augmented)/(len(augmented)+filtered_out)*100:.1f}%")

        # 统计一致性分布
        if augmented:
            scores = [a['consistency_score'] for a in augmented]
            print(f"  一致性得分: mean={np.mean(scores):.4f}, "
                  f"min={np.min(scores):.4f}, max={np.max(scores):.4f}")

        return augmented

    def generate_backtranslation_augment(self, entries: List[Dict], n_augment: int = 300) -> List[Dict]:
        """
        回译增强：
        1. CN → SK → CN'（验证CN'与原CN的一致性）
        2. TB → SK → TB'（验证TB'与原TB的一致性）
        """
        print(f"\n{'='*60}")
        print(f"📦 回译增强 (目标: {n_augment} 条)")
        print(f"{'='*60}")

        source_entries = random.sample(entries, min(n_augment * 2, len(entries)))
        augmented = []

        for entry in tqdm(source_entries, desc="Back-translation"):
            if len(augmented) >= n_augment:
                break

            # CN → SK → CN'
            original_cn = entry['chinese']
            generated_sk = self.translate(original_cn, 'cn', 'sk')
            back_cn = self.translate(generated_sk, 'sk', 'cn')

            cn_consistency = self._text_similarity(original_cn, back_cn)

            # TB → SK → TB'
            original_tb = entry['tibetan']
            generated_sk2 = self.translate(original_tb, 'tb', 'sk')
            back_tb = self.translate(generated_sk2, 'sk', 'tb')

            tb_consistency = self._text_similarity(original_tb, back_tb)

            avg_consistency = (cn_consistency + tb_consistency) / 2

            if avg_consistency >= 0.12:
                augmented.append({
                    'id': f"bt_{entry['id']}",
                    'sanskrit': entry['sanskrit'],
                    'tibetan': back_tb,
                    'chinese': back_cn,
                    'consistency_score': avg_consistency,
                    'cn_consistency': cn_consistency,
                    'tb_consistency': tb_consistency,
                    'source': 'backtranslation',
                })

        print(f"✓ 回译增强: {len(augmented)} 条")
        return augmented

    def convert_to_training_format(self, augmented_entries: List[Dict]) -> List[Dict]:
        """将增强数据转换为训练格式"""
        samples = []

        directions = [
            ('sk', 'cn', None),
            ('sk', 'tb', None),
            ('tb', 'cn', None),
            ('sk', 'cn', 'tb'),  # 互证
            ('tb', 'cn', 'sk'),  # 互证
        ]

        for entry in augmented_entries:
            for source, target, pivot in directions:
                source_text = entry.get(LANG_FIELDS[source], '')
                target_text = entry.get(LANG_FIELDS[target], '')

                if not source_text or not target_text:
                    continue

                sn, tn = LANG_NAMES[source], LANG_NAMES[target]
                user_content = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。\n\n{sn}：{source_text}"

                if pivot:
                    pivot_text = entry.get(LANG_FIELDS[pivot], '')
                    if pivot_text:
                        pn = LANG_NAMES[pivot]
                        user_content = (f"请将以下{sn}翻译为{tn}，参考{pn}辅助理解。只输出翻译。"
                                       f"\n\n{sn}：{source_text}\n{pn}参考：{pivot_text}")

                samples.append({
                    'messages': [
                        {'role': 'system', 'content': '你是梵藏汉古典文献翻译专家。'},
                        {'role': 'user', 'content': user_content},
                        {'role': 'assistant', 'content': target_text}
                    ]
                })

        random.shuffle(samples)
        return samples

def run_step5_experiment(max_augment=300, max_test=50):
    """Step 5 完整实验流程"""

    print("=" * 70)
    print("🚀 Step 5: 自迭代三角数据增强实验")
    print("=" * 70)

    # 加载数据
    with open(f'{DATA_DIR}/splits/train.jsonl', 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]

    with open(f'{DATA_DIR}/splits/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:max_test]

    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {max_test} 条")

    # 初始化增强器
    augmentor = TriangleAugmentor()

    # ============================================================
    # Phase 1: 生成伪平行数据
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 1: 生成伪平行数据")
    print("=" * 70)

    # 三角增强
    triangle_augmented = augmentor.generate_pseudo_parallel(
        train_data[:2000], n_augment=max_augment
    )

    # 回译增强
    backtrans_augmented = augmentor.generate_backtranslation_augment(
        train_data[:1000], n_augment=max_augment // 2
    )

    all_augmented = triangle_augmented + backtrans_augmented
    print(f"\n总增强数据: {len(all_augmented)} 条")
    print(f"  三角增强: {len(triangle_augmented)} 条")
    print(f"  回译增强: {len(backtrans_augmented)} 条")

    # 保存增强数据
    aug_path = f"{AUGMENT_DIR}/augmented_data.jsonl"
    with open(aug_path, 'w', encoding='utf-8') as f:
        for entry in all_augmented:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"💾 增强数据: {aug_path}")

    # ============================================================
    # Phase 2: 三角一致性分析
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 2: 三角一致性分析")
    print("=" * 70)

    # 对测试集计算三角一致性
    consistency_results = []
    for entry in tqdm(test_data[:20], desc="Triangle consistency"):
        scores = augmentor.compute_triangle_consistency(entry)
        consistency_results.append({
            'id': entry['id'],
            **{k: v for k, v in scores.items() if isinstance(v, float)}
        })

    # 统计
    if consistency_results:
        overall_scores = [r['overall'] for r in consistency_results]
        print(f"\n三角一致性统计 (测试集 {len(consistency_results)} 条):")
        print(f"  overall: mean={np.mean(overall_scores):.4f}, std={np.std(overall_scores):.4f}")

        if any('sk_cn_consistency' in r for r in consistency_results):
            sk_cn = [r['sk_cn_consistency'] for r in consistency_results if 'sk_cn_consistency' in r]
            print(f"  SK→CN一致性: mean={np.mean(sk_cn):.4f}")

        if any('backtrans_consistency' in r for r in consistency_results):
            bt = [r['backtrans_consistency'] for r in consistency_results if 'backtrans_consistency' in r]
            print(f"  回译一致性: mean={np.mean(bt):.4f}")

    # 保存一致性结果
    cons_path = f"{RESULTS_DIR}/triangle_consistency.json"
    with open(cons_path, 'w', encoding='utf-8') as f:
        json.dump(consistency_results, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Phase 3: 增强数据质量评估
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 3: 增强数据质量评估")
    print("=" * 70)

    # 对比增强数据与原始数据的翻译质量
    # 用增强数据中的 original_cn 作为参考，评估生成的 chinese
    if triangle_augmented:
        from sacrebleu.metrics import CHRF
        chrf_scorer = CHRF(word_order=2)

        # 三角增强质量
        aug_preds = [a['chinese'] for a in triangle_augmented if 'original_cn' in a]
        aug_refs = [a['original_cn'] for a in triangle_augmented if 'original_cn' in a]

        if aug_preds and aug_refs:
            aug_chrf = chrf_scorer.corpus_score(aug_preds, [aug_refs]).score
            print(f"三角增强数据质量 (vs 原始翻译):")
            print(f"  chrF++: {aug_chrf:.2f}")
            print(f"  平均一致性: {np.mean([a['consistency_score'] for a in triangle_augmented]):.4f}")

    # ============================================================
    # Phase 4: 转换为训练格式并保存
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 4: 转换训练数据")
    print("=" * 70)

    training_samples = augmentor.convert_to_training_format(all_augmented)
    print(f"增强训练样本: {len(training_samples)} 条")

    # 保存增强训练数据
    aug_train_path = f"{AUGMENT_DIR}/train_augmented.jsonl"
    with open(aug_train_path, 'w', encoding='utf-8') as f:
        for s in training_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"💾 增强训练数据: {aug_train_path}")

    # 合并原始 + 增强数据
    original_train_path = f"{DATA_DIR}/train_qwen_medium.jsonl"
    if Path(original_train_path).exists():
        with open(original_train_path, 'r', encoding='utf-8') as f:
            original_samples = [json.loads(line) for line in f]

        combined = original_samples + training_samples
        random.shuffle(combined)

        combined_path = f"{AUGMENT_DIR}/train_combined.jsonl"
        with open(combined_path, 'w', encoding='utf-8') as f:
            for s in combined:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"💾 合并训练数据: {combined_path} ({len(combined)} 条)")
        print(f"  原始: {len(original_samples)} + 增强: {len(training_samples)}")

    # ============================================================
    # Phase 5: 增强前后对比评估
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 5: 增强效果评估 (使用当前模型)")
    print("=" * 70)

    # 用当前模型在测试集上评估（作为增强前基线）
    from sacrebleu.metrics import CHRF
    chrf_scorer = CHRF(word_order=2)

    directions = [('sk', 'cn'), ('tb', 'cn'), ('sk', 'tb')]

    print(f"\n{'方向':<8} {'chrF++':>7} {'TermAcc':>8} | 当前模型(增强前)")
    print("-" * 50)

    for sl, tl in directions:
        preds, refs = [], []
        term_total, term_correct = 0, 0
        target_key = {'sk': None, 'tb': 'tibetan', 'cn': 'chinese'}[tl]

        for entry in tqdm(test_data[:30], desc=f"{sl}->{tl}", leave=False):
            pred = augmentor.translate(entry[LANG_FIELDS[sl]], sl, tl)
            preds.append(pred)
            refs.append(entry[LANG_FIELDS[tl]])

            # 术语
            for sk, info in augmentor.terminology.items():
                if sk not in entry[LANG_FIELDS[sl]]:
                    continue
                exp = sk if tl == 'sk' else info.get(target_key, '')
                if not exp:
                    continue
                if exp in entry[LANG_FIELDS[tl]]:
                    term_total += 1
                    if exp in pred:
                        term_correct += 1

        chrf = chrf_scorer.corpus_score(preds, [refs]).score
        ta = term_correct / term_total if term_total > 0 else 0
        print(f"{sl}->{tl:<5} {chrf:>7.2f} {ta:>8.4f}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ Step 5 数据增强实验完成")
    print("=" * 70)
    print(f"""
结果文件:
├── {AUGMENT_DIR}/
│   ├── augmented_data.jsonl        ({len(all_augmented)} 条增强数据)
│   ├── train_augmented.jsonl       ({len(training_samples)} 条训练格式)
│   └── train_combined.jsonl        (原始+增强合并)
├── {RESULTS_DIR}/
│   └── triangle_consistency.json   (三角一致性分析)

下一步：
1. 用 train_combined.jsonl 重新微调模型 (增强后版本)
2. 对比增强前后的翻译质量
3. 分析不同一致性阈值对增强效果的影响
""")

    # 清理
    del augmentor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run_step5_experiment(max_augment=n, max_test=50)
