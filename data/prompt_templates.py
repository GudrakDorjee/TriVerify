# data_preparation/prompt_templates.py
from typing import Dict, List
import json
import os

class PromptTemplateManager:
    """Prompt模板管理器"""

    def __init__(self, terminology_dict: Dict = None):
        self.terminology_dict = terminology_dict or {}
        self.templates = self._init_templates()

    def _init_templates(self) -> Dict:
        """初始化所有prompt模板"""
        return {
            # 直接翻译模板
            'direct_sk_to_cn': (
                "你是梵藏汉古典文献翻译专家。请将以下梵文翻译为汉文，保持古典文献的文体风格。\n\n"
                "术语参考：\n{terminology}\n\n"
                "梵文原文：\n{source_text}\n\n"
                "汉文翻译："
            ),
            'direct_sk_to_tb': (
                "你是梵藏汉古典文献翻译专家。请将以下梵文翻译为藏文。\n\n"
                "术语参考：\n{terminology}\n\n"
                "梵文原文：\n{source_text}\n\n"
                "藏文翻译："
            ),
            'direct_tb_to_cn': (
                "你是梵藏汉古典文献翻译专家。请将以下藏文翻译为汉文，保持古典文献的文体风格。\n\n"
                "术语参考：\n{terminology}\n\n"
                "藏文原文：\n{source_text}\n\n"
                "汉文翻译："
            ),
            'direct_tb_to_sk': (
                "你是梵藏汉古典文献翻译专家。请将以下藏文翻译为梵文。\n\n"
                "术语参考：\n{terminology}\n\n"
                "藏文原文：\n{source_text}\n\n"
                "梵文翻译："
            ),
            'direct_cn_to_sk': (
                "你是梵藏汉古典文献翻译专家。请将以下汉文翻译为梵文。\n\n"
                "术语参考：\n{terminology}\n\n"
                "汉文原文：\n{source_text}\n\n"
                "梵文翻译："
            ),
            'direct_cn_to_tb': (
                "你是梵藏汉古典文献翻译专家。请将以下汉文翻译为藏文。\n\n"
                "术语参考：\n{terminology}\n\n"
                "汉文原文：\n{source_text}\n\n"
                "藏文翻译："
            ),
            # 互证翻译模板（引入第三语言）
            'mutual_sk_to_cn_via_tb': (
                "你是梵藏汉古典文献翻译专家。请将以下梵文翻译为汉文。参考藏文译本辅助理解原文语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "梵文原文：\n{source_text}\n\n"
                "藏文参考译本：\n{pivot_text}\n\n"
                "请基于梵文原文和藏文参考，给出准确的汉文翻译："
            ),
            'mutual_sk_to_tb_via_cn': (
                "你是梵藏汉古典文献翻译专家。请将以下梵文翻译为藏文。参考汉文译本辅助理解原文语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "梵文原文：\n{source_text}\n\n"
                "汉文参考译本：\n{pivot_text}\n\n"
                "请基于梵文原文和汉文参考，给出准确的藏文翻译："
            ),
            'mutual_tb_to_cn_via_sk': (
                "你是梵藏汉古典文献翻译专家。请将以下藏文翻译为汉文。参考梵文原典辅助理解语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "藏文原文：\n{source_text}\n\n"
                "梵文参考原典：\n{pivot_text}\n\n"
                "请基于藏文原文和梵文参考，给出准确的汉文翻译："
            ),
            'mutual_tb_to_sk_via_cn': (
                "你是梵藏汉古典文献翻译专家。请将以下藏文翻译为梵文。参考汉文译本辅助理解语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "藏文原文：\n{source_text}\n\n"
                "汉文参考译本：\n{pivot_text}\n\n"
                "请基于藏文原文和汉文参考，给出准确的梵文翻译："
            ),
            'mutual_cn_to_sk_via_tb': (
                "你是梵藏汉古典文献翻译专家。请将以下汉文翻译为梵文。参考藏文译本辅助理解语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "汉文原文：\n{source_text}\n\n"
                "藏文参考译本：\n{pivot_text}\n\n"
                "请基于汉文原文和藏文参考，给出准确的梵文翻译："
            ),
            'mutual_cn_to_tb_via_sk': (
                "你是梵藏汉古典文献翻译专家。请将以下汉文翻译为藏文。参考梵文原典辅助理解语义。\n\n"
                "术语参考：\n{terminology}\n\n"
                "汉文原文：\n{source_text}\n\n"
                "梵文参考原典：\n{pivot_text}\n\n"
                "请基于汉文原文和梵文参考，给出准确的藏文翻译："
            ),
        }

    def get_relevant_terminology(self, text: str, max_terms: int = 5) -> str:
        """获取文本相关的术语"""
        if not self.terminology_dict:
            return "（无术语参考）"

        relevant = []
        for sk_term, translations in self.terminology_dict.items():
            if (sk_term in text or
                translations.get('tibetan', '') in text or
                translations.get('chinese', '') in text):
                relevant.append(
                    f"- {sk_term} = {translations.get('tibetan', '?')} = "
                    f"{translations.get('chinese', '?')} ({translations.get('category', 'term')})"
                )

        if not relevant:
            return "（本句无特殊术语）"

        # 限制术语数量，避免prompt过长
        return '\n'.join(relevant[:max_terms])

    def build_prompt(self, template_key: str, entry: Dict,
                     source_lang: str, target_lang: str,
                     pivot_lang: str = None) -> str:
        """构建完整prompt"""

        lang_map = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

        source_text = entry[lang_map[source_lang]]

        # 获取相关术语
        terminology = self.get_relevant_terminology(source_text)

        if pivot_lang and f'mutual_{source_lang}_to_{target_lang}_via_{pivot_lang}' in self.templates:
            # 互证翻译
            key = f'mutual_{source_lang}_to_{target_lang}_via_{pivot_lang}'
            pivot_text = entry[lang_map[pivot_lang]]
            prompt = self.templates[key].format(
                terminology=terminology,
                source_text=source_text,
                pivot_text=pivot_text
            )
        else:
            # 直接翻译
            key = f'direct_{source_lang}_to_{target_lang}'
            prompt = self.templates[key].format(
                terminology=terminology,
                source_text=source_text
            )

        return prompt

    def build_training_sample(self, entry: Dict, source_lang: str,
                              target_lang: str, pivot_lang: str = None) -> Dict:
        """构建训练样本（prompt + completion）"""

        lang_map = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

        prompt = self.build_prompt(
            template_key=None,
            entry=entry,
            source_lang=source_lang,
            target_lang=target_lang,
            pivot_lang=pivot_lang
        )

        completion = entry[lang_map[target_lang]]

        return {
            'id': entry['id'],
            'source_lang': source_lang,
            'target_lang': target_lang,
            'pivot_lang': pivot_lang,
            'prompt': prompt,
            'completion': completion,
            'full_text': prompt + completion
        }

    def build_all_training_pairs(self, entries: List[Dict]) -> Dict[str, List[Dict]]:
        """为所有翻译方向构建训练数据"""

        directions = [
            ('sk', 'cn', 'tb'),  # 梵→汉，藏辅助
            ('sk', 'tb', 'cn'),  # 梵→藏，汉辅助
            ('tb', 'cn', 'sk'),  # 藏→汉，梵辅助
            ('tb', 'sk', 'cn'),  # 藏→梵，汉辅助
            ('cn', 'sk', 'tb'),  # 汉→梵，藏辅助
            ('cn', 'tb', 'sk'),  # 汉→藏，梵辅助
        ]

        all_samples = {}

        for source, target, pivot in directions:
            direction_key = f"{source}_to_{target}"
            samples_direct = []
            samples_mutual = []

            for entry in entries:
                sample_direct = self.build_training_sample(
                    entry, source, target, pivot_lang=None
                )
                samples_direct.append(sample_direct)

                sample_mutual = self.build_training_sample(
                    entry, source, target, pivot_lang=pivot
                )
                samples_mutual.append(sample_mutual)

            all_samples[f"{direction_key}_direct"] = samples_direct
            all_samples[f"{direction_key}_mutual"] = samples_mutual

            print(f"  ✓ {direction_key}: {len(samples_direct)} direct + {len(samples_mutual)} mutual")

        return all_samples

class PromptValidator:
    """Prompt模板验证器"""

    def __init__(self, template_manager: PromptTemplateManager):
        self.manager = template_manager
        self.validation_results = []

    def validate_format(self, prompt: str) -> Dict:
        """验证prompt格式"""
        import re
        issues = []

        # 检查必要字段是否被正确填充
        unfilled = re.findall(r'\{(\w+)\}', prompt)
        if unfilled:
            issues.append(f"未填充的占位符: {unfilled}")

        # 检查长度
        if len(prompt) > 4096:
            issues.append(f"Prompt过长: {len(prompt)} 字符（建议 < 4096）")

        if len(prompt) < 50:
            issues.append(f"Prompt过短: {len(prompt)} 字符")

        # 检查编码问题
        try:
            prompt.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(f"编码问题: {e}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'char_count': len(prompt),
            'token_estimate': len(prompt) // 2
        }

    def validate_all_templates(self, sample_entries: List[Dict]) -> Dict:
        """验证所有模板"""
        print("=" * 60)
        print("🔍 Prompt模板验证")
        print("=" * 60)

        directions = [
            ('sk', 'cn', 'tb'),
            ('sk', 'tb', 'cn'),
            ('tb', 'cn', 'sk'),
            ('tb', 'sk', 'cn'),
            ('cn', 'sk', 'tb'),
            ('cn', 'tb', 'sk'),
        ]

        results = {'passed': 0, 'failed': 0, 'details': []}

        for source, target, pivot in directions:
            for entry in sample_entries[:3]:
                # 测试直接翻译
                prompt_direct = self.manager.build_prompt(
                    template_key=None,
                    entry=entry,
                    source_lang=source,
                    target_lang=target,
                    pivot_lang=None
                )
                result_direct = self.validate_format(prompt_direct)

                # 测试互证翻译
                prompt_mutual = self.manager.build_prompt(
                    template_key=None,
                    entry=entry,
                    source_lang=source,
                    target_lang=target,
                    pivot_lang=pivot
                )
                result_mutual = self.validate_format(prompt_mutual)

                for label, result in [('direct', result_direct), ('mutual', result_mutual)]:
                    if result['is_valid']:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'direction': f"{source}->{target} ({label})",
                            'entry_id': entry['id'],
                            'issues': result['issues']
                        })

        # 打印结果
        total = results['passed'] + results['failed']
        print(f"\n通过: {results['passed']}/{total}")
        print(f"失败: {results['failed']}/{total}")

        if results['details']:
            print("\n❌ 失败详情:")
            for detail in results['details']:
                print(f"  [{detail['direction']}] {detail['entry_id']}: {detail['issues']}")
        else:
            print("\n✅ 所有模板验证通过")

        return results

    def show_prompt_examples(self, entries: List[Dict]):
        """展示各方向的prompt示例"""
        print("\n" + "=" * 80)
        print("📋 Prompt示例展示")
        print("=" * 80)

        entry = entries[0]
        target_lang_map = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}

        examples = [
            ('sk', 'cn', None, "梵->汉 直接翻译"),
            ('sk', 'cn', 'tb', "梵->汉 互证翻译（藏文辅助）"),
            ('tb', 'cn', 'sk', "藏->汉 互证翻译（梵文辅助）"),
        ]

        for source, target, pivot, label in examples:
            prompt = self.manager.build_prompt(
                template_key=None,
                entry=entry,
                source_lang=source,
                target_lang=target,
                pivot_lang=pivot
            )

            print(f"\n{'─'*80}")
            print(f"📌 {label} [条目 {entry['id']}]")
            print(f"{'─'*80}")
            print(prompt)
            print(f"\n[期望输出] {entry[target_lang_map[target]]}")
            print(f"[字符数] {len(prompt)}")

class TrainingDataFormatter:
    """训练数据格式化器（适配不同模型的输入格式）"""

    def __init__(self, template_manager: PromptTemplateManager):
        self.manager = template_manager

    def format_for_gemma(self, sample: Dict) -> Dict:
        """格式化为Gemma模型的训练格式"""
        return {
            'text': (
                f"<start_of_turn>user\n{sample['prompt']}<end_of_turn>\n"
                f"<start_of_turn>model\n{sample['completion']}<end_of_turn>"
            )
        }

    def format_for_qwen(self, sample: Dict) -> Dict:
        """格式化为Qwen模型的训练格式"""
        return {
            'messages': [
                {
                    'role': 'system',
                    'content': '你是梵藏汉古典文献翻译专家，精通梵文、藏文和汉文之间的互译。'
                },
                {
                    'role': 'user',
                    'content': sample['prompt']
                },
                {
                    'role': 'assistant',
                    'content': sample['completion']
                }
            ]
        }

    def format_for_alpaca(self, sample: Dict) -> Dict:
        """格式化为Alpaca通用格式"""
        parts = sample['prompt'].split('原文：\n')
        if len(parts) == 2:
            instruction = parts[0].strip()
            input_text = parts[1].split('\n\n')[0].strip()
        else:
            instruction = sample['prompt']
            input_text = ""

        return {
            'instruction': instruction,
            'input': input_text,
            'output': sample['completion']
        }

    def export_training_data(self, samples: List[Dict], model_type: str,
                             output_path: str):
        """导出指定模型格式的训练数据"""

        format_func = {
            'gemma': self.format_for_gemma,
            'qwen': self.format_for_qwen,
            'alpaca': self.format_for_alpaca
        }

        if model_type not in format_func:
            raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(format_func.keys())}")

        formatted = [format_func[model_type](s) for s in samples]

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in formatted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"💾 已导出 {len(formatted)} 条 {model_type} 格式训练数据到 {output_path}")

        # 打印样本预览
        print(f"\n📋 样本预览 ({model_type}):")
        preview = json.dumps(formatted[0], ensure_ascii=False, indent=2)
        if len(preview) > 500:
            preview = preview[:500] + "..."
        print(preview)

        return formatted

# ============================================================
# 独立运行演示（使用内置样本数据）
# ============================================================

def get_sample_data() -> List[Dict]:
    """内置样本数据，用于独立运行验证"""
    return [
        {
            'id': '7.92.14',
            'chapter': 7,
            'section': 92,
            'verse': 14,
            'sanskrit': 'भरतॊ ऽपि तथैवॊष्य संवत्सरम अथाधिकम अयॊध्यां पुनर अगम्य रामपादाव उपागमत',
            'tibetan': 'དེ་བཞིན་དཔའ་བོ་བྷ་ར་ཏ།  ། གྲོང་དེར་ལོ་ངོ་གཅིག་ལྷག་བཞུགས།  ། དེ་ནས་འཐབ་བྲལ་གྲོང་ཁྱེར་དུ།  ། ལོག་ནས་རཱ་མར་གུས་ཕྱག་ཕུལ།  །',
            'chinese': '婆罗多就是这样，在那里住一年多；向罗摩双脚致敬，他也回到阿逾陀。'
        },
        {
            'id': '7.92.15',
            'chapter': 7,
            'section': 92,
            'verse': 15,
            'sanskrit': 'उभौ सौमित्रिभरतौ रामपादाव अनुव्रतौ कालं गतम अपि सनेहान न जज्ञाते ऽतिधार्मिकौ',
            'tibetan': 'ཤིན་ཏུ་དད་པ་བརྟན་པོ་ཡི།  ། ལ་ཀྵ་མ་ན་བྷ་ར་ཏ།  ། རཱ་མའི་ཞབས་ལ་བློ་དཀར་ནས།  ། འཆི་བདག་བསླེབས་པའང་ཚོར་མ་གྱུར།  །',
            'chinese': '罗什曼那、婆罗多，忠诚于罗摩双脚；两个非常虔诚人，不知死神已来到。'
        },
        {
            'id': '7.92.16',
            'chapter': 7,
            'section': 92,
            'verse': 16,
            'sanskrit': 'एवं वर्षसहस्राणि दशतेषां ययुस तदा धर्मे परयतमानानां पौरकार्येषु नित्यदा',
            'tibetan': 'དེ་རྣམས་འབད་པས་ཆོས་བཞིན་སྤྱད།  ། འབངས་ཀྱི་ལས་ཀུན་ལེགས་པར་བསྒྲུབས།  ། དེ་ལྟར་བགྲང་བྱ་ཁྲི་ཕྲག་གཅིག  ། སྐད་ཅིག་གིས་ནི་འདས་པར་གྱུར།  །',
            'chinese': '他们就都是这样，转瞬过了一万年；精勤努力守达磨，为城市人把事办。'
        },
        {
            'id': '7.92.17',
            'chapter': 7,
            'section': 92,
            'verse': 17,
            'sanskrit': 'विहृत्य लाकं परिपूर्णमानसाः शरिया वृता धर्मपथे परे सथिताः तरयः समिद्धा इव दीप्ततेजसा हुताग्नयः साधु महाध्वरे तरयः',
            'tibetan': 'གཟི་བྱིན་ལྡན་པའི་སྤུན་གསུམ་དགའ་བདེའི་ངང་།  ། རྨད་བྱུང་ཆོས་ལ་སྤྱོད་ཅིང་མངའ་ཐང་རྒྱས།  ། སྦྱིན་སྲེག་ཆེན་པོ་གཅིག་ཏུ་འབར་བ་ཡི།  ། མེ་ལྕེའི་ཕུང་བོ་བཞིན་དུ་བཀྲག་མདངས་འབར།  །',
            'chinese': '他们愉快过日子，他们心情都舒畅；他们高贵有荣华，站在最高达磨上。三人威严有光辉，好像烈焰在焚燃；他们都像是祭火，同在一个大祭典。'
        },
        {
            'id': '7.93.1',
            'chapter': 7,
            'section': 93,
            'verse': 1,
            'sanskrit': 'कस्य चित तव अथ कालस्य रामे धर्मपथे सथिते कालस तापसरूपेण राजद्वारम उपागमत',
            'tibetan': 'རཱ་མས་ཆོས་བཞིན་ལེགས་སྤྱད་དེ།  ། སླར་ཡང་རེ་ཞིག་འདས་པ་ན།  ། འཆི་བདག་དཀའ་ཐུབ་སྤྱོད་པ་ལ།  ། སྤྲུལ་ནས་ཕོ་བྲང་སྒོ་ཁར་བསླེབས།  །',
            'chinese': '罗摩在达磨路上，又过了一些时候；死神变成苦行者，向他宫门里面走。'
        },
    ]

def get_core_terminology() -> Dict:
    """内置核心术语表"""
    return {
        'भरत': {'tibetan': 'བྷ་ར་ཏ', 'chinese': '婆罗多', 'category': 'person_name'},
        'राम': {'tibetan': 'རཱ་མ', 'chinese': '罗摩', 'category': 'person_name'},
        'सौमित्रि': {'tibetan': 'ལ་ཀྵ་མ་ན', 'chinese': '罗什曼那', 'category': 'person_name'},
        'अयॊध्या': {'tibetan': 'འཐབ་བྲལ་གྲོང་ཁྱེར', 'chinese': '阿逾陀', 'category': 'place_name'},
        'धर्म': {'tibetan': 'ཆོས', 'chinese': '达磨', 'category': 'term'},
        'काल': {'tibetan': 'འཆི་བདག', 'chinese': '死神', 'category': 'term'},
        'तापस': {'tibetan': 'དཀའ་ཐུབ་སྤྱོད་པ', 'chinese': '苦行者', 'category': 'term'},
        'अग्नि': {'tibetan': 'མེ', 'chinese': '祭火', 'category': 'term'},
        'महाध्वर': {'tibetan': 'སྦྱིན་སྲེག་ཆེན་པོ', 'chinese': '大祭典', 'category': 'term'},
    }

def run_demo():
    """独立运行演示"""

    print("=" * 80)
    print("🚀 Prompt模板系统 - 独立验证演示")
    print("=" * 80)

    # 加载内置数据
    entries = get_sample_data()
    terminology = get_core_terminology()

    print(f"\n📊 样本数据: {len(entries)} 条")
    print(f"📚 术语表: {len(terminology)} 个术语")

    # 初始化模板管理器
    manager = PromptTemplateManager(terminology_dict=terminology)

    # ---- 验证模板 ----
    print("\n" + "-" * 80)
    validator = PromptValidator(manager)
    results = validator.validate_all_templates(entries)

    # ---- 展示示例 ----
    validator.show_prompt_examples(entries)

    # ---- 构建训练数据 ----
    print("\n" + "-" * 80)
    print("📦 构建训练数据...")
    all_samples = manager.build_all_training_pairs(entries)

    # ---- 导出各模型格式 ----
    print("\n" + "-" * 80)
    print("📦 导出训练数据...")
    formatter = TrainingDataFormatter(manager)

    # Gemma格式
    gemma_samples = (
        all_samples['sk_to_cn_direct'] +
        all_samples['sk_to_cn_mutual'] +
        all_samples['sk_to_tb_direct'] +
        all_samples['sk_to_tb_mutual']
    )
    formatter.export_training_data(gemma_samples, 'gemma', 'data/train_gemma.jsonl')

    # Qwen格式
    qwen_samples = (
        all_samples['sk_to_cn_direct'] +
        all_samples['sk_to_cn_mutual'] +
        all_samples['tb_to_cn_direct'] +
        all_samples['tb_to_cn_mutual']
    )
    formatter.export_training_data(qwen_samples, 'qwen', 'data/train_qwen.jsonl')

    # Alpaca格式
    alpaca_samples = all_samples['sk_to_cn_direct'] + all_samples['tb_to_cn_direct']
    formatter.export_training_data(alpaca_samples, 'alpaca', 'data/train_alpaca.jsonl')

    # ---- 保存术语表 ----
    os.makedirs('data', exist_ok=True)
    with open('data/terminology_final.json', 'w', encoding='utf-8') as f:
        json.dump(terminology, f, ensure_ascii=False, indent=2)
    print(f"\n💾 术语表已保存: data/terminology_final.json ({len(terminology)} 个术语)")

    # ---- 保存样本数据为jsonl ----
    with open('data/ramayana_trilingual.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"💾 三语平行语料已保存: data/ramayana_trilingual.jsonl ({len(entries)} 条)")

    # ---- 统计汇总 ----
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)

    total_samples = sum(len(v) for v in all_samples.values())
    print(f"""
汇总:
├── 样本条目: {len(entries)} 条三语平行数据
├── 术语表: {len(terminology)} 个核心术语
├── 翻译方向: 6 个方向 x 2 模式(直接+互证) = 12 组
├── 训练样本总数: {total_samples} 条
├── 模板验证: {results['passed']} 通过 / {results['failed']} 失败
│
├── 输出文件:
│   ├── data/ramayana_trilingual.jsonl
│   ├── data/terminology_final.json
│   ├── data/train_gemma.jsonl
│   ├── data/train_qwen.jsonl
│   └── data/train_alpaca.jsonl
    """)

    return {
        'total_entries': len(entries),
        'terminology_count': len(terminology),
        'total_training_samples': total_samples,
        'validation': results
    }

if __name__ == "__main__":
    results = run_demo()