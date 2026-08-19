# terminology_extraction.py
import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class TerminologyExtractor:
    """专名术语自动抽取器 - 适配天城文梵文 + 藏文 + 汉文"""

    def __init__(self):
        # 已知专名种子表（从你的语料中人工识别的核心专名）
        # 这是最可靠的方式：先用种子表匹配，再扩展
        self.seed_terms = [
            {
                'sanskrit': ['भरत', 'भरतॊ', 'भरतौ'],
                'tibetan': ['བྷ་ར་ཏ'],
                'chinese': ['婆罗多'],
                'category': 'person_name'
            },
            {
                'sanskrit': ['राम', 'रामे', 'रामपादाव', 'रामपादौ'],
                'tibetan': ['རཱ་མ', 'རཱ་མའི'],
                'chinese': ['罗摩'],
                'category': 'person_name'
            },
            {
                'sanskrit': ['सौमित्रि', 'सौमित्रिभरतौ'],
                'tibetan': ['ལ་ཀྵ་མ་ན'],
                'chinese': ['罗什曼那'],
                'category': 'person_name'
            },
            {
                'sanskrit': ['अयॊध्या', 'अयॊध्यां'],
                'tibetan': ['འཐབ་བྲལ་གྲོང་ཁྱེར'],
                'chinese': ['阿逾陀'],
                'category': 'place_name'
            },
            {
                'sanskrit': ['धर्म', 'धर्मे', 'धर्मपथे'],
                'tibetan': ['ཆོས'],
                'chinese': ['达磨', '法'],
                'category': 'term'
            },
            {
                'sanskrit': ['काल', 'कालस', 'कालं'],
                'tibetan': ['འཆི་བདག'],
                'chinese': ['死神'],
                'category': 'term'
            },
            {
                'sanskrit': ['तापस', 'तापसरूपेण'],
                'tibetan': ['དཀའ་ཐུབ'],
                'chinese': ['苦行者', '苦行'],
                'category': 'term'
            },
            {
                'sanskrit': ['अग्नि', 'हुताग्नयः'],
                'tibetan': ['མེ', 'སྦྱིན་སྲེག'],
                'chinese': ['祭火', '火'],
                'category': 'term'
            },
            {
                'sanskrit': ['राजद्वार', 'राजद्वारम'],
                'tibetan': ['ཕོ་བྲང་སྒོ'],
                'chinese': ['宫门'],
                'category': 'place_name'
            },
        ]

    def match_seed_terms(self, entries: List[Dict]) -> List[Dict]:
        """用种子表在语料中匹配并统计频次"""
        results = []

        for seed in self.seed_terms:
            freq = 0
            for entry in entries:
                sk_match = any(t in entry['sanskrit'] for t in seed['sanskrit'])
                tb_match = any(t in entry['tibetan'] for t in seed['tibetan'])
                cn_match = any(t in entry['chinese'] for t in seed['chinese'])

                # 至少两种语言匹配到
                if sum([sk_match, tb_match, cn_match]) >= 2:
                    freq += 1

            results.append({
                'sanskrit': seed['sanskrit'][0],  # 取词根形式
                'sanskrit_variants': seed['sanskrit'],
                'tibetan': seed['tibetan'][0],
                'tibetan_variants': seed['tibetan'],
                'chinese': seed['chinese'][0],
                'chinese_variants': seed['chinese'],
                'category': seed['category'],
                'frequency': freq
            })

        results.sort(key=lambda x: x['frequency'], reverse=True)
        return results

    def discover_chinese_names(self, entries: List[Dict]) -> List[str]:
        """从汉文中发现潜在音译专名"""
        # 音译常用字
        transliteration_chars = set(
            '罗摩婆陀那曼迦耶阿逾什达磨梨舍利弗提毗湿奴'
            '伐楼拿因陁帝释梵天悉多波旬乾闼'
        )

        candidates = Counter()
        for entry in entries:
            text = entry['chinese']
            i = 0
            while i < len(text):
                if text[i] in transliteration_chars:
                    name = text[i]
                    j = i + 1
                    while j < len(text) and text[j] in transliteration_chars:
                        name += text[j]
                        j += 1
                    if len(name) >= 2:
                        candidates[name] += 1
                    i = j
                else:
                    i += 1

        # 过滤低频
        return [(name, freq) for name, freq in candidates.most_common(50) if freq >= 2]

    def discover_tibetan_names(self, entries: List[Dict]) -> List[str]:
        """从藏文中发现潜在音译专名（含梵文借词标记字符）"""
        # 藏文中梵文音译常含这些字符组合
        sanskrit_markers = ['བྷ', 'དྷ', 'གྷ', 'ཛྷ', 'ཀྵ', 'རཱ', 'ཏྲ', 'ཤྲ']

        candidates = Counter()
        for entry in entries:
            text = entry['tibetan']
            for marker in sanskrit_markers:
                if marker in text:
                    # 提取包含该标记的音节段
                    # 藏文用 ་ (tsheg) 分隔音节
                    syllables = text.split('་')
                    for idx, syl in enumerate(syllables):
                        if marker in syl:
                            # 取该音节及前后各1个音节作为候选名
                            start = max(0, idx - 1)
                            end = min(len(syllables), idx + 3)
                            name = '་'.join(syllables[start:end])
                            if name:
                                candidates[name] += 1

        return [(name, freq) for name, freq in candidates.most_common(30) if freq >= 2]

    def extract_terminology(self, entries: List[Dict]) -> Dict:
        """完整术语抽取流程"""
        print("🔍 开始提取专名术语...")
        print(f"   语料条目数: {len(entries)}")

        # 1. 种子表匹配
        print("\n📌 Step 1: 种子表匹配")
        seed_results = self.match_seed_terms(entries)
        print(f"   匹配到 {len([r for r in seed_results if r['frequency'] > 0])} 个种子术语")

        # 2. 发现新的汉文专名
        print("\n📌 Step 2: 发现汉文音译专名")
        cn_candidates = self.discover_chinese_names(entries)
        print(f"   发现 {len(cn_candidates)} 个候选汉文专名")

        # 3. 发现新的藏文专名
        print("\n📌 Step 3: 发现藏文音译专名")
        tb_candidates = self.discover_tibetan_names(entries)
        print(f"   发现 {len(tb_candidates)} 个候选藏文专名")

        return {
            'verified_terms': seed_results,
            'cn_candidates': cn_candidates,
            'tb_candidates': tb_candidates
        }

    def save_terminology(self, results: Dict, output_dir: str):
        """保存术语表"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存已验证术语表（JSON格式，供模型使用）
        verified_path = os.path.join(output_dir, 'terminology_verified.json')
        with open(verified_path, 'w', encoding='utf-8') as f:
            json.dump(results['verified_terms'], f, ensure_ascii=False, indent=2)
        print(f"\n💾 已验证术语表: {verified_path}")

        # 2. 保存待标注文件
        annotate_path = os.path.join(output_dir, 'terminology_to_annotate.txt')
        with open(annotate_path, 'w', encoding='utf-8') as f:
            f.write("# 专名术语标注文件\n")
            f.write("# ============================================\n\n")

            f.write("## 已验证术语（种子表匹配）\n\n")
            f.write(f"{'梵文':<20} {'藏文':<25} {'汉文':<10} {'类别':<15} {'频次':>5}\n")
            f.write("-" * 80 + "\n")
            for term in results['verified_terms']:
                f.write(f"{term['sanskrit']:<20} {term['tibetan']:<25} "
                        f"{term['chinese']:<10} {term['category']:<15} {term['frequency']:>5}\n")

            f.write("\n\n## 待确认：汉文音译候选\n\n")
            for name, freq in results['cn_candidates']:
                f.write(f"  {name} (出现 {freq} 次)\n")

            f.write("\n\n## 待确认：藏文音译候选\n\n")
            for name, freq in results['tb_candidates']:
                f.write(f"  {name} (出现 {freq} 次)\n")

        print(f"📝 待标注文件: {annotate_path}")

        # 3. 保存为模型可用的简洁格式
        lookup_path = os.path.join(output_dir, 'term_lookup.json')
        lookup = {}
        for term in results['verified_terms']:
            if term['frequency'] > 0:
                lookup[term['sanskrit']] = {
                    'tb': term['tibetan'],
                    'cn': term['chinese'],
                    'variants_sk': term['sanskrit_variants'],
                    'variants_tb': term['tibetan_variants'],
                    'variants_cn': term['chinese_variants'],
                    'category': term['category']
                }
        with open(lookup_path, 'w', encoding='utf-8') as f:
            json.dump(lookup, f, ensure_ascii=False, indent=2)
        print(f"🔑 模型查询表: {lookup_path}")

    def print_summary(self, results: Dict):
        """打印摘要"""
        print("\n" + "=" * 80)
        print("📋 术语抽取结果摘要")
        print("=" * 80)

        print(f"\n{'梵文':<20} {'藏文':<25} {'汉文':<10} {'类别':<15} {'频次':>5}")
        print("-" * 80)
        for term in results['verified_terms']:
            if term['frequency'] > 0:
                print(f"{term['sanskrit']:<20} {term['tibetan']:<25} "
                      f"{term['chinese']:<10} {term['category']:<15} {term['frequency']:>5}")

        print(f"\n汉文音译候选 (前10):")
        for name, freq in results['cn_candidates'][:10]:
            print(f"  {name}: {freq}次")

        print(f"\n藏文音译候选 (前10):")
        for name, freq in results['tb_candidates'][:10]:
            print(f"  {name}: {freq}次")

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import os

    # 加载数据
    data_path = "/root/autodl-tmp/LlamaFactory-main/data_preparation/data/ramayana_trilingual.jsonl"

    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}")
        print("   请先运行 parser.py 生成数据文件")
        print("   或者直接使用示例数据...")

        # 使用你提供的示例数据
        entries = [
            {
                "id": "7.92.14", "chapter": 7, "section": 92, "verse": 14,
                "sanskrit": "भरतॊ ऽपि तथैवॊष्य संवत्सरम अथाधिकम अयॊध्यां पुनर अगम्य रामपादाव उपागमत",
                "tibetan": "དེ་བཞིན་དཔའ་བོ་བྷ་ར་ཏ།  ། གྲོང་དེར་ལོ་ངོ་གཅིག་ལྷག་བཞུགས།  ། དེ་ནས་འཐབ་བྲལ་གྲོང་ཁྱེར་དུ།  ། ལོག་ནས་རཱ་མར་གུས་ཕྱག་ཕུལ།  །",
                "chinese": "婆罗多就是这样，在那里住一年多；向罗摩双脚致敬，他也回到阿逾陀。"
            },
            {
                "id": "7.92.15", "chapter": 7, "section": 92, "verse": 15,
                "sanskrit": "उभौ सौमित्रिभरतौ रामपादाव अनुव्रतौ कालं गतम अपि सनेहान न जज्ञाते ऽतिधार्मिकौ",
                "tibetan": "ཤིན་ཏུ་དད་པ་བརྟན་པོ་ཡི།  ། ལ་ཀྵ་མ་ན་བྷ་ར་ཏ།  ། རཱ་མའི་ཞབས་ལ་བློ་དཀར་ནས།  ། འཆི་བདག་བསླེབས་པའང་ཚོར་མ་གྱུར།  །",
                "chinese": "罗什曼那、婆罗多，忠诚于罗摩双脚； 两个非常虔诚人， 不知死神已来到。"
            },
            {
                "id": "7.92.16", "chapter": 7, "section": 92, "verse": 16,
                "sanskrit": "एवं वर्षसहस्राणि दशतेषां ययुस तदा धर्मे परयतमानानां पौरकार्येषु नित्यदा",
                "tibetan": "དེ་རྣམས་འབད་པས་ཆོས་བཞིན་སྤྱད།  ། འབངས་ཀྱི་ལས་ཀུན་ལེགས་པར་བསྒྲུབས།  ། དེ་ལྟར་བགྲང་བྱ་ཁྲི་ཕྲག་གཅིག  ། སྐད་ཅིག་གིས་ནི་འདས་པར་གྱུར།  །",
                "chinese": "他们就都是这样，转瞬过了一万年；精勤努力守达磨，为城市人把事办。"
            },
            {
                "id": "7.92.17", "chapter": 7, "section": 92, "verse": 17,
                "sanskrit": "विहृत्य लाकं परिपूर्णमानसाः; शरिया वृता धर्मपथे परे सथिताः तरयः समिद्धा इव दीप्ततेजसा; हुताग्नयः साधु महाध्वरे तरयः",
                "tibetan": "གཟི་བྱིན་ལྡན་པའི་སྤུན་གསུམ་དགའ་བདེའི་ངང་།  ། རྨད་བྱུང་ཆོས་ལ་སྤྱོད་ཅིང་མངའ་ཐང་རྒྱས།  ། སྦྱིན་སྲེག་ཆེན་པོ་གཅིག་ཏུ་འབར་བ་ཡི།  ། མེ་ལྕེའི་ཕུང་བོ་བཞིན་དུ་བཀྲག་མདངས་འབར།  །",
                "chinese": "他们愉快过日子，他们心情都舒畅；他们高贵有荣华，站在最高达磨上。三人威严有光辉，好像烈焰在焚燃；他们都像是祭火，同在一个大祭典。"
            },
            {
                "id": "7.93.1", "chapter": 7, "section": 93, "verse": 1,
                "sanskrit": "कस्य चित तव अथ कालस्य रामे धर्मपथे सथिते कालस तापसरूपेण राजद्वारम उपागमत",
                "tibetan": "ཞེས་རཱ་མ་ཎའི་རྟོགས་བརྗོད་ཀྱི་མཇུག་གི་སྡེ་ལས་ལེའུ་གོ་གཉིས་པའོ།  ། ལེའུ་གོ་གསུམ་པ། རཱ་མས་ཆོས་བཞིན་ལེགས་སྤྱད་དེ།  ། སླར་ཡང་རེ་ཞིག་འདས་པ་ན།  ། འཆི་བདག་དཀའ་ཐུབ་སྤྱོད་པ་ལ།  ། སྤྲུལ་ནས་ཕོ་བྲང་སྒོ་ཁར་བསླེབས།  །",
                "chinese": "《罗摩衍那（七）·后篇》第九十二章终 第九十三章 罗摩在达磨路上，又过了一些时候；死神变成苦行者，向他宫门里面走。"
            }
        ]
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]

    # 提取术语
    extractor = TerminologyExtractor()
    results = extractor.extract_terminology(entries)

    # 打印摘要
    extractor.print_summary(results)

    # 保存
    output_dir = "/root/autodl-tmp/LlamaFactory-main/data_preparation/data"
    extractor.save_terminology(results, output_dir)