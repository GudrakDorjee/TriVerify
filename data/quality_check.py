# data_preparation/quality_check.py
import json
from typing import List, Dict
from collections import Counter

class DataQualityChecker:
    """数据质量检查"""
    
    def __init__(self, entries: List[Dict]):
        self.entries = entries
        self.issues = []
    
    def check_alignment(self):
        """检查三语对齐质量"""
        print("🔍 检查三语对齐...")
        
        misaligned = []
        for entry in self.entries:
            sk_len = len(entry['sanskrit'])
            tb_len = len(entry['tibetan'])
            cn_len = len(entry['chinese'])
            
            # 长度比例异常检测（某语言文本异常短或长）
            avg_len = (sk_len + tb_len + cn_len) / 3
            for lang, length in [('sanskrit', sk_len), ('tibetan', tb_len), ('chinese', cn_len)]:
                ratio = length / avg_len if avg_len > 0 else 0
                if ratio < 0.2 or ratio > 3.0:
                    misaligned.append({
                        'id': entry['id'],
                        'lang': lang,
                        'ratio': ratio,
                        'text': entry[lang][:50]
                    })
        
        if misaligned:
            print(f"  ⚠️  发现 {len(misaligned)} 条可能对齐异常的条目")
            for item in misaligned[:5]:
                print(f"    [{item['id']}] {item['lang']}: ratio={item['ratio']:.2f}")
        else:
            print("  ✓ 对齐质量良好")
        
        return misaligned
    
    def check_duplicates(self):
        """检查重复条目"""
        print("🔍 检查重复条目...")
        
        id_counts = Counter(e['id'] for e in self.entries)
        duplicates = {k: v for k, v in id_counts.items() if v > 1}
        
        if duplicates:
            print(f"  ⚠️  发现 {len(duplicates)} 个重复ID")
            for id_, count in list(duplicates.items())[:5]:
                print(f"    {id_}: 出现 {count} 次")
        else:
            print("  ✓ 无重复条目")
        
        return duplicates
    
    def check_encoding(self):
        """检查编码问题"""
        print("🔍 检查编码问题...")
        
        encoding_issues = []
        for entry in self.entries:
            for lang in ['sanskrit', 'tibetan', 'chinese']:
                text = entry[lang]
                # 检查是否包含替换字符或乱码
                if '\ufffd' in text or '?' * 3 in text:
                    encoding_issues.append({
                        'id': entry['id'],
                        'lang': lang,
                        'sample': text[:30]
                    })
        
        if encoding_issues:
            print(f"  ⚠️  发现 {len(encoding_issues)} 条编码问题")
        else:
            print("  ✓ 编码正常")
        
        return encoding_issues
    
    def check_completeness(self):
        """检查完整性"""
        print("🔍 检查数据完整性...")
        
        incomplete = []
        for entry in self.entries:
            missing = []
            for lang in ['sanskrit', 'tibetan', 'chinese']:
                if not entry.get(lang) or len(entry[lang].strip()) == 0:
                    missing.append(lang)
            if missing:
                incomplete.append({'id': entry['id'], 'missing': missing})
        
        if incomplete:
            print(f"  ⚠️  发现 {len(incomplete)} 条不完整条目")
        else:
            print("  ✓ 所有条目完整")
        
        return incomplete
    
    def check_terminology_coverage(self, terminology: Dict):
        """检查术语在语料中的覆盖率"""
        print("🔍 检查术语覆盖率...")
        
        term_coverage = {}
        for term, translations in terminology.items():
            count = 0
            for entry in self.entries:
                if (term in entry['sanskrit'] or 
                    translations.get('tibetan', '') in entry['tibetan'] or
                    translations.get('chinese', '') in entry['chinese']):
                    count += 1
            term_coverage[term] = {
                'count': count,
                'ratio': count / len(self.entries)
            }
        
        print(f"  术语覆盖统计:")
        for term, stats in sorted(term_coverage.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"    {term}: {stats['count']} 条 ({stats['ratio']*100:.1f}%)")
        
        return term_coverage
    
    def run_all_checks(self, terminology: Dict = None) -> Dict:
        """运行所有质量检查"""
        print("=" * 60)
        print("🏥 数据质量全面检查")
        print("=" * 60)
        print(f"总条目数: {len(self.entries)}\n")
        
        results = {
            'total_entries': len(self.entries),
            'alignment_issues': self.check_alignment(),
            'duplicates': self.check_duplicates(),
            'encoding_issues': self.check_encoding(),
            'incomplete': self.check_completeness(),
        }
        
        if terminology:
            results['term_coverage'] = self.check_terminology_coverage(terminology)
        
        # 总结
        total_issues = (
            len(results['alignment_issues']) + 
            len(results['duplicates']) + 
            len(results['encoding_issues']) + 
            len(results['incomplete'])
        )
        
        print(f"\n{'='*60}")
        print(f"📋 检查总结: 发现 {total_issues} 个问题")
        if total_issues == 0:
            print("✅ 数据质量优秀，可以进入下一步")
        elif total_issues < 10:
            print("⚠️  少量问题，建议修复后继续")
        else:
            print("❌ 问题较多，建议仔细清洗后再继续")
        
        return results

# 使用示例
if __name__ == "__main__":
    with open("/root/autodl-tmp/LlamaFactory-main/data/ramayana_trilingual.jsonl", 'r', encoding='utf-8-sig') as f:
        entries = [json.loads(line) for line in f]
    
    with open("/root/autodl-tmp/LlamaFactory-main/data_preparation/data/terminology_final.json", 'r', encoding='utf-8-sig') as f:
        terminology = json.load(f)
    
    checker = DataQualityChecker(entries)
    results = checker.run_all_checks(terminology=terminology)