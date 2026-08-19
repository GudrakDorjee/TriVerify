# data_preparation/annotation_helper.py
import json
from typing import Dict, List

class AnnotationHelper:
    """人工标注辅助工具"""
    
    def __init__(self, entries: List[Dict], terminology_candidates: List[Dict]):
        self.entries = entries
        self.candidates = terminology_candidates
    
    def show_context(self, term: Dict, max_examples: int = 5):
        """显示术语在语料中的上下文"""
        sk = term['sanskrit']
        tb = term['tibetan']
        cn = term['chinese']
        
        print(f"\n{'='*80}")
        print(f"术语: {sk} | {tb} | {cn}")
        print(f"频次: {term['frequency']}")
        print(f"{'='*80}\n")
        
        count = 0
        for entry in self.entries:
            if sk in entry['sanskrit'] and tb in entry['tibetan'] and cn in entry['chinese']:
                print(f"[{entry['id']}]")
                print(f"SK: {entry['sanskrit']}")
                print(f"TB: {entry['tibetan']}")
                print(f"CN: {entry['chinese']}")
                print("-" * 80)
                
                count += 1
                if count >= max_examples:
                    break
        
        if count == 0:
            print("⚠️  未找到包含该术语的完整上下文")
    
    def interactive_annotation(self, start_idx: int = 0):
        """交互式标注"""
        categories = {
            '1': 'person_name',
            '2': 'place_name',
            '3': 'term',
            '4': 'deity',
            '5': 'other'
        }
        
        annotated = []
        
        for i, term in enumerate(self.candidates[start_idx:], start=start_idx):
            self.show_context(term, max_examples=3)
            
            print("\n请选择类别:")
            for key, value in categories.items():
                print(f"  {key}. {value}")
            print("  s. 跳过")
            print("  q. 退出并保存")
            
            choice = input("\n选择 (1-5/s/q): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 's':
                continue
            elif choice in categories:
                is_valid = input("是否有效? (y/n): ").strip().lower()
                if is_valid == 'y':
                    term['category'] = categories[choice]
                    term['is_valid'] = True
                    annotated.append(term)
                    print(f"✓ 已标注为 {categories[choice]}")
            
            print(f"\n进度: {i+1}/{len(self.candidates)}")
        
        return annotated
    
    def save_annotated(self, annotated: List[Dict], output_path: str):
        """保存标注结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已保存 {len(annotated)} 个标注结果到 {output_path}")

# 使用示例
if __name__ == "__main__":
    # 加载数据
    with open("/root/autodl-tmp/LlamaFactory-main/data_preparation/data/ramayana_trilingual.jsonl", 'r') as f:
        entries = [json.loads(line) for line in f]
    
    with open("/root/autodl-tmp/LlamaFactory-main/data_preparation/data/terminology_verified.json", 'r') as f:
        candidates = json.load(f)
    
    # 交互式标注
    helper = AnnotationHelper(entries, candidates)
    annotated = helper.interactive_annotation(start_idx=0)
    
    # 保存
    helper.save_annotated(annotated, "/root/autodl-tmp/LlamaFactory-main/data_preparation/data/terminology_annotated.json")