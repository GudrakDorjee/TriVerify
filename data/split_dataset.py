# data_preparation/split_dataset.py
import json
import random
from collections import Counter, defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetAnalyzer:
    """数据集统计分析"""
    
    def __init__(self, jsonl_path: str):
        self.entries = self.load_jsonl(jsonl_path)
    
    def load_jsonl(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def basic_statistics(self):
        """基础统计"""
        print("=" * 60)
        print("📊 数据集基础统计")
        print("=" * 60)
        
        total = len(self.entries)
        print(f"总条目数: {total}")
        
        # 章节分布
        chapters = Counter(e['chapter'] for e in self.entries)
        print(f"\n章节数: {len(chapters)}")
        print(f"章节分布: {dict(sorted(chapters.items())[:5])}...")
        
        # 文本长度统计
        sk_lengths = [len(e['sanskrit']) for e in self.entries]
        tb_lengths = [len(e['tibetan']) for e in self.entries]
        cn_lengths = [len(e['chinese']) for e in self.entries]
        
        print(f"\n平均长度:")
        print(f"  梵文: {sum(sk_lengths)/total:.1f} 字符")
        print(f"  藏文: {sum(tb_lengths)/total:.1f} 字符")
        print(f"  汉文: {sum(cn_lengths)/total:.1f} 字符")
        
        return {
            'total': total,
            'chapters': len(chapters),
            'avg_sk_len': sum(sk_lengths)/total,
            'avg_tb_len': sum(tb_lengths)/total,
            'avg_cn_len': sum(cn_lengths)/total
        }
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                     seed=42, stratify_by_chapter=True):
        """划分数据集"""
        random.seed(seed)
        
        if stratify_by_chapter:
            # 按章节分层划分，保证各章节在三个集合中都有代表
            splits = {'train': [], 'val': [], 'test': []}
            
            by_chapter = defaultdict(list)
            for entry in self.entries:
                by_chapter[entry['chapter']].append(entry)
            
            for chapter, chapter_entries in by_chapter.items():
                random.shuffle(chapter_entries)
                n = len(chapter_entries)
                
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)
                
                splits['train'].extend(chapter_entries[:train_end])
                splits['val'].extend(chapter_entries[train_end:val_end])
                splits['test'].extend(chapter_entries[val_end:])
            
            # 打乱顺序
            for split in splits.values():
                random.shuffle(split)
        else:
            # 简单随机划分
            random.shuffle(self.entries)
            n = len(self.entries)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': self.entries[:train_end],
                'val': self.entries[train_end:val_end],
                'test': self.entries[val_end:]
            }
        
        print("\n" + "=" * 60)
        print("✂️  数据集划分结果")
        print("=" * 60)
        for name, data in splits.items():
            print(f"{name:>6}: {len(data):>5} 条 ({len(data)/len(self.entries)*100:.1f}%)")
        
        return splits
    
    def save_splits(self, splits: Dict[str, List], output_dir: str):
        """保存划分后的数据集"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for split_name, data in splits.items():
            output_path = f"{output_dir}/{split_name}.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"💾 {split_name}: {output_path}")
    
    def visualize_distribution(self, splits: Dict[str, List], save_path: str = None):
        """可视化数据分布"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各split的章节分布
        for split_name, data in splits.items():
            chapters = [e['chapter'] for e in data]
            axes[0, 0].hist(chapters, alpha=0.5, label=split_name, bins=20)
        axes[0, 0].set_xlabel('Chapter')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Chapter Distribution by Split')
        axes[0, 0].legend()
        
        # 2. 文本长度分布
        for lang, idx in [('sanskrit', 0), ('tibetan', 1), ('chinese', 2)]:
            lengths = [len(e[lang]) for e in self.entries]
            axes[0, 1].hist(lengths, alpha=0.5, label=lang.capitalize(), bins=30)
        axes[0, 1].set_xlabel('Text Length (characters)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].legend()
        
        # 3. 各split大小对比
        split_sizes = {name: len(data) for name, data in splits.items()}
        axes[1, 0].bar(split_sizes.keys(), split_sizes.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[1, 0].set_ylabel('Number of Entries')
        axes[1, 0].set_title('Dataset Split Sizes')
        for i, (name, size) in enumerate(split_sizes.items()):
            axes[1, 0].text(i, size + 50, str(size), ha='center', fontweight='bold')
        
        # 4. 三语长度相关性
        sk_lens = [len(e['sanskrit']) for e in self.entries]
        cn_lens = [len(e['chinese']) for e in self.entries]
        axes[1, 1].scatter(sk_lens, cn_lens, alpha=0.3, s=10)
        axes[1, 1].set_xlabel('Sanskrit Length')
        axes[1, 1].set_ylabel('Chinese Length')
        axes[1, 1].set_title('Sanskrit vs Chinese Length Correlation')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化图表已保存: {save_path}")
        plt.show()

# 使用示例
if __name__ == "__main__":
    analyzer = DatasetAnalyzer("/root/autodl-tmp/LlamaFactory-main/data/ramayana_trilingual.jsonl")
    
    # 统计分析
    stats = analyzer.basic_statistics()
    
    # 划分数据集
    splits = analyzer.split_dataset(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_by_chapter=True
    )
    
    # 保存
    analyzer.save_splits(splits, "data/splits")
    
    # 可视化
    analyzer.visualize_distribution(splits, "/root/autodl-tmp/LlamaFactory-main/data/dataset_distribution.png")