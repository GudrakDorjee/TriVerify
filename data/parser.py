# parser.py
import re
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class TrilingualEntry:
    """三语平行条目"""
    id: str
    chapter: int
    section: int
    verse: int
    sanskrit: str
    tibetan: str
    chinese: str
    
    def to_dict(self):
        return asdict(self)
    
    def is_valid(self) -> bool:
        return all([
            self.sanskrit.strip(),
            self.tibetan.strip(),
            self.chinese.strip()
        ])

class RamayanaCorpusParser:
    """罗摩衍那语料解析器"""
    
    def parse_file(self, filepath: str) -> List[TrilingualEntry]:
        """解析语料文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        entries = []
        
        # 按 "SK" 开头切分每条记录
        segments = re.split(r'\n\s*\n', content)  # 按空行分割
        
        # 如果没有空行分隔，就按 SK 开头切分
        if len(segments) <= 1:
            segments = re.split(r'(?=SK\d+\.\d+\.\d+：)', content)
        
        for seg in segments:
            seg = seg.strip()
            if not seg or not seg.startswith('SK'):
                continue
            
            # 提取三部分：用 ；TB 和 ；CN 作为分隔点
            sk_match = re.match(
                r'SK(\d+\.\d+\.\d+)：(.+?)；TB\d+\.\d+\.\d+：(.+?)；CN\d+\.\d+\.\d+：(.+)',
                seg,
                re.DOTALL
            )
            
            if sk_match:
                id_str = sk_match.group(1)
                sk_text = sk_match.group(2).strip()
                tb_text = sk_match.group(3).strip()
                cn_text = sk_match.group(4).strip()
                
                parts_id = id_str.split('.')
                entry = TrilingualEntry(
                    id=id_str,
                    chapter=int(parts_id[0]),
                    section=int(parts_id[1]),
                    verse=int(parts_id[2]),
                    sanskrit=sk_text,
                    tibetan=tb_text,
                    chinese=cn_text
                )
                
                if entry.is_valid():
                    entries.append(entry)
                else:
                    print(f"⚠️  无效条目: {id_str}")
            else:
                # 调试：打印未匹配的前80字符
                preview = seg[:80].replace('\n', ' ')
                print(f"⚠️  未匹配: {preview}...")
        
        print(f"✓ 解析到 {len(entries)} 条有效条目")
        return entries
    
    def save_to_jsonl(self, entries: List[TrilingualEntry], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        print(f"💾 已保存 {len(entries)} 条到 {output_path}")
    
    def save_to_csv(self, entries: List[TrilingualEntry], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([e.to_dict() for e in entries])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 已保存CSV到 {output_path}")

if __name__ == "__main__":
    parser = RamayanaCorpusParser()
    
    entries = parser.parse_file("/root/autodl-tmp/LlamaFactory-main/SK-TB-CN.txt")
    
    # 展示前3条验证
    for e in entries[:3]:
        print(f"\n[{e.id}]")
        print(f"  SK: {e.sanskrit[:60]}...")
        print(f"  TB: {e.tibetan[:60]}...")
        print(f"  CN: {e.chinese[:60]}...")
    
    # 保存
    parser.save_to_jsonl(entries, "/root/autodl-tmp/LlamaFactory-main/data/ramayana_trilingual.jsonl")
    parser.save_to_csv(entries, "/root/autodl-tmp/LlamaFactory-main/data/ramayana_trilingual.csv")