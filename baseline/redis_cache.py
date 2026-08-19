#!/usr/bin/env python3
"""
Redis 缓存加速模块
用于缓存模型推理结果，避免重复翻译
"""
import json
import hashlib
import time
from typing import Optional, Dict, List
import redis

class TranslationCache:
    """翻译结果缓存"""

    def __init__(self, host='localhost', port=6379, db=0, 
                 prefix='trans', ttl=86400*30):
        """
        Args:
            host: Redis 主机
            port: Redis 端口
            db: Redis 数据库编号
            prefix: 缓存键前缀
            ttl: 缓存过期时间（秒），默认30天
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = prefix
        self.ttl = ttl
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0}

        # 验证连接
        try:
            self.client.ping()
            print(f"✓ Redis 连接成功 ({host}:{port}, db={db})")
            print(f"  已有缓存条目: {self.client.dbsize()}")
        except redis.ConnectionError:
            print(f"⚠️  Redis 连接失败，将使用内存缓存")
            self.client = None
            self._memory_cache = {}

    def _make_key(self, text: str, source_lang: str, target_lang: str,
                  method: str = 'direct', pivot_text: str = None,
                  model: str = 'qwen', temperature: float = 0.3) -> str:
        """生成缓存键"""
        # 用内容哈希作为键，避免键过长
        content = f"{text}|{source_lang}|{target_lang}|{method}|{pivot_text or ''}|{model}|{temperature}"
        hash_val = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{self.prefix}:{source_lang}:{target_lang}:{method}:{hash_val}"

    def get(self, text: str, source_lang: str, target_lang: str,
            method: str = 'direct', pivot_text: str = None,
            model: str = 'qwen', temperature: float = 0.3) -> Optional[str]:
        """查询缓存"""
        key = self._make_key(text, source_lang, target_lang, method, pivot_text, model, temperature)

        if self.client:
            result = self.client.get(key)
        else:
            result = self._memory_cache.get(key)

        if result:
            self.stats['hits'] += 1
            return result
        else:
            self.stats['misses'] += 1
            return None

    def set(self, text: str, source_lang: str, target_lang: str,
            translation: str, method: str = 'direct', pivot_text: str = None,
            model: str = 'qwen', temperature: float = 0.3):
        """写入缓存"""
        key = self._make_key(text, source_lang, target_lang, method, pivot_text, model, temperature)

        if self.client:
            self.client.setex(key, self.ttl, translation)
        else:
            self._memory_cache[key] = translation

        self.stats['sets'] += 1

    def get_or_generate(self, text: str, source_lang: str, target_lang: str,
                        generate_fn, method: str = 'direct', pivot_text: str = None,
                        model: str = 'qwen', temperature: float = 0.3) -> str:
        """查询缓存，未命中则调用生成函数"""
        cached = self.get(text, source_lang, target_lang, method, pivot_text, model, temperature)
        if cached is not None:
            return cached

        # 缓存未命中，调用模型生成
        translation = generate_fn()

        # 写入缓存
        self.set(text, source_lang, target_lang, translation, method, pivot_text, model, temperature)

        return translation

    def batch_get(self, items: List[Dict]) -> List[Optional[str]]:
        """批量查询"""
        if not self.client:
            return [self._memory_cache.get(
                self._make_key(item['text'], item['source_lang'], item['target_lang'],
                              item.get('method', 'direct'), item.get('pivot_text'),
                              item.get('model', 'qwen'), item.get('temperature', 0.3))
            ) for item in items]

        pipe = self.client.pipeline()
        keys = []
        for item in items:
            key = self._make_key(
                item['text'], item['source_lang'], item['target_lang'],
                item.get('method', 'direct'), item.get('pivot_text'),
                item.get('model', 'qwen'), item.get('temperature', 0.3)
            )
            keys.append(key)
            pipe.get(key)

        results = pipe.execute()

        for r in results:
            if r:
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1

        return results

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0

        stats = {
            **self.stats,
            'total_queries': total,
            'hit_rate': f"{hit_rate:.1%}",
        }

        if self.client:
            stats['redis_keys'] = self.client.dbsize()
            info = self.client.info('memory')
            stats['memory_used'] = f"{info['used_memory_human']}"

        return stats

    def clear(self, pattern: str = None):
        """清除缓存"""
        if self.client:
            if pattern:
                keys = self.client.keys(f"{self.prefix}:{pattern}*")
            else:
                keys = self.client.keys(f"{self.prefix}:*")
            if keys:
                self.client.delete(*keys)
                print(f"🗑️  清除 {len(keys)} 条缓存")
        else:
            self._memory_cache.clear()

    def print_stats(self):
        """打印缓存统计"""
        stats = self.get_stats()
        print(f"\n📊 缓存统计:")
        print(f"  命中: {stats['hits']}, 未命中: {stats['misses']}")
        print(f"  命中率: {stats['hit_rate']}")
        print(f"  写入: {stats['sets']}")
        if 'redis_keys' in stats:
            print(f"  Redis键数: {stats['redis_keys']}")
            print(f"  内存占用: {stats['memory_used']}")

class CachedTranslator:
    """带缓存的翻译器包装"""

    def __init__(self, translator, cache: TranslationCache = None):
        """
        Args:
            translator: 原始翻译器（需要有 generate, _format_prompt 方法）
            cache: TranslationCache 实例
        """
        self.translator = translator
        self.cache = cache or TranslationCache()
        self.model_name = 'qwen_lora'

    def translate(self, text: str, source_lang: str, target_lang: str,
                  method: str = 'direct', pivot_text: str = None,
                  temperature: float = 0.3) -> str:
        """带缓存的翻译"""

        def _generate():
            """实际调用模型生成"""
            LANG_NAMES = {'sk': '梵文', 'tb': '藏文', 'cn': '汉文'}
            sn, tn = LANG_NAMES[source_lang], LANG_NAMES[target_lang]

            if method == 'mutual' and pivot_text:
                pivot_lang = list({'sk', 'tb', 'cn'} - {source_lang, target_lang})[0]
                pn = LANG_NAMES[pivot_lang]
                user_msg = f"请将以下{sn}翻译为{tn}，参考{pn}辅助理解。只输出翻译。\n\n{sn}：{text}\n{pn}参考：{pivot_text}"
            else:
                user_msg = f"请将以下{sn}翻译为{tn}，保持古典文献文体。只输出翻译。\n\n{sn}：{text}"

            prompt = self.translator._format_prompt(user_msg)
            return self.translator.generate(prompt, temperature=temperature)

        return self.cache.get_or_generate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            generate_fn=_generate,
            method=method,
            pivot_text=pivot_text,
            model=self.model_name,
            temperature=temperature
        )

    def translate_batch(self, entries: List[Dict], source_lang: str, target_lang: str,
                        method: str = 'direct') -> List[str]:
        """批量翻译（带缓存）"""
        from tqdm import tqdm

        LANG_FIELDS = {'sk': 'sanskrit', 'tb': 'tibetan', 'cn': 'chinese'}
        pivot_lang = list({'sk', 'tb', 'cn'} - {source_lang, target_lang})[0]

        results = []
        cached_count = 0

        for entry in tqdm(entries, desc=f"{source_lang}->{target_lang} ({method})"):
            src = entry[LANG_FIELDS[source_lang]]
            pivot = entry[LANG_FIELDS[pivot_lang]] if method == 'mutual' else None

            # 先查缓存
            cached = self.cache.get(src, source_lang, target_lang, method, pivot, self.model_name)
            if cached:
                results.append(cached)
                cached_count += 1
            else:
                pred = self.translate(src, source_lang, target_lang, method, pivot)
                results.append(pred)

        if cached_count > 0:
            print(f"  ⚡ 缓存命中: {cached_count}/{len(entries)} ({cached_count/len(entries)*100:.1f}%)")

        return results

# ============================================================
# 使用示例和性能测试
# ============================================================

def benchmark_cache():
    """缓存性能基准测试"""
    print("=" * 60)
    print("⚡ Redis 缓存性能测试")
    print("=" * 60)

    cache = TranslationCache()

    # 写入测试
    print("\n写入 1000 条...")
    start = time.time()
    for i in range(1000):
        cache.set(
            text=f"test_text_{i}" * 10,
            source_lang='sk',
            target_lang='cn',
            translation=f"翻译结果_{i}" * 5,
            method='direct'
        )
    write_time = time.time() - start
    print(f"  写入耗时: {write_time:.3f}s ({1000/write_time:.0f} ops/s)")

    # 读取测试
    print("\n读取 1000 条...")
    start = time.time()
    hits = 0
    for i in range(1000):
        result = cache.get(
            text=f"test_text_{i}" * 10,
            source_lang='sk',
            target_lang='cn',
            method='direct'
        )
        if result:
            hits += 1
    read_time = time.time() - start
    print(f"  读取耗时: {read_time:.3f}s ({1000/read_time:.0f} ops/s)")
    print(f"  命中率: {hits/1000*100:.1f}%")

    # 对比：模型推理 vs 缓存读取
    print(f"\n📊 加速比估算:")
    avg_inference_time = 2.5  # 秒/条（基于之前实验）
    avg_cache_time = read_time / 1000
    speedup = avg_inference_time / avg_cache_time
    print(f"  模型推理: ~{avg_inference_time}s/条")
    print(f"  缓存读取: ~{avg_cache_time*1000:.2f}ms/条")
    print(f"  加速比: {speedup:.0f}x")

    # 清理测试数据
    cache.clear('sk:cn:direct')
    cache.print_stats()

if __name__ == "__main__":
    benchmark_cache()
