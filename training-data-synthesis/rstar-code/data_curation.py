#!/usr/bin/env python3
"""
rStar-Coder seed_sft 数据清洗脚本

Cleaning pipeline:
1. Retain only samples marked both verified and is_passed.
2. Remove samples whose chain-of-thought exceeds 16k tokens.
3. Deduplicate problems by adjacent-hash matching on the problem statement.

"""

import os
import json
import hashlib
import pyarrow.parquet as pq
import gc
from collections import defaultdict

DATA_DIR = "data/to/rStar-Coder/seed_sft"
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])

# 16k tokens ≈ 64k chars (rough estimate: 1 token ≈ 4 chars)
MAX_COT_TOKENS = 16000
MAX_COT_CHARS = MAX_COT_TOKENS * 4


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数 (4 chars ≈ 1 token)"""
    return len(text) // 4


def adjacent_hash(text: str, ngram_size: int = 5) -> str:
    """
    基于相邻 n-gram 的哈希，用于问题文本去重。
    对问题文本进行规范化后，取相邻 word n-gram 的集合哈希。
    """
    # 规范化：小写，去除多余空白
    normalized = ' '.join(text.lower().split())
    words = normalized.split()
    if len(words) < ngram_size:
        # 文本太短，直接用全文哈希
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    # 取所有相邻 n-gram
    ngrams = set()
    for i in range(len(words) - ngram_size + 1):
        ngram = ' '.join(words[i:i + ngram_size])
        ngrams.add(ngram)
    
    # 对 n-gram 集合排序后哈希
    ngram_str = '\n'.join(sorted(ngrams))
    return hashlib.md5(ngram_str.encode('utf-8')).hexdigest()


# ============================================================
# Step 1 & 2: Filter by verified + is_passed, remove long CoT
# ============================================================
print("Step 1 & 2: Filtering (verified=True & is_passed=True, CoT <= 16k tokens)...")

output_data = []
stats = {
    'total_rows': 0,
    'not_verified': 0,
    'not_passed': 0,
    'cot_too_long': 0,
    'passed_filter': 0,
}

for fi, fpath in enumerate(files):
    pf = pq.ParquetFile(fpath)
    file_kept = 0
    
    for rg in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(rg)
        n_rows = batch.num_rows
        stats['total_rows'] += n_rows
        
        # 获取所有需要的列
        qids = batch.column('question_id').to_pylist()
        is_passed_list = batch.column('is_passed').to_pylist()
        verified_list = batch.column('verified').to_pylist()
        questions = batch.column('question').to_pylist()
        responses = batch.column('response').to_pylist()
        codes = batch.column('code').to_pylist()
        
        # 尝试获取 starter_code（可能不存在）
        try:
            starters = batch.column('starter_code').to_pylist()
        except KeyError:
            starters = [""] * n_rows
        
        for i in range(n_rows):
            # Step 1: 必须同时 verified 和 is_passed
            if not verified_list[i]:
                stats['not_verified'] += 1
                continue
            if not is_passed_list[i]:
                stats['not_passed'] += 1
                continue
            
            qid = qids[i]
            q = questions[i] or ""
            r = responses[i] or ""
            c = codes[i] or ""
            sc = starters[i] or ""
            
            # Step 2: CoT (response) 不超过 16k tokens
            cot_tokens = estimate_tokens(r)
            if cot_tokens > MAX_COT_TOKENS:
                stats['cot_too_long'] += 1
                continue
            
            stats['passed_filter'] += 1
            file_kept += 1
            
            # 构建 SFT 格式
            user_msg = q
            if sc.strip():
                user_msg += f"\n\nStarter code:\n```python\n{sc}\n```"
            
            assistant_msg = r
            if c.strip() and c.strip()[:50] not in r[-len(c)-500:]:
                assistant_msg += f"\n\n```python\n{c}\n```"
            
            entry = {
                "id": qid,
                "question_text": q,  # 临时保留用于去重
                "conversations": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ],
                "metadata": {
                    "question_length": len(q),
                    "response_length": len(r),
                    "code_length": len(c),
                    "cot_tokens": cot_tokens,
                }
            }
            output_data.append(entry)
        
        del batch, qids, is_passed_list, verified_list, questions, responses, codes, starters
    
    print(f"  [{fi+1}/{len(files)}] {os.path.basename(fpath)}: kept={file_kept}", flush=True)
    gc.collect()

print(f"\nStep 1 & 2 done.")
print(f"  Total rows scanned: {stats['total_rows']:,}")
print(f"  Not verified: {stats['not_verified']:,}")
print(f"  Not passed: {stats['not_passed']:,}")
print(f"  CoT > 16k tokens: {stats['cot_too_long']:,}")
print(f"  Passed filter: {stats['passed_filter']:,}")

# ============================================================
# Step 3: Deduplicate by adjacent-hash matching on problem statement
# ============================================================
print(f"\nStep 3: Deduplicating by adjacent-hash on problem statement...")
print(f"  Before dedup: {len(output_data):,}")

seen_hashes = set()
deduped_data = []

for entry in output_data:
    q_text = entry['question_text']
    h = adjacent_hash(q_text)
    
    if h not in seen_hashes:
        seen_hashes.add(h)
        deduped_data.append(entry)

duplicates_removed = len(output_data) - len(deduped_data)
print(f"  Duplicates removed: {duplicates_removed:,}")
print(f"  After dedup: {len(deduped_data):,}")

# 清理临时字段
for entry in deduped_data:
    del entry['question_text']

output_data = deduped_data
del deduped_data
gc.collect()

# ============================================================
# 保存结果
# ============================================================
# 按 id 排序
output_data.sort(key=lambda x: x['id'])

# 保存 JSONL
output_jsonl = os.path.join(OUTPUT_DIR, "seed_sft_cleaned.jsonl")
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for entry in output_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
print(f"\nSaved JSONL: {output_jsonl}")

# 保存 JSON
output_json = os.path.join(OUTPUT_DIR, "seed_sft_cleaned.json")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print(f"Saved JSON: {output_json}")

# ============================================================
# 统计
# ============================================================
n = len(output_data)
if n > 0:
    q_lens = sorted([x['metadata']['question_length'] for x in output_data])
    r_lens = sorted([x['metadata']['response_length'] for x in output_data])
    c_lens = sorted([x['metadata']['code_length'] for x in output_data])
    cot_tokens_list = sorted([x['metadata']['cot_tokens'] for x in output_data])

    print(f"\n{'='*50}")
    print(f"Final Dataset Stats")
    print(f"{'='*50}")
    print(f"Total entries: {n:,}")
    print(f"Q length (chars): min={q_lens[0]}, P50={q_lens[n//2]}, max={q_lens[-1]}")
    print(f"R length (chars): min={r_lens[0]}, P50={r_lens[n//2]}, max={r_lens[-1]}")
    print(f"C length (chars): min={c_lens[0]}, P50={c_lens[n//2]}, max={c_lens[-1]}")
    print(f"CoT tokens:       min={cot_tokens_list[0]}, P50={cot_tokens_list[n//2]}, max={cot_tokens_list[-1]}")

    # 总 token 估算
    total_chars = sum(x['metadata']['response_length'] + x['metadata']['question_length'] for x in output_data)
    print(f"\nEstimated total tokens: ~{total_chars // 4:,}")
    print(f"Average tokens per entry: ~{total_chars // 4 // n:,}")
else:
    print("\nNo data remaining after filtering!")

print("\nDone!")
