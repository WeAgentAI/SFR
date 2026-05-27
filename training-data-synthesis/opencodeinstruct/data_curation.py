"""
筛选 nvidia/OpenCodeInstruct 最优数据 + 完全相同题目去重

筛选条件：
1. llm_judgement 中:
   - requirement_conformance score = 5
   - logical_correctness score = 5
   - edge_case_consideration score >= 4
2. tests_execution_status 中所有测试用例均为 "pass"

去重策略：
- 对 input 文本做 strip 后精确匹配去重
- 同一题目保留 edge_case_consideration 最高、output 最长的那条

输出: JSONL 格式
"""

import os
import json
import glob
import hashlib

import pandas as pd
from multiprocessing import Pool, cpu_count

INPUT_DIR = "data/to/opencodeinstruct/data"
OUTPUT_DIR = ""
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "opencodeinstruct_best.jsonl")


def is_best_sample(row):
    """判断一条数据是否满足最优条件，返回 (通过?, edge_score)"""
    try:
        judgement = json.loads(row["llm_judgement"])
        req_score = judgement.get("requirement_conformance", {}).get("score", 0)
        logic_score = judgement.get("logical_correctness", {}).get("score", 0)
        edge_score = judgement.get("edge_case_consideration", {}).get("score", 0)

        if req_score < 5 or logic_score < 5 or edge_score < 4:
            return False, 0

        statuses = json.loads(row["tests_execution_status"])
        if not all(s == "pass" for s in statuses):
            return False, 0

        return True, edge_score
    except (json.JSONDecodeError, TypeError, KeyError):
        return False, 0


def process_parquet(filepath):
    """处理单个 parquet 文件，返回筛选后的记录列表"""
    filename = os.path.basename(filepath)
    df = pd.read_parquet(filepath)
    total = len(df)

    results = []
    for _, row in df.iterrows():
        passed, edge_score = is_best_sample(row)
        if passed:
            results.append({
                "id": row["id"],
                "input": row["input"],
                "output": row["output"],
                "domain": row["domain"],
                "generation_algorithm": row["generation_algorithm"],
                "_edge_score": edge_score,
            })

    print(f"  {filename}: {len(results)}/{total} quality-filtered ({len(results)/total*100:.1f}%)")
    return results


def dedup_exact(records):
    """按 input 精确去重，保留质量最好的"""
    print(f"\nDedup: processing {len(records)} records...")

    # input md5 -> best record index
    best_by_hash = {}
    for i, rec in enumerate(records):
        key = hashlib.md5(rec["input"].strip().encode("utf-8")).hexdigest()
        if key not in best_by_hash:
            best_by_hash[key] = i
        else:
            old = records[best_by_hash[key]]
            # 优先 edge_score 高，其次 output 短（更简洁）
            if (rec["_edge_score"], -len(rec["output"])) > (old["_edge_score"], -len(old["output"])):
                best_by_hash[key] = i

    kept = set(best_by_hash.values())
    dup_count = len(records) - len(kept)
    print(f"  Unique inputs: {len(kept)}, duplicates removed: {dup_count}")
    return kept


def main():
    parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "train-*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Step 1: Quality filtering")
    print(f"  - requirement_conformance = 5")
    print(f"  - logical_correctness = 5")
    print(f"  - edge_case_consideration >= 4")
    print(f"  - all tests pass")
    print(f"Step 2: Exact input dedup")
    print()

    # Step 1: 质量筛选 (多进程)
    num_workers = min(cpu_count(), 16)
    print(f"Using {num_workers} workers for quality filtering...")
    with Pool(num_workers) as pool:
        all_results = pool.map(process_parquet, parquet_files)

    all_records = []
    for results in all_results:
        all_records.extend(results)
    print(f"\nTotal quality-filtered: {len(all_records)}")

    # Step 2: 精确去重
    kept_indices = dedup_exact(all_records)

    # Step 3: 写入
    total_selected = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in sorted(kept_indices):
            rec = all_records[i]
            rec.pop("_edge_score", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_selected += 1

    print(f"\nDone!")
    print(f"Final selected: {total_selected} (from ~5,000,000 original)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
