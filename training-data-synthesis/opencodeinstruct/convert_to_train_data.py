"""
将清洗后的代码数据转为 SFT 训练数据格式

输入格式: {"id", "input", "output", "domain", "generation_algorithm"}
输出格式: {"messages": [{"role": "system", "content": ...}, {"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""

import json
import os

INPUT_FILE = ""
SYSTEM_PROMPT_FILE = ""
OUTPUT_FILE = ""


def main():
    # 读取 system prompt
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    print(f"System prompt: {system_prompt}")

    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            train_sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data["input"]},
                    {"role": "assistant", "content": data["output"]},
                ]
            }
            fout.write(json.dumps(train_sample, ensure_ascii=False) + "\n")
            total += 1

    print(f"Done! {total} training samples written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
