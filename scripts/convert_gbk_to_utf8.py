#!/usr/bin/env python
"""将 DeepRAG 结果目录中的 GBK/GB18030 编码 JSONL 文件转成 UTF-8。

用于恢复旧版本（未指定 encoding="utf-8"）在 Windows 中文环境（默认 GBK/cp936）
下写入的结果文件。流程：备份目录 -> 逐文件转码 -> 逐行 json.loads 验证，
最后一行因写入失败而残缺时只丢弃该残缺行，其余记录保留。

注意：
- 转换前会先把整个目录备份到 <dir>.gbk_backup。
- 已是合法 UTF-8 的文件会跳过，不会重复转码。
- 不要使用 errors="ignore"，避免静默损坏预测答案。

用法:
    python scripts/convert_gbk_to_utf8.py --dir results/smoke_hotpotqa/0
"""
import os
import json
import shutil
import argparse


def convert_file(path):
    with open(path, "rb") as f:
        raw = f.read()

    # 已经是合法 UTF-8 则跳过
    try:
        raw.decode("utf-8")
        print(f"[skip] already utf-8: {path}")
        return
    except UnicodeDecodeError:
        pass

    # 按 GB18030（GBK 超集）解码，失败再退回 GBK
    try:
        text = raw.decode("gb18030")
        enc = "gb18030"
    except UnicodeDecodeError:
        text = raw.decode("gbk")
        enc = "gbk"

    lines = text.split("\n")
    kept = []
    dropped = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
            kept.append(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                # 最后一行因之前写入失败而残缺，只丢弃这一行
                dropped += 1
                print(f"[warn] drop incomplete last line: {path}")
            else:
                # 中间行损坏则保留原样并告警，避免静默丢数据
                print(f"[error] corrupt line {i} (not last), kept as-is: {line[:80]}...")
                kept.append(line)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"[done] {path}: {enc} -> utf-8, kept {len(kept)} lines, dropped {dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="结果目录（含 output.jsonl / output_subprocess_*.jsonl）")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.dir)
    if not os.path.isdir(result_dir):
        raise SystemExit(f"目录不存在: {result_dir}")

    # 1. 备份目录
    backup_dir = result_dir.rstrip("/\\") + ".gbk_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(result_dir, backup_dir)
        print(f"[backup] {result_dir} -> {backup_dir}")
    else:
        print(f"[backup] already exists, skip: {backup_dir}")

    # 2. 转码所有 jsonl 结果文件
    targets = ["output.jsonl"] + sorted(
        f for f in os.listdir(result_dir)
        if f.startswith("output_subprocess_") and f.endswith(".jsonl")
    )
    for fname in targets:
        path = os.path.join(result_dir, fname)
        if os.path.exists(path):
            convert_file(path)

    print("转换完成。请用 evaluate_ans.py 重新评测验证。")


if __name__ == "__main__":
    main()
