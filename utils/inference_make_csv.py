#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从“统一映射后的数据根目录”生成配对 CSV（去重版）：
  columns: case_id, image_path, mask_path
  - 仅收录成对的 image/label
  - case_id 尽量取数字；若同一个数字出现多次，则依次命名为 7, 7-2, 7-3 ...
  - 排序：数字优先，其次按数值升序，再按重复序号升序；无数字的排在所有数字之后
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import csv
from typing import Optional, Tuple, List, Dict

IMG_NAMES  = ("image.nii", "image.nii.gz")   # 只认 image，忽略 image2
LAB_NAMES  = ("label.nii", "label.nii.gz")

NUM_PATS = [
    re.compile(r"^(\d+)$"),        # 纯数字目录，如 01 / 123
    re.compile(r".*?(\d+).*"),     # 含数字的目录，如 pancreas_001
]

def is_nii_or_gz(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def find_pair_in_dir(d: Path) -> Optional[Tuple[Path, Path]]:
    """在目录 d 中找到 image 与 label 的成对文件。"""
    if not d.is_dir():
        return None
    files = {f.name.lower(): f for f in d.iterdir() if f.is_file() and is_nii_or_gz(f)}
    img = next((files[nm] for nm in IMG_NAMES if nm in files), None)
    lab = next((files[nm] for nm in LAB_NAMES if nm in files), None)
    if img is not None and lab is not None:
        return img.resolve(), lab.resolve()
    return None

def extract_numeric_from_path(dirpath: Path) -> Tuple[Optional[int], str]:
    """
    从目录层级中尽量抽取“数字 case_id”。优先最内层目录名，若无则逐层向上。
    返回 (num_or_none, fallback_str)
    """
    for comp in [dirpath.name] + [p.name for p in dirpath.parents]:
        for pat in NUM_PATS:
            m = pat.match(comp)
            if m:
                try:
                    return int(m.group(1)), comp
                except Exception:
                    pass
    return None, dirpath.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="统一后的数据根目录（上一个映射脚本的 --out）")
    ap.add_argument("--out", type=Path, default=Path("./dataset.csv"), help="输出 CSV 路径")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"根目录不存在：{root}")

    rows: List[Tuple[str, str, str, Tuple[int, int, int, str]]] = []
    counts: Dict[str, int] = {}  # 记录每个“基础 case_id”（数字或回退名）的出现次数

    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        pair = find_pair_in_dir(d)
        if pair is None:
            continue
        img, lab = pair

        num, fallback = extract_numeric_from_path(d)
        base_id = str(num) if num is not None else fallback  # 基础 ID（可能会重复）

        # 去重：给重复的基础 ID 添加 -2、-3 后缀
        counts[base_id] = counts.get(base_id, 0) + 1
        dup_idx = counts[base_id]                     # 第几次出现
        final_id = base_id if dup_idx == 1 else f"{base_id}-{dup_idx}"

        # 排序键：
        #   数字优先 -> (0/1)
        #   数值升序 -> num or 大数占位
        #   重复序号 -> dup_idx
        #   最后按 final_id 稳定化
        sort_key = (0 if num is not None else 1,
                    num if num is not None else 10**12,
                    dup_idx,
                    final_id)

        rows.append((final_id, str(img), str(lab), sort_key))

    if not rows:
        raise SystemExit("未在任何目录中找到 image/label 成对文件。")

    rows.sort(key=lambda x: x[3])

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "image_path", "mask_path"])
        for case_id, img, lab, _ in rows:
            w.writerow([case_id, img, lab])

    print(f"已写出 {len(rows)} 行到 {args.out}")
    # 额外提示：展示有重复基础 ID 的情况（便于你检查数据）
    dup_bases = {bid: c for bid, c in counts.items() if c > 1}
    if dup_bases:
        top = ", ".join(f"{k}×{v}" for k, v in sorted(dup_bases.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
        print(f"注意：存在重复的基础 case_id，共 {len(dup_bases)} 个，例如：{top}")

if __name__ == "__main__":
    main()
