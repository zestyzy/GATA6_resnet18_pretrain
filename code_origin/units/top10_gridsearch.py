#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
top10_gridsearch.py
----------------------------------------
扫描 gridsearch 的输出目录，读取每个 run 目录下的 shuffle_auc.csv，
计算 mean/std/95%CI，并据此选出综合表现最好的 Top-10。

用法示例：
  python /data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/code/units/top10_gridsearch.py \
      --root /data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/results/transformer_test/grid_search \
      --out  /data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/results/transformer_test/grid_search

如未指定 --out，则默认写回到 --root 目录。
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

def ci95(mean: float, std: float, n: int):
    """基于 t 分布计算 95% 置信区间。"""
    if n <= 1 or np.isnan(std):
        return mean, mean
    try:
        from scipy.stats import t
        t_crit = t.ppf(0.975, df=n-1)
    except Exception:
        # df=4 的近似值
        t_crit = 2.776
    half = t_crit * std / np.sqrt(n)
    return max(0.0, mean - half), min(1.0, mean + half)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="gridsearch 的根目录（包含各 run 子目录）")
    ap.add_argument("--out",  default=None, help="输出目录（默认与 root 一致）")
    ap.add_argument("--pattern", default="shuffle_auc.csv",
                    help="每个 run 目录中的 AUC 明细文件名（默认 shuffle_auc.csv）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out or args.root)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    # 递归查找所有 shuffle_auc.csv
    for f in glob.glob(os.path.join(root, "**", args.pattern), recursive=True):
        run_dir = os.path.dirname(f)
        tag = os.path.basename(run_dir)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] 读取失败，跳过: {f} | {e}")
            continue

        if "auc" not in df.columns:
            print(f"[WARN] 文件中无 'auc' 列，跳过: {f}")
            continue

        aucs = df["auc"].to_numpy(dtype=float)
        n = np.count_nonzero(~np.isnan(aucs))
        if n == 0:
            print(f"[WARN] 无有效 AUC，跳过: {f}")
            continue

        mean = float(np.nanmean(aucs))
        std  = float(np.nanstd(aucs, ddof=1)) if n > 1 else float("nan")
        lo, hi = ci95(mean, std, n)

        rows.append({
            "tag": tag,
            "dir": run_dir,
            "n_folds": n,
            "mean_auc": mean,
            "std_auc": std,
            "ci_low": lo,
            "ci_high": hi,
        })

    if not rows:
        print(f"[ERROR] 在 {root} 下没有找到任何 {args.pattern}")
        return

    all_df = pd.DataFrame(rows)

    # 综合排序：mean_auc 降序，其次 std_auc 升序（更稳定优先）
    all_df = all_df.sort_values(
        by=["mean_auc", "std_auc"],
        ascending=[False, True],
        kind="mergesort"  # 稳定排序
    ).reset_index(drop=True)

    # 保存总表 & Top-10
    all_csv = os.path.join(out_dir, "all_runs_summary.csv")
    top10_csv = os.path.join(out_dir, "top10.csv")

    all_df.to_csv(all_csv, index=False)
    top10 = all_df.head(10).copy()
    top10.to_csv(top10_csv, index=False)

    print("\n===== Top-10 (by mean_auc desc, std_auc asc) =====")
    # 只打印关键信息
    print(top10[["tag", "mean_auc", "std_auc", "ci_low", "ci_high", "n_folds", "dir"]].to_string(index=False))
    print(f"\n[OK] 全量汇总: {all_csv}")
    print(f"[OK] Top-10: {top10_csv}")

if __name__ == "__main__":
    main()
