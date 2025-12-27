#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从病例目录树 + GATA6_label.xlsx 生成单一的数据清单 dataset.csv
并将 136 例划分为：
- teacher 训练集：95 例（默认，其实是 “总数 - 测试数”）
- 内部测试集：41 例（默认，可通过 --teacher-test-n 修改）

支持：
- --labels-sheet 传名称/索引/auto（默认 auto：自动寻找包含 case_ID/编号 与 GATA6 的表）
- Excel 中的“case_ID”或“编号”均可作为病例ID列

输出：
    dataset.csv            # 全部匹配到的清单
    all_with_prefix.csv    # 含核对信息（可选查看）
    teacher_train_*.csv    # teacher 模型训练集（默认 95）
    internal_test_*.csv    # 内部测试集（默认 41）
"""

import argparse
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# 1~4 位起始数字前缀
DIGITS = re.compile(r"^(\d{1,4})")

# 允许作为“病例ID列”的列名（不区分大小写；中文无需 lower）
CASE_ID_ALIASES = {"case_id", "编号", "caseid", "id"}

def num_prefix(s: str) -> str:
    s = str(s).strip()
    m = DIGITS.match(s)
    return m.group(1) if m else ""

def scan_cases(root: Path) -> pd.DataFrame:
    rows = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_full = case_dir.name
        case_num = num_prefix(case_full)
        image = None; mask = None; file_num = ""
        for p in sorted(case_dir.iterdir()):
            n = p.name.lower()
            if n.endswith((".nii", ".nii.gz")):
                if ("image" in n) or ("img" in n):
                    image = p
                    # 仅用于日志核对
                    stem = p.name[:-4] if n.endswith(".nii") else p.name[:-7]
                    fn = num_prefix(stem)
                    if fn:
                        file_num = fn
                elif ("label" in n) or ("mask" in n):
                    mask = p
        if image is None or mask is None:
            print(f"[WARN] 缺少 image/label → 跳过：{case_full}")
            continue
        rows.append({
            "case_id_full": case_full,
            "case_num": case_num,
            "file_num": file_num,
            "image_path": str(image.resolve()),
            "mask_path": str(mask.resolve()),
        })
    return pd.DataFrame(rows)

def _find_sheet_with_columns(xls: pd.ExcelFile) -> str:
    """
    自动在工作簿中寻找同时包含 (case_ID/编号/…) 与 GATA6 列的 sheet，返回 sheet 名称。
    """
    candidates = []
    for s in xls.sheet_names:
        try:
            df = xls.parse(s, nrows=5)
        except Exception:
            continue
        cols_raw = [str(c).strip() for c in df.columns]
        cols_l = [c.lower() for c in cols_raw]

        have_case = any((c in CASE_ID_ALIASES) or (c.lower() in CASE_ID_ALIASES) for c in cols_raw)
        have_g6   = any(c == "gata6" for c in cols_l)
        if have_case and have_g6:
            candidates.append(s)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for s in candidates:
            if "gata6" in s.lower():
                return s
        print(f"[WARN] 检测到多个可能的工作表：{candidates}，将使用第一个：{candidates[0]}")
        return candidates[0]
    raise SystemExit(
        f"未在任何工作表中同时找到病例ID列({list(CASE_ID_ALIASES)})和 GATA6 列。"
        f" 可用工作表：{xls.sheet_names}"
    )

def read_labels_xlsx(path: Path, sheet_spec=None) -> pd.DataFrame:
    """
    读取 Excel 标签，支持：
      - sheet_spec 为空/auto：自动搜索包含 (case_ID/编号/…) 与 GATA6 的工作表
      - sheet_spec 为名称：按名称读取（大小写不敏感）
      - sheet_spec 为整数：按索引读取
    返回列：case_num, label
    """
    xls = pd.ExcelFile(path)
    print(f"[INFO] Excel 文件包含工作表：{xls.sheet_names}")

    # 解析 sheet
    if sheet_spec is None or str(sheet_spec).strip().lower() == "auto":
        use_sheet = _find_sheet_with_columns(xls)
    else:
        ss = str(sheet_spec).strip()
        try:
            idx = int(ss)
            if idx < 0 or idx >= len(xls.sheet_names):
                raise SystemExit(f"--labels-sheet 索引越界：{idx}，可选 0~{len(xls.sheet_names)-1}")
            use_sheet = xls.sheet_names[idx]
        except ValueError:
            lower_map = {s.lower(): s for s in xls.sheet_names}
            if ss in xls.sheet_names:
                use_sheet = ss
            elif ss.lower() in lower_map:
                use_sheet = lower_map[ss.lower()]
            else:
                raise SystemExit(f"--labels-sheet 未找到工作表：{ss}；可选：{xls.sheet_names}")

    print(f"[OK] 使用 Excel 工作表：{use_sheet}")
    df = xls.parse(use_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # 定位病例ID列与 GATA6 列
    case_cols = [c for c in df.columns if (c in CASE_ID_ALIASES) or (c.lower() in CASE_ID_ALIASES)]
    g6_cols   = [c for c in df.columns if c.strip().lower() == "gata6"]
    if not case_cols or not g6_cols:
        raise SystemExit(
            f"工作表 {use_sheet!r} 不包含病例ID列({list(CASE_ID_ALIASES)})和/或 GATA6 列。"
        )

    case_col = case_cols[0]  # 优先第一个匹配
    g6_col   = g6_cols[0]
    sub = df[[case_col, g6_col]].copy()
    sub.columns = ["case_id_raw", "GATA6_raw"]

    # 抽取数字前缀
    sub["case_num"] = sub["case_id_raw"].astype(str).map(num_prefix)

    def map_gata6(v):
        s = str(v).strip().lower()
        if s in {"1","high","positive","阳性","阳","高"}: return 1
        if s in {"0","low","negative","阴性","阴","低"}:   return 0
        try: return int(float(s))
        except: raise ValueError(f"无法识别的 GATA6 值: {v!r}")
    sub["label"] = sub["GATA6_raw"].map(map_gata6).astype(int)

    if (sub["case_num"] == "").any():
        bad = sub[sub["case_num"]==""].head(5)
        raise SystemExit(f"Excel 中存在无法抽取编号前缀的病例ID（示例）：\n{bad}")

    return sub[["case_num","label"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="病例根目录")
    ap.add_argument("--labels", required=True, help="GATA6_label.xlsx 路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--labels-sheet", default="auto",
                    help="Excel 工作表 名称/索引/auto（默认 auto：自动寻找包含 case_ID/编号 与 GATA6 的表）")
    ap.add_argument("--teacher-test-n", type=int, default=41,
                    help="内部测试集中样本数（默认 41；剩余作为 teacher 训练集）")
    ap.add_argument("--seed", type=int, default=2025, help="分层划分随机种子")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 扫目录
    print("\n=================== 扫描病例目录 ===================")
    df_cases = scan_cases(root)
    if df_cases.empty:
        raise SystemExit("未发现任何包含 *_image.nii 与 *_label.nii 的病例。")
    print(f"[OK] 目录扫描到 {len(df_cases)} 个病例文件夹")

    # 2) 读 Excel 标签（支持“编号”/case_ID）
    print("\n=================== 读取 GATA6 标签 ===================")
    df_lab = read_labels_xlsx(Path(args.labels), sheet_spec=args.labels_sheet)
    print(f"[OK] Excel 标签条目：{len(df_lab)}")

    # 3) 统一位宽（01/1 对齐）
    width = max(1, int(df_cases["case_num"].astype(str).str.len().max()))
    df_cases["case_num"] = df_cases["case_num"].astype(str).str.zfill(width)
    df_lab["case_num"]   = df_lab["case_num"].astype(str).str.zfill(width)

    # 4) 合并
    print("\n=================== 对齐与合并 ===================")
    df = df_cases.merge(df_lab, on="case_num", how="inner")
    if df.empty:
        raise SystemExit("目录前缀与 Excel 的病例ID（case_ID/编号）前缀没有交集，请检查。")

    # 5) 输出 dataset.csv / all_with_prefix.csv
    df["case_id"] = df["case_num"]
    dataset_cols = ["case_id","image_path","mask_path","label"]
    dataset_csv = outdir / "dataset.csv"
    df[dataset_cols].to_csv(dataset_csv, index=False, encoding="utf-8-sig")

    all_csv = outdir / "all_with_prefix.csv"
    df[["case_id","case_num","case_id_full","file_num","image_path","mask_path","label"]].to_csv(
        all_csv, index=False, encoding="utf-8-sig"
    )

    mism = df[(df["case_num"]!="") & (df["file_num"]!="") & (df["case_num"]!=df["file_num"])]
    if len(mism):
        print(f"[WARN] 发现 {len(mism)} 个病例：文件夹前缀 != 文件名前缀（以文件夹前缀为准）。")

    print(f"[OK] 写入：{dataset_csv}  ({len(df)})")
    print(f"[OK] 写入：{all_csv}     ({len(df)})")
    if len(mism):
        print(f"[WARN] 详情见：{all_csv}（含 case_id_full 与 file_num 便于核对）")

    # 6) 教师模型 95/41 划分（或按 --teacher-test-n）
    print("\n=================== 教师模型划分（训练/内部测试） ===================")
    total = len(df)
    test_n = int(args.teacher_test_n)
    if test_n <= 0 or test_n >= total:
        test_n = max(1, min(total-1, 41))
        print(f"[WARN] --teacher-test-n={args.teacher_test_n} 不合理，自动改为 {test_n}")

    y_all = df["label"].astype(int).values
    try:
        df_train, df_test = train_test_split(
            df[dataset_cols], test_size=test_n,
            stratify=y_all, random_state=args.seed, shuffle=True
        )
        stratified = True
    except ValueError as e:
        print(f"[WARN] 分层抽样失败（{e}），将改为不分层的随机划分。")
        df_train, df_test = train_test_split(
            df[dataset_cols], test_size=test_n,
            random_state=args.seed, shuffle=True
        )
        stratified = False

    train_path = outdir / f"teacher_train_{len(df_train)}.csv"
    test_path  = outdir / f"internal_test_{len(df_test)}.csv"
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    tr_cnt = df_train["label"].value_counts().to_dict()
    te_cnt = df_test["label"].value_counts().to_dict()
    print(f"[OK] 教师训练集：{len(df_train)} 例 | 标签分布：{tr_cnt}")
    print(f"[OK] 内部测试集：{len(df_test)} 例 | 标签分布：{te_cnt}")
    print(f"[INFO] 划分方式：{'分层' if stratified else '非分层随机'} | 随机种子：{args.seed}")
    print(f"[OK] 写入：{train_path}")
    print(f"[OK] 写入：{test_path}")

    print("\n=================== 完成 ===================")
    print("✅ 已生成 dataset.csv + all_with_prefix.csv，并完成 95/41（默认）教师模型划分。")

if __name__ == "__main__":
    main()
