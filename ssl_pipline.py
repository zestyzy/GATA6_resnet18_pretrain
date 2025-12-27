#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ssl_pipline.py — 一键 SSL 管线（适配 ResNet18 + BBox + Margin 空间对齐版本）
最终输出给分析脚本的 CSV 仅包含：
    case_id, image_path, mask_path, label + 纯数值 radiomics 特征列

显式剔除的辅助列：
    conf, mu, var, alea, n_rep, __ord_true, __ord_conf
并过滤所有以下划线 '_' 开头的技工列。

用法示例：
python ssl_pipline.py \
  --out ./results/test \
  --labeled-136-csv ./dataset/labeled.csv \
  --unlabeled-550-csv ./dataset/unlabeled.csv \
  --external-54-csv ./dataset/external.csv \
  --rad-config ./config/config.json \
  --pretrain-path ./weights/resnet18_23dataset.pth \
  --epochs 30 --batch-size 8 --workers 8 --cv 10 --topk 10 \
  --keep-fracs 0.3 0.5 0.7 0.9 --teacher-use-youden \
  --bbox --bbox-pad 20
"""

from __future__ import annotations
import argparse, os, sys, json, shutil, subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==== 按你的环境改这两个常量（其它无需改） ==== #
EXACT_SCRIPT    = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/code_temp/analysis/exact_radiomics_from_csv.py"
ANALYSIS_SCRIPT = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/code_temp/analysis/radiomics_train_test_topk.py"
# ================================================= #

def run(cmd: List[str], cwd: Optional[str]=None):
    print("$", " ".join(cmd)); subprocess.run(cmd, check=True, cwd=cwd)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    m = {}
    for std, alts in {
        "case_id":    ["case_id","caseid","case","编号","id"],
        "image_path": ["image_path","imagepath","img_path","ct_path","ct"],
        "mask_path":  ["mask_path","maskpath","seg_path","label_path","mask"],
    }.items():
        hit = None
        for a in alts:
            if a in low: hit = low[a]; break
        if hit is None: raise SystemExit(f"CSV 缺少列：{std}（允许别名：{alts}）")
        m[hit] = std
    if "label" in df.columns: m["label"] = "label"
    return df.rename(columns=m)

def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)

# ---------- 1) 136 → 95/41 (自动适配 Debug 模式) ----------
def split_136(labeled_csv: Path, seed: int, outdir: Path) -> Tuple[Path, Path]:
    outdir = ensure_dir(outdir)
    tr_p, te_p = outdir / "train95.csv", outdir / "int41.csv"
    
    # 强制重新划分（Debug时可能需要反复刷），如果想保留缓存逻辑可以不去掉下面这两行
    # if tr_p.exists() and te_p.exists():
    #     print(f"[SPLIT] 复用现有拆分：{tr_p} / {te_p}"); return tr_p, te_p
    
    df = std_cols(pd.read_csv(labeled_csv, encoding="utf-8-sig", low_memory=False))
    if "label" not in df.columns: raise SystemExit("labeled-136-csv 必须包含 label 列")
    y = df["label"].astype(int).values
    
    # [修改点] 动态决定拆分大小
    total_samples = len(df)
    if total_samples > 50:
        # 正常模式：严格按照 41 例做内部测试集
        t_size = 41
        print(f"[SPLIT] 标准模式: 总数 {total_samples} -> 拆分 {total_samples-41}/{41}")
    else:
        # Debug 模式：数据太少，切不出 41 例，改用 30% 比例
        t_size = 0.15
        print(f"[SPLIT] 调试模式: 总数 {total_samples} -> 切换为比例拆分 test_size={t_size}")

    try:
        tr, te = train_test_split(df, test_size=t_size, random_state=seed, stratify=y)
    except ValueError as e:
        # 极度少样本（比如正样本不够分层抽样）时，取消 stratify
        print(f"[WARN] 分层抽样失败（可能样本太少），降级为随机抽样: {e}")
        tr, te = train_test_split(df, test_size=t_size, random_state=seed)

    tr.to_csv(tr_p, index=False, encoding="utf-8-sig")
    te.to_csv(te_p, index=False, encoding="utf-8-sig")
    return tr_p, te_p

# ---------- 2) Teacher 训练/复用 ----------
def train_teacher(
    train95_csv: Path, 
    teacher_root: Path, 
    epochs: int, 
    workers: int, 
    batch_size: int, 
    seed: int, 
    val_split: float,
    pretrain_path: str | None = None
) -> Path:
    # [修改点 1] 目录名改为 resnet18_bbox 反映真实配置
    run_dir = ensure_dir(teacher_root / f"teacher_resnet18_2ch_bbox_e{epochs}")
    
    reps = list_rep_dirs(run_dir)
    if reps:
        print(f"[TEACHER] 检测到现有 repeats：{[r.name for r in reps]}，跳过训练。"); return run_dir
    
    cmd = [
        sys.executable, "-m", "teacher_model.train",
        "--data-root", "/",
        "--label-csv", str(train95_csv),
        "--outdir", str(run_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        
        # [关键修改] 切换为 ResNet18 + 适合微调的低学习率
        "--arch", "resnet18",
        "--lr", "0.00005", # 5e-5，预训练微调推荐值 (若无预训练，train.py 会自动 fallback 到 1e-5 或 train.py 默认值)
        "--in-channels", "2",
        "--norm", "gn", "--gn-groups", "32",
        
        # [关键修改] 空间参数适配新 train.py
        "--crop-mode", "bbox",         
        "--margin", "6",            # 6mm 物理扩边
        "--target-size", "112", "144", "144",
        "--target-spacing", "1.5", "1.0", "1.0",

        "--val-split", str(val_split),
        "--seed", str(seed),
        "--use-light-aug",
    ]

    # [新增] 传入预训练权重路径
    if pretrain_path and os.path.exists(pretrain_path):
        cmd += ["--pretrain-path", str(pretrain_path)]
    elif pretrain_path:
        print(f"[WARN] 指定的预训练权重不存在: {pretrain_path}，将使用默认初始化")

    run(cmd); return run_dir

def list_rep_dirs(run_dir: Path) -> List[Path]:
    reps = []
    if not run_dir.exists(): return reps
    for p in sorted(run_dir.iterdir()):
        if p.is_dir() and p.name.startswith("rep_") and (p/"last_model.pth").exists():
            reps.append(p)
    return reps

def copy_last_as_best(rep_dir: Path) -> Path:
    tmp = ensure_dir(rep_dir/"use_last_for_infer")
    for fn in ["channel_stats.pt","metrics.csv","loss_curve.png"]:
        src = rep_dir/fn
        if src.exists(): shutil.copy2(src, tmp/src.name)
    last = rep_dir/"last_model.pth"
    if not last.exists(): raise SystemExit(f"未发现 {last}")
    shutil.copy2(last, tmp/"best_model.pth")
    print(f"[WEIGHTS] ({rep_dir.name}) 使用 last_model.pth 作为推理权重 (best_model.pth)")
    return tmp

# ---------- 3) 伪标签推理（默认 thr=0.5；可选 logit/Youden） ----------
def infer_pseudolabels(
    unlabeled_csv: Path,
    infer_run_dir: Path,
    outdir: Path,
    workers: int,
    batch_size: int,
    force_thr: float = 0.5,
    *,
    use_youden: bool = False,
    logit_q: float | None = None,
    logit_thr: float | None = None,
    device: str = "auto",
) -> Path:
    """
    生成伪标签（已适配 ResNet18 + BBox 逻辑）。
    必须确保推理时的裁剪（ROI）、分辨率与训练时完全一致。
    """
    if (logit_q is not None) and (logit_thr is not None):
        raise SystemExit("参数冲突：--teacher-logit-q 与 --teacher-logit-thr 只能二选一。")

    outdir = ensure_dir(outdir)

    if device == "auto":
        dev = "cuda" if torch_cuda_available() else "cpu"
    else:
        dev = device

    cmd = [
        sys.executable, "-m", "teacher_model.inference",
        "--csv", str(unlabeled_csv),
        "--run-dir", str(infer_run_dir),
        "--out", str(outdir),
        
        # [关键修改] 推理配置必须与训练 (train_teacher) 严丝合缝
        "--arch", "resnet18",          # 必须是 resnet18
        "--in-ch", "2", 
        "--norm", "gn", "--gn-groups", "32",
        
        # [关键修改] 空间参数必须一致
        "--crop-mode", "bbox",         
        "--margin", "6",
        "--target-size", "112", "144", "144",
        "--target-spacing", "1.5", "1.0", "1.0",

        "--batch-size", str(batch_size), 
        "--workers", str(workers),
        "--device", dev,
    ]

    # 概率阈值策略
    if use_youden:
        cmd += ["--use-youden"]
    else:
        cmd += ["--thr", str(force_thr)]

    # logit 幅值策略
    if logit_q is not None:
        cmd += ["--logit-q", str(logit_q)]
    if logit_thr is not None:
        cmd += ["--logit-thr", str(logit_thr)]

    run(cmd)
    out_csv = outdir/"inference.csv"
    if not out_csv.exists(): raise SystemExit(f"未找到伪标签输出 inference.csv：{out_csv}")
    return out_csv

# ---------- 4) Radiomics 特征提取 ----------
def make_train_pool_csv(train95_csv: Path, unlabeled550_csv: Path, outdir: Path) -> Path:
    tr = std_cols(pd.read_csv(train95_csv,  encoding="utf-8-sig", low_memory=False))
    ul = std_cols(pd.read_csv(unlabeled550_csv, encoding="utf-8-sig", low_memory=False))
    all_df = pd.concat([tr[["case_id","image_path","mask_path","label"]],
                        ul[["case_id","image_path","mask_path"]]], ignore_index=True)
    p = outdir/"train_pool_95_plus_550.csv"; all_df.to_csv(p, index=False, encoding="utf-8-sig"); return p

def extract_feats_base(csv_path: Path, config: Path, out_csv: Path, *, use_bbox: bool, bbox_pad: int, bin_count: int):
    # 注意：此处 --bbox-pad 是传给 EXACT_SCRIPT (pyradiomics) 的，通常保持原样
    cmd = [sys.executable, EXACT_SCRIPT,
           "--csv", str(csv_path), "--config", str(config), "--out", str(out_csv),
           "--join-key", "case_id", "--bin-count", str(bin_count), "--no-prob"]
    if not use_bbox: cmd += ["--no-bbox"]
    if bbox_pad > 0: cmd += ["--bbox-pad", str(bbox_pad)]
    run(cmd)

def extract_feats_test(csv_path: Path, config: Path, out_csv: Path, *, use_bbox: bool, bbox_pad: int, bin_count: int) -> Path:
    cmd = [sys.executable, EXACT_SCRIPT,
           "--csv", str(csv_path), "--config", str(config), "--out", str(out_csv),
           "--join-key", "case_id", "--bin-count", str(bin_count)]
    if not use_bbox: cmd += ["--no-bbox"]
    if bbox_pad > 0: cmd += ["--bbox-pad", str(bbox_pad)]
    run(cmd); return out_csv

def attach_labels(feat_csv: Path, label_map_csv: Path, out_csv: Path) -> Path:
    f  = std_cols(pd.read_csv(feat_csv, encoding="utf-8-sig", low_memory=False))
    lm = std_cols(pd.read_csv(label_map_csv, encoding="utf-8-sig", low_memory=False))[["case_id","label"]].drop_duplicates("case_id")
    m = f.merge(lm, on="case_id", how="left", suffixes=("", "_true"))
    if "label_true" in m.columns:
        m["label"] = np.where(m["label_true"].notna(), m["label_true"], m.get("label", np.nan))
        m.drop(columns=["label_true"], inplace=True)
    m.to_csv(out_csv, index=False, encoding="utf-8-sig"); return out_csv

# ---------- 5) 计算“集成置信度” ----------
def compute_ensemble_confidence(rep_to_pseudo: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for rep, path in rep_to_pseudo.items():
        df = std_cols(pd.read_csv(path, encoding="utf-8-sig", low_memory=False))
        if "prob_pos" in df.columns: df = df.rename(columns={"prob_pos":"gata6_prob"})
        take = df[["case_id","gata6_prob"]].dropna().copy()
        take["case_id"] = take["case_id"].astype(str)
        take.rename(columns={"gata6_prob": f"p_{rep}"}, inplace=True)
        frames.append(take)
    if not frames: raise SystemExit("没有任何 repeat 的伪标签概率。")
    base = frames[0]
    for fr in frames[1:]:
        base = base.merge(fr, on="case_id", how="outer")

    p_cols = [c for c in base.columns if c.startswith("p_")]
    mu_list, var_list, alea_list, n_list = [], [], [], []
    for _, row in base.iterrows():
        ps = [float(row[c]) for c in p_cols if pd.notna(row[c])]
        if len(ps) == 0:
            mu_list.append(np.nan); var_list.append(np.nan); alea_list.append(np.nan); n_list.append(0); continue
        ps = np.clip(np.array(ps, dtype=float), 1e-6, 1-1e-6)
        mu = float(ps.mean()); var = float(ps.var(ddof=0)); alea = float(np.mean(ps*(1-ps)))
        mu_list.append(mu); var_list.append(var); alea_list.append(alea); n_list.append(len(ps))
    base["mu"] = mu_list; base["var"] = var_list; base["alea"] = alea_list; base["n_rep"] = n_list

    margin = np.abs(base["mu"] - 0.5) / 0.5
    unc = 0.7*base["var"] + 0.3*base["alea"]
    conf = margin * (1 - np.clip(unc/0.25, 0.0, 1.0))

    out = base[["case_id"]].copy()
    out["conf"] = conf.fillna(0.0)
    out["mu"]   = base["mu"].fillna(0.5)
    out["var"]  = base["var"].fillna(0.0)
    out["alea"] = base["alea"].fillna(0.0)
    out["n_rep"]= base["n_rep"].fillna(0).astype(int)
    return out

# ---------- 6) 最简列裁剪（显式剔除辅助列） ----------
def minimal_feature_view(df: pd.DataFrame) -> pd.DataFrame:
    df = std_cols(df)
    reserved = {"case_id","image_path","mask_path","label"}
    # 严格过滤：以 '_' 开头的技工列、以及明确列名
    drop_prefixes = ("diagnostics_","pre_","post_","src_","case_uid","series_uid","study_uid","_")
    drop_exact = {
        "gata6_prob","gata6_label","prob_pos","prediction","pred","score","model_prob","used_threshold",
        # 显式剔除的辅助/诊断列：
        "conf","mu","var","alea","n_rep","__ord_true","__ord_conf"
    }
    keep_cols = []
    for c in df.columns:
        if c in reserved:
            keep_cols.append(c); continue
        cl = c.lower()
        if c in drop_exact:
            continue
        if any(cl.startswith(p) for p in drop_prefixes):
            continue
        if is_numeric_series(df[c]):   # 仅保留纯数值 radiomics
            keep_cols.append(c)
    return df[keep_cols]

# ---------- 7) 构造“按置信度 Top-K%”的训练 CSV ----------
def build_train_tables_by_conf(
    base_train_feats_csv: Path,
    conf_table: pd.DataFrame,
    rep_pseudo_csv: Path,
    train95_csv: Path,
    keep_fracs: List[float],
    out_dir: Path,
    rep_name: str
) -> Dict[str,str]:
    """
    - 基于 conf_table 对 550 部分排序，选前 K%；
    - 用当前 repeat 的伪标签 CSV 提供这些样本的 label；
    - 与 95 真值直接 concat（不按 case_id 去重）；路径冲突时丢弃伪标签；
    - 裁成最简列后保存（显式剔除辅助列）。
    """
    out_map: Dict[str,str] = {}
    feats = std_cols(pd.read_csv(base_train_feats_csv, encoding="utf-8-sig", low_memory=False))
    feats["case_id"] = feats["case_id"].astype(str)

    # 95 真值映射
    tr95_map = std_cols(pd.read_csv(train95_csv, encoding="utf-8-sig", low_memory=False))[["case_id","image_path","mask_path","label"]]
    tr95_map["case_id"] = tr95_map["case_id"].astype(str)
    feats = feats.merge(tr95_map[["case_id","label"]], on="case_id", how="left", suffixes=("", "_true"))

    set95 = set(tr95_map["case_id"].tolist())

    # 置信度
    conf = conf_table.copy()
    conf["case_id"] = conf["case_id"].astype(str)
    conf_550 = conf[~conf["case_id"].isin(set95)].copy()
    if conf_550.empty:
        raise SystemExit("置信度表未覆盖 550 伪标签数据。")

    # 当前 repeat 的伪标签（提供 gata6_label）
    pseudo = std_cols(pd.read_csv(rep_pseudo_csv, encoding="utf-8-sig", low_memory=False))
    if "prob_pos" in pseudo.columns: pseudo = pseudo.rename(columns={"prob_pos":"gata6_prob"})
    pseudo["case_id"] = pseudo["case_id"].astype(str)
    if "gata6_label" not in pseudo.columns and "label" in pseudo.columns:
        pseudo = pseudo.rename(columns={"label":"gata6_label"})

    # 合并 conf 与 gata6_label
    feats = feats.merge(conf[["case_id","conf"]], on="case_id", how="left")
    feats = feats.merge(pseudo[["case_id","gata6_label"]], on="case_id", how="left")

    # 训练集真值部分
    df_true = feats[feats["case_id"].isin(set95)].copy()
    if "label" not in df_true.columns:
        raise SystemExit("95 真值子集未带入 label，请检查 train95.csv。")
    df_true = df_true[df_true["label"].notna()].copy()
    df_true["label"] = df_true["label"].astype(int)

    # 保真丢伪：路径冲突键
    true_key = set((df_true["image_path"].astype(str) + "|" + df_true["mask_path"].astype(str)).tolist())

    # 伪标签池
    df_pseu = feats[~feats["case_id"].isin(set95)].copy()
    df_pseu = df_pseu[df_pseu["conf"].notna()].copy()
    df_pseu["path_key"] = df_pseu["image_path"].astype(str) + "|" + df_pseu["mask_path"].astype(str)
    df_pseu = df_pseu[~df_pseu["path_key"].isin(true_key)].copy()
    df_pseu.drop(columns=["path_key"], inplace=True, errors="ignore")

    # 按 conf 排序
    df_pseu.sort_values(by="conf", ascending=False, inplace=True, kind="mergesort")
    n_all_p = len(df_pseu)
    print(f"[CONF] {rep_name}: 真值={len(df_true)}, 伪标签可用={n_all_p}（按 conf 排序）")

    # 真值顺序，用于排序优先
    order_true = tr95_map["case_id"].tolist()
    map_true = {cid:i for i,cid in enumerate(order_true)}

    for frac in keep_fracs:
        frac = float(frac); assert 0.0 < frac <= 1.0
        k = max(1, int(np.round(n_all_p * frac)))
        top_p = df_pseu.iloc[:k].copy()

        # 由当前 repeat 的伪标签给出 label
        top_p["label"] = top_p["gata6_label"]
        top_p = top_p[top_p["label"].notna()].copy()
        top_p["label"] = top_p["label"].astype(int)

        # 纵向拼接（不按 case_id 去重）
        merged = pd.concat([df_true, top_p], ignore_index=True)

        # 排序键（在裁剪前计算）
        merged["__ord_true"] = merged["case_id"].map(map_true).fillna(1e12)
        merged["__ord_conf"] = -merged["conf"].fillna(-1e9).astype(float)
        merged.sort_values(by=["__ord_true","__ord_conf"], inplace=True, kind="mergesort")
        merged.reset_index(drop=True, inplace=True)

        # 最简列裁剪（显式剔除辅助列）
        merged_min = minimal_feature_view(merged)
        drop_cols = [c for c in ["conf","mu","var","alea","n_rep","__ord_true","__ord_conf"] if c in merged_min.columns]
        if drop_cols:
            merged_min = merged_min.drop(columns=drop_cols)

        # 导出
        q_str = f"{frac:.2f}".rstrip("0").rstrip(".")
        out_csv = out_dir / f"feats_train_min_{rep_name}_q{q_str}.csv"
        merged_min.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] 训练 CSV（{rep_name}，Top-{int(frac*100)}% by conf）-> {out_csv} | 规模={merged_min.shape}")
        out_map[q_str] = str(out_csv)

    return out_map

# ---------- 8) 测试集最简 CSV ----------
def minimalize_test_csv(src_csv: Path, out_csv: Path) -> Path:
    df = pd.read_csv(src_csv, encoding="utf-8-sig", low_memory=False)
    dfm = minimal_feature_view(df)
    if "label" not in dfm.columns:
        raise SystemExit(f"测试集缺少 label 列：{src_csv}")
    # 双保险剔除
    drop_cols = [c for c in ["conf","mu","var","alea","n_rep","__ord_true","__ord_conf"] if c in dfm.columns]
    if drop_cols:
        dfm = dfm.drop(columns=drop_cols)
    dfm.to_csv(out_csv, index=False, encoding="utf-8-sig"); return out_csv

# ---------- 9) 传统 ML （不使用 prob） ----------
def run_radiomics_ml(train_feats: Path, test_feats: Path, outdir: Path, *, seed: int, cv: int, topk: int):
    ensure_dir(outdir)
    cmd = [sys.executable, ANALYSIS_SCRIPT,
           "--train-csv", str(train_feats), "--test-csv", str(test_feats),
           "--outdir", str(outdir),
           "--seed", str(seed), "--cv", str(cv), "--topk", str(topk),
           "--overlap-by", "image_path", "--on-overlap", "error",
           "--no-prob"]
    run(cmd)

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser("SSL GATA6 pipeline (ensemble confidence, top-fraction filtering, minimal CSVs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labeled-136-csv", required=True)
    ap.add_argument("--unlabeled-550-csv", required=True)
    ap.add_argument("--external-54-csv", required=True)
    ap.add_argument("--rad-config", required=True)
    # Teacher
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--teacher-val-split", type=float, default=0.1)
    
    # [新增] 预训练权重路径参数
    ap.add_argument("--pretrain-path", type=str, default=None, 
                    help="Path to MedicalNet .pth file (e.g. resnet18_23dataset.pth)")

    # --- Teacher 推理可选项（默认仍是概率阈值0.5，不改逻辑） ---
    ap.add_argument("--teacher-prob-thr", type=float, default=0.5,
                    help="Teacher 概率阈值（未指定 --teacher-use-youden 且未启用 logit 时生效，默认 0.5）")
    ap.add_argument("--teacher-use-youden", action="store_true", default=False,
                    help="从 metrics.csv 读取 Youden 概率阈值（与 --teacher-prob-thr 互斥，优先级更高）")
    ap.add_argument("--teacher-logit-q", type=float, default=None,
                    help="按 |logit| 的分位数筛高置信伪标（例如 0.7）；与 --teacher-logit-thr 互斥")
    ap.add_argument("--teacher-logit-thr", type=float, default=None,
                    help="按固定 |logit| 门限筛高置信伪标（例如 2.0）；与 --teacher-logit-q 互斥")
    ap.add_argument("--teacher-device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Teacher 推理设备（默认 auto：优先用 CUDA）")

    # Radiomics
    ap.add_argument("--bbox", dest="use_bbox", action="store_true", default=True)
    ap.add_argument("--no-bbox", dest="use_bbox", action="store_false")
    ap.add_argument("--bbox-pad", type=int, default=0)
    ap.add_argument("--bin-count", type=int, default=64)
    # ML
    ap.add_argument("--cv", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)
    # 伪标签保留前 K%
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out))
    teacher_root  = ensure_dir(out_root/"teacher")
    radiomics_root= ensure_dir(out_root/"radiomics")
    ml_root       = ensure_dir(out_root/"ml_eval")

    # 注意：argparse 会把横杠参数转为下划线属性
    labeled_136_csv   = Path(args.labeled_136_csv)
    unlabeled_550_csv = Path(args.unlabeled_550_csv)
    external_54_csv   = Path(args.external_54_csv)

    # 1) 95/41
    split_dir = ensure_dir(teacher_root/"split")
    train95_csv, int41_csv = split_136(labeled_136_csv, seed=args.seed, outdir=split_dir)

    # 2) teacher
    run_dir = train_teacher(
        train95_csv, teacher_root, 
        epochs=args.epochs,
        workers=args.workers, 
        batch_size=args.batch_size,
        seed=args.seed, 
        val_split=args.teacher_val_split,
        pretrain_path=args.pretrain_path # Pass the path down
    )
    reps = list_rep_dirs(run_dir)
    if not reps: raise SystemExit(f"[ERROR] 未找到 repeat：{run_dir} 下无 rep_*/last_model.pth")

    # 3) 伪标签（每个 repeat）
    pseudo_parent = ensure_dir(teacher_root/"pseudo550")
    rep_to_pseudo: Dict[str,str] = {}
    for rep_dir in reps:
        infer_dir = copy_last_as_best(rep_dir)
        outdir = ensure_dir(pseudo_parent/rep_dir.name)
        pseudo_csv = infer_pseudolabels(
            unlabeled_550_csv, infer_dir, outdir,
            workers=args.workers, batch_size=args.batch_size,
            force_thr=args.teacher_prob_thr,
            use_youden=args.teacher_use_youden,
            logit_q=args.teacher_logit_q,
            logit_thr=args.teacher_logit_thr,
            device=args.teacher_device,
        )
        rep_to_pseudo[rep_dir.name] = str(pseudo_csv)

    # 4) 基础特征 + 测试集
    rad_dir = radiomics_root
    pool_csv = make_train_pool_csv(train95_csv, unlabeled_550_csv, rad_dir)
    base_train_feats = rad_dir/"feats_train_pool_base.csv"
    if not base_train_feats.exists():
        extract_feats_base(pool_csv, Path(args.rad_config), base_train_feats,
                           use_bbox=args.use_bbox, bbox_pad=args.bbox_pad, bin_count=args.bin_count)
    else:
        print(f"[RAD] 复用基础特征：{base_train_feats}")

    int_raw = rad_dir/"feats_int41.csv"; int_min = rad_dir/"feats_int41_min.csv"
    if not int_min.exists():
        extract_feats_test(int41_csv, Path(args.rad_config), int_raw,
                           use_bbox=args.use_bbox, bbox_pad=args.bbox_pad, bin_count=args.bin_count)
        attach_labels(int_raw, int41_csv, rad_dir/"feats_int41_with_label.csv")
        minimalize_test_csv(rad_dir/"feats_int41_with_label.csv", int_min)

    ext_raw = rad_dir/"feats_ext54.csv"; ext_min = rad_dir/"feats_ext54_min.csv"
    if not ext_min.exists():
        extract_feats_test(external_54_csv, Path(args.rad_config), ext_raw,
                           use_bbox=args.use_bbox, bbox_pad=args.bbox_pad, bin_count=args.bin_count)
        attach_labels(ext_raw, external_54_csv, rad_dir/"feats_ext54_with_label.csv")
        minimalize_test_csv(rad_dir/"feats_ext54_with_label.csv", ext_min)

    mer_min = rad_dir/"feats_merged_min.csv"
    if not mer_min.exists():
        pd.concat([pd.read_csv(int_min, encoding="utf-8-sig", low_memory=False),
                   pd.read_csv(ext_min, encoding="utf-8-sig", low_memory=False)],
                  ignore_index=True).to_csv(mer_min, index=False, encoding="utf-8-sig")

    # 5) 计算“集成置信度”并构造各 repeat 的训练 CSV（按 Top-K% by conf）
    conf_table = compute_ensemble_confidence(rep_to_pseudo)
    manifest_q: Dict[str,Dict] = {}
    for rep_dir in reps:
        rep_name = rep_dir.name
        rep_train_dir = ensure_dir(radiomics_root/f"train_min_{rep_name}")
        train_csvs = build_train_tables_by_conf(
            base_train_feats, conf_table, Path(rep_to_pseudo[rep_name]),
            train95_csv, keep_fracs=args.keep_fracs,
            out_dir=rep_train_dir, rep_name=rep_name
        )
        # 评估（不使用 prob）
        rep_root = ensure_dir(ml_root/rep_name)
        rep_manifest: Dict[str,Dict] = {}
        for frac_str, train_csv_path in sorted(train_csvs.items()):
            q_root = ensure_dir(rep_root/f"quantile_{frac_str}")
            int_dir, ext_dir, mer_dir = ensure_dir(q_root/"int_res"), ensure_dir(q_root/"ext_res"), ensure_dir(q_root/"mer_res")
            run_radiomics_ml(Path(train_csv_path), int_min, int_dir, seed=args.seed, cv=args.cv, topk=args.topk)
            run_radiomics_ml(Path(train_csv_path), ext_min, ext_dir, seed=args.seed, cv=args.cv, topk=args.topk)
            run_radiomics_ml(Path(train_csv_path), mer_min, mer_dir, seed=args.seed, cv=args.cv, topk=args.topk)
            rep_manifest[frac_str] = {"int_res": str(int_dir), "ext_res": str(ext_dir), "mer_res": str(mer_dir)}
        manifest_q[rep_name] = rep_manifest

    # 6) 清单
    manifest = {
        "split": {"train95_csv": str(train95_csv), "int41_csv": str(int41_csv)},
        "teacher_run_dir": str(run_dir),
        "pseudo_by_rep": rep_to_pseudo,
        "confidence_table_rows": int(conf_table.shape[0]),
        "radiomics": {
            "base_train_feats": str(base_train_feats),
            "int_feats_min": str(int_min), "ext_feats_min": str(ext_min), "mer_feats_min": str(mer_min),
        },
        "ml_results": manifest_q
    }
    with open(out_root/"_pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n✅ 全部完成。评估结果位置：")
    print("  teacher:   ", teacher_root)
    print("  radiomics: ", radiomics_root)
    print("  ml_eval:   ", ml_root)

if __name__ == "__main__":
    main()