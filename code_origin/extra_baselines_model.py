#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extra_baselines_pseudo5fold.py
------------------------------------------------------------
名义 5-Fold，但本质是：全量数据按固定比例随机打乱划分 train/val，重复 5 次。
- StratifiedShuffleSplit(n_splits=FOLDS, test_size=...)
- 每次随机划分都训练与验证（共 FOLDS 次）
- 输出：每模型的 ROC（5条曲线）、AUC柱状图、结果CSV 与汇总表

数据处理：
- 忽略: case_id, image_path, mask_path, GATA6
- 其余列 -> 数值（无法转的强制为 NaN）
- 仅用训练划分的中位数填充；仅对 LR/SVM 做 StandardScaler

不平衡：
- LR/SVM: class_weight="balanced"
- RF/ET: class_weight="balanced_subsample"
- GBDT: 通过 sample_weight 注入类别权重
- XGBoost: scale_pos_weight = neg/pos
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# 可选：XGBoost（没有就跳过）
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None


# ---------- 画图风格 ----------
def set_fig_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })


PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def nice_axes(ax, square=False):
    ax.grid(True)
    if square:
        ax.set_aspect('equal', adjustable='box')
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#333")
        ax.spines[s].set_linewidth(1.2)


# ---------- 工具函数 ----------
def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def median_impute(train_df: pd.DataFrame, test_df: pd.DataFrame):
    med = train_df.median(axis=0)
    return train_df.fillna(med), test_df.fillna(med)


def ci95_from_samples(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    n = np.count_nonzero(~np.isnan(v))
    mean = float(np.nanmean(v))
    std = float(np.nanstd(v, ddof=1)) if n > 1 else float("nan")
    if n <= 1 or np.isnan(std):
        return mean, std, mean, mean
    try:
        from scipy.stats import t
        t_crit = t.ppf(0.975, df=n - 1)
    except Exception:
        t_crit = 2.776
    half = t_crit * std / np.sqrt(n)
    return mean, std, max(0.0, mean - half), min(1.0, mean + half)


def compute_class_weights(y_tr: np.ndarray):
    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    n = len(y_tr)
    w0 = n / (2.0 * max(neg, 1))
    w1 = n / (2.0 * max(pos, 1))
    weights = {0: w0, 1: w1}
    sample_weight = np.where(y_tr == 1, w1, w0).astype(np.float32)
    return weights, sample_weight


def get_models() -> Dict[str, dict]:
    """
    模型及参数（为小样本&不稳定性做的“保守初试参数”）
    """
    models = {
        "LogReg": {
            "factory": lambda cw: LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=cw
            ),
            "need_scaler": True,
            "use_sample_weight": False
        },
        "SVM_RBF": {
            "factory": lambda cw: SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight=cw
            ),
            "need_scaler": True,
            "use_sample_weight": False
        },
        "RandomForest": {
            "factory": lambda cw: RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "need_scaler": False,
            "use_sample_weight": False
        },
        "ExtraTrees": {
            "factory": lambda cw: ExtraTreesClassifier(
                n_estimators=600,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "need_scaler": False,
            "use_sample_weight": False
        },
        "GradientBoosting": {
            "factory": lambda cw: GradientBoostingClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                max_features=None,
                random_state=42,
            ),
            "need_scaler": False,
            "use_sample_weight": True  # 用 sample_weight 注入不平衡
        },
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "factory": lambda spw: XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=600,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=1.0,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                reg_alpha=0.0,
                gamma=0.0,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                scale_pos_weight=spw
            ),
            "need_scaler": False,
            "use_sample_weight": False
        }
    else:
        print("[WARN] 未检测到 xgboost，将跳过 XGBoost。可用 `pip install xgboost` 安装。")

    return models


def fit_predict_auc(model_name: str, model_obj, X_tr, y_tr, X_te, y_te,
                    need_scaler: bool, use_sample_weight: bool):
    # 缩放（仅 LR/SVM）
    if need_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    # 类别不平衡
    class_w, sample_w = compute_class_weights(y_tr)

    # 拟合
    if use_sample_weight:
        model_obj.fit(X_tr, y_tr, sample_weight=sample_w)
    else:
        try:
            model_obj.fit(X_tr, y_tr)
        except TypeError:
            model_obj.fit(X_tr, y_tr, sample_weight=sample_w)

    # 概率
    if hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(X_te)[:, 1]
    else:
        if hasattr(model_obj, "decision_function"):
            scores = model_obj.decision_function(X_te)
            from sklearn.preprocessing import MinMaxScaler
            probs = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
        else:
            preds = model_obj.predict(X_te)
            probs = preds.astype(float)

    auc = roc_auc_score(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    return auc, (fpr, tpr)


def evaluate_pseudo5fold(model_name: str, cfg: dict, X: np.ndarray, y: np.ndarray,
                         outdir: str, folds: int = 5, test_size: float = 0.2, seed: int = 2025):
    """
    伪 5-Fold：用 StratifiedShuffleSplit 把全量数据随机划分 (train/val)，重复 `folds` 次。
    每次随机划分都视为一个“fold”。总计 `folds` 个 AUC。
    """
    os.makedirs(outdir, exist_ok=True)
    splitter = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, random_state=seed)

    aucs, rocs = [], []
    for i, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 缺失填充（用训练划分统计）
        X_tr_df, X_te_df = pd.DataFrame(X_tr), pd.DataFrame(X_te)
        X_tr_df, X_te_df = median_impute(X_tr_df, X_te_df)
        X_tr, X_te = X_tr_df.values, X_te_df.values

        # 模型
        if model_name in ["LogReg", "SVM_RBF"]:
            model = cfg["factory"]("balanced")
        elif model_name == "XGBoost":
            pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
            spw = (neg / max(pos, 1)) if pos > 0 else 1.0
            model = cfg["factory"](spw)
        else:
            model = cfg["factory"](None)

        auc, roc = fit_predict_auc(model_name, model, X_tr, y_tr, X_te, y_te,
                                   cfg["need_scaler"], cfg["use_sample_weight"])
        aucs.append(auc); rocs.append(roc)
        print(f"[Pseudo5Fold-{model_name}] Fold {i}: AUC={auc:.4f}")

    # 汇总
    aucs = np.array(aucs, dtype=float)
    mean, std, lo, hi = ci95_from_samples(aucs)
    xs = np.arange(1, folds + 1)

    # ROC
    plt.figure(figsize=(6.4, 6.4))
    for i, (fpr, tpr) in enumerate(rocs, 1):
        c = PALETTE[(i-1) % len(PALETTE)]
        plt.step(fpr, tpr, where="post", color=c, lw=2.0, alpha=0.95,
                 label=f"Fold {i} (AUC={aucs[i-1]:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1.2, color="#888")
    nice_axes(plt.gca(), square=True)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — Pseudo 5-Fold (shuffle x{folds})\n"
              f"Mean AUC={mean:.3f} ± {std:.3f} | 95% CI [{lo:.3f},{hi:.3f}]")
    plt.legend(loc="lower right", frameon=False)
    roc_path = os.path.join(outdir, f"{model_name}_pseudo5fold_roc.png")
    plt.tight_layout(); plt.savefig(roc_path); plt.close()

    # AUC 条形图
    plt.figure(figsize=(7.0, 4.2))
    bars = plt.bar(xs, aucs, width=0.6, color="#A7C7E7", edgecolor="#2B547E", linewidth=1.2)
    for r, v in zip(bars, aucs):
        plt.text(r.get_x()+r.get_width()/2, r.get_height()+0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)
    plt.hlines([mean, lo, hi], 0.5, folds+0.5, colors=["#C23B22", "#C23B22", "#C23B22"],
               linestyles=["--", ":", ":"], linewidths=[1.6, 1.4, 1.4],
               label=f"Mean {mean:.3f}, 95% CI [{lo:.3f},{hi:.3f}]")
    nice_axes(plt.gca())
    plt.ylim(0, 1.0); plt.ylabel("AUC"); plt.xticks(xs, [f"Fold {i}" for i in xs])
    plt.title(f"{model_name} — Fold AUC (Pseudo 5-Fold)")
    plt.legend(loc="lower right", frameon=False)
    bar_path = os.path.join(outdir, f"{model_name}_pseudo5fold_auc_bar.png")
    plt.tight_layout(); plt.savefig(bar_path); plt.close()

    # 保存 CSV
    pd.DataFrame({
        "fold": [f"fold_{i}" for i in xs],
        "auc": aucs,
        "mean_auc": mean, "std_auc": std, "ci_low": lo, "ci_high": hi
    }).to_csv(os.path.join(outdir, f"{model_name}_pseudo5fold_auc.csv"), index=False)

    print(f"[Pseudo5Fold-{model_name}] mean={mean:.4f} ± {std:.4f} | CI [{lo:.4f},{hi:.4f}]")
    print(f"[OK] ROC: {roc_path}")
    print(f"[OK] AUC bars: {bar_path}")
    print(f"[OK] CSV : {os.path.join(outdir, f'{model_name}_pseudo5fold_auc.csv')}")


# ---------- 主流程 ----------
def main():
    set_fig_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含 GATA6 标签与组学特征的 CSV")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--folds", type=int, default=10, help="打乱次数（当作fold数）")
    ap.add_argument("--test_size", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读入数据
    df = pd.read_csv(args.csv)
    assert "GATA6" in df.columns, "CSV 必须包含标签列 GATA6（0/1）"
    drop_cols = {"case_id", "image_path", "mask_path", "GATA6"}

    feat_cols = [c for c in df.columns if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("未找到组学特征列。")

    X_df = numeric_df(df[feat_cols])
    y = df["GATA6"].astype(int).values
    X = X_df.values

    models = get_models()

    # 对每个模型进行 pseudo-5fold 评估
    rows = []
    for name, cfg in models.items():
        print(f"\n===== {name}: Pseudo 5-Fold (shuffle x{args.folds}) =====")
        subdir = os.path.join(args.outdir, name)
        evaluate_pseudo5fold(name, cfg, X, y, outdir=subdir,
                             folds=args.folds, test_size=args.test_size, seed=args.seed)

        # 收集汇总
        csv_p = os.path.join(subdir, f"{name}_pseudo5fold_auc.csv")
        if os.path.exists(csv_p):
            d = pd.read_csv(csv_p)
            rows.append({
                "model": name,
                "mean_auc": d["mean_auc"].iloc[0],
                "std_auc": d["std_auc"].iloc[0],
                "ci_low": d["ci_low"].iloc[0],
                "ci_high": d["ci_high"].iloc[0],
            })

    # 总表
    if rows:
        summary = pd.DataFrame(rows).sort_values(["mean_auc", "std_auc"], ascending=[False, True])
        summary.to_csv(os.path.join(args.outdir, "baselines_pseudo5fold_summary.csv"), index=False)
        print("\n===== Pseudo 5-Fold Summary (sorted by mean_auc desc, std asc) =====")
        print(summary.to_string(index=False))
        print(f"\n[OK] baselines_pseudo5fold_summary.csv saved to {args.outdir}")
    else:
        print("[WARN] 没有生成任何结果文件，检查上方报错。")


if __name__ == "__main__":
    main()
