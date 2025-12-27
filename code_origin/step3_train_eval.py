#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_eval_shuffle5.py  (per-fold leak-safe preprocessing + checkpoints)
-----------------------------------------------------------------------
- 模型: from model.Transformer import TabPatchTransformer
- Repeated Stratified Shuffle Split: 随机打乱 N 次，train/val=1-test_size
- 训练: AdamW + Warmup(10%) + CosineAnnealingLR, 梯度裁剪, 早停(监控 val AUC)
- 预处理(每折内拟合): 缺失值中位数填充 -> Scaler(standard/robust) -> 可选 L1 选择 -> 可选 PCA
- 输出:
  1) 每折 ROC 图 + AUC 柱状图 + 训练/验证损失 & 验证AUC 曲线
  2) CSV: 每折 AUC 与各曲线
  3) Checkpoints: outdir/checkpoints/fold{K}_best.pt
     （内含: model_state, feat_cols, median, scaler信息, L1选择信息, PCA信息, model_args 等）
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve

# 仅在 L1 选择时用
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
import warnings

import matplotlib.pyplot as plt

# === 你的模型文件 ===
from model.Transformer import TabPatchTransformer


# ---------------------- Matplotlib 风格 ----------------------
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
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def nice_axes(ax, square=False):
    ax.grid(True)
    if square:
        ax.set_aspect('equal', adjustable='box')
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(1.2)


# ======================= Dataset =======================
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =================== Train / Eval utils ===================

def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps=1, max_grad_norm=1.0):
    model.train()
    loss_vals, step_count = [], 0
    optimizer.zero_grad()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y) / accum_steps
        loss.backward()
        step_count += 1
        if step_count % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        loss_vals.append(loss.item() * accum_steps)
    return float(np.mean(loss_vals))


@torch.no_grad()
def infer_logits(model, loader, device):
    model.eval()
    outs = []
    for X, _ in loader:
        X = X.to(device)
        outs.append(model(X).cpu().numpy())
    return np.concatenate(outs, axis=0)


# =================== Per-fold preprocessing helpers ===================

def build_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    elif kind == "standard":
        return StandardScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'robust'")

def fit_l1_selector(X_tr_scaled: np.ndarray, y_tr: np.ndarray, keep_top: int = 0, random_state: int = 2025):
    """
    用 L1-LogisticCV 找到稀疏系数，返回:
      - mask: bool array, 选中的列
      - info: dict, 包含 best_C / coef / selected_indices
    keep_top: 若 >0，在非零中按 |coef| 仅保留 Top-K
    """
    warnings.simplefilter("ignore", ConvergenceWarning)
    Cs = np.logspace(-6, 6, 2000)
    lrcv = LogisticRegressionCV(
        Cs=Cs, cv=10, penalty="l1", solver="liblinear",
        scoring="roc_auc", class_weight="balanced",
        max_iter=10000, n_jobs=-1, refit=True, random_state=random_state
    )
    lrcv.fit(X_tr_scaled, y_tr)
    coef = lrcv.coef_.ravel()
    nz = np.flatnonzero(coef != 0.0)
    if nz.size == 0:
        # 兜底：若无非零，按 |coef| 取前 16（或前 N/4）
        order = np.argsort(-np.abs(coef))
        k = keep_top if keep_top > 0 else min(16, max(1, X_tr_scaled.shape[1] // 4))
        sel = order[:k]
    else:
        if keep_top > 0 and nz.size > keep_top:
            order = nz[np.argsort(-np.abs(coef[nz]))[:keep_top]]
            sel = order
        else:
            sel = nz
    mask = np.zeros_like(coef, dtype=bool)
    mask[sel] = True
    info = {
        "best_C": float(lrcv.C_[0]),
        "coef": coef.tolist(),
        "selected_indices": sel.tolist()
    }
    return mask, info

def fit_pca(X_tr_scaled_sel: np.ndarray, n_components: int, random_state: int = 2025):
    pca = PCA(n_components=min(n_components, X_tr_scaled_sel.shape[1]), random_state=random_state)
    Z_tr = pca.fit_transform(X_tr_scaled_sel)
    return pca, Z_tr


# =================== Main Loop (Repeated Shuffle) ===================

def run_repeated_shuffle(args):
    set_fig_style()
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # 标签
    assert 'GATA6' in df.columns, "CSV 必须包含标签列 GATA6（0/1）"
    y_all = df['GATA6'].astype(int).values

    # 特征列
    drop_cols = {'case_id', 'image_path', 'mask_path', 'GATA6'}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("未找到组学特征列，请检查 CSV。")

    # 数值化
    X_df_all = df[feat_cols].apply(pd.to_numeric, errors='coerce')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = StratifiedShuffleSplit(
        n_splits=args.repeats, test_size=args.test_size, random_state=args.seed
    )

    fold_aucs, roc_curves = [], []
    train_losses_all, val_losses_all, val_aucs_all = [], [], []

    for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X_df_all, y_all), start=1):
        # 子种子
        torch.manual_seed(args.seed + fold_id)
        np.random.seed(args.seed + fold_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + fold_id)

        X_tr_df = X_df_all.iloc[tr_idx].copy()
        X_te_df = X_df_all.iloc[te_idx].copy()
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # ===== 缺失值以训练集统计量填充 =====
        med = X_tr_df.median(axis=0)
        X_tr_df = X_tr_df.fillna(med)
        X_te_df = X_te_df.fillna(med)

        # ===== Scaler (每折训练集上拟合) =====
        scaler = build_scaler(args.scaler)
        X_tr_scaled = scaler.fit_transform(X_tr_df.values.astype(np.float32))
        X_te_scaled = scaler.transform(X_te_df.values.astype(np.float32))

        # 用于记录的原始特征名（L1 选择前）
        current_feat_names = feat_cols[:]  # list copy
        l1_info_to_save = None
        pca_info_to_save = None

        # ===== 可选 L1 特征选择（在 scaled 上 fit）=====
        if args.preproc in ("l1", "l1_pca"):
            mask, l1info = fit_l1_selector(
                X_tr_scaled, y_tr, keep_top=max(0, args.l1_keep), random_state=args.seed + fold_id
            )
            # 应用选择
            X_tr_scaled = X_tr_scaled[:, mask]
            X_te_scaled = X_te_scaled[:, mask]
            # 记录被选中的列名
            selected_names = [name for i, name in enumerate(current_feat_names) if mask[i]]
            current_feat_names = selected_names
            l1_info_to_save = {
                "selected_names": selected_names,
                "selected_mask": mask.astype(int).tolist(),
                **l1info
            }

        # ===== 可选 PCA（在已选择/未选择的 scaled 上 fit）=====
        if args.preproc in ("pca", "l1_pca"):
            pca, Z_tr = fit_pca(X_tr_scaled, n_components=args.pca_dim, random_state=args.seed + fold_id)
            Z_te = pca.transform(X_te_scaled)
            X_tr_proc, X_te_proc = Z_tr, Z_te
            pca_info_to_save = {
                "n_components": int(pca.n_components_),
                "components": pca.components_.tolist(),
                "mean": pca.mean_.tolist(),
                "explained_variance": pca.explained_variance_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
            final_in_feats = X_tr_proc.shape[1]
            final_feat_names = [f"pc{i+1}" for i in range(final_in_feats)]
        else:
            X_tr_proc, X_te_proc = X_tr_scaled, X_te_scaled
            final_in_feats = X_tr_proc.shape[1]
            final_feat_names = current_feat_names

        # DataLoader
        train_ds, val_ds = TabularDataset(X_tr_proc, y_tr), TabularDataset(X_te_proc, y_te)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # 类别权重
        pos, neg = (y_tr == 1).sum(), (y_tr == 0).sum()
        pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)

        # 模型
        model = TabPatchTransformer(
            in_feats=final_in_feats,
            d_model=args.d_model,
            nhead=args.nhead,
            depth=args.depth,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            patch_size=args.patch_size,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc, best_state, patience = -1.0, None, 0
        train_losses, val_losses, val_aucs = [], [], []

        for epoch in range(1, args.epochs + 1):
            # Warmup
            warmup_epochs = max(1, int(0.1 * args.epochs))
            if epoch <= warmup_epochs:
                lr_scale = epoch / float(warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            # Train
            tr_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                accum_steps=args.accum_steps, max_grad_norm=1.0
            )

            # Validate
            logits = infer_logits(model, val_loader, device)
            probs = 1.0 / (1.0 + np.exp(-logits))
            val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.detach().cpu())
            val_loss = val_loss_fn(
                torch.from_numpy(logits).float(),
                torch.from_numpy(y_te.astype(np.float32))
            ).item()

            try:
                val_auc = roc_auc_score(y_te, probs)
            except ValueError:
                val_auc = float("nan")

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)

            # Early Stopping
            min_delta = 1e-3
            if val_auc > (best_auc + min_delta):
                best_auc = val_auc
                best_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"[Fold {fold_id}] Early stop at epoch {epoch}, best val AUC={best_auc:.4f}")
                    break

            scheduler.step()

            print(f"[Fold {fold_id}] Epoch {epoch:03d} | "
                  f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | valAUC={val_auc:.4f}")

        # ===== 保存本折的最佳权重 + 全部预处理信息（供推理复现） =====
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold_id}_best.pt")

        scaler_blob = {"type": args.scaler}
        if args.scaler == "standard":
            scaler_blob.update({
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "n_features_in": int(getattr(scaler, "n_features_in_", X_tr_df.shape[1])),
            })
        else:  # robust
            # RobustScaler 的属性为 center_ / scale_
            scaler_blob.update({
                "center": scaler.center_.tolist(),
                "scale": scaler.scale_.tolist(),
                "quantile_range": list(getattr(scaler, "quantile_range", (25.0, 75.0))),
                "with_centering": bool(scaler.with_centering),
                "with_scaling": bool(scaler.with_scaling),
                "n_features_in": int(getattr(scaler, "n_features_in_", X_tr_df.shape[1])),
            })

        torch.save({
            "model_state": best_state,
            "auc": float(best_auc),

            # 供复现对齐
            "feat_cols": feat_cols,           # 原始列顺序
            "median": med.to_dict(),          # 缺失值填充用

            # 预处理信息
            "scaler": scaler_blob,
            "l1_selector": l1_info_to_save,   # 可能为 None
            "pca": pca_info_to_save,          # 可能为 None
            "final_feat_names": final_feat_names,  # 经过 L1/PCA 后的特征名（PCA 用 pc1..）

            # 模型结构
            "model_args": {
                "in_feats": int(final_in_feats),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "depth": args.depth,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
                "patch_size": args.patch_size,
            },
            "train_meta": {
                "seed": args.seed + fold_id,
                "epochs_trained": len(train_losses),
                "pos_weight": float(neg / max(pos, 1)),
                "preproc": args.preproc,
                "pca_dim": args.pca_dim,
                "l1_keep": args.l1_keep,
                "scaler_kind": args.scaler,
            }
        }, ckpt_path)
        print(f"[Fold {fold_id}] Saved checkpoint -> {ckpt_path}")

        # ===== ROC 收集 =====
        model.load_state_dict(best_state)
        logits = infer_logits(model, val_loader, device)
        probs = 1 / (1 + np.exp(-logits))
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_curves.append((fpr, tpr))
        fold_aucs.append(best_auc)
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)
        val_aucs_all.append(val_aucs)

        print(f"[Fold {fold_id}] Best val AUC = {best_auc:.4f}")

    # ===== 汇总 & 作图 & CSV（与原逻辑一致） =====
    fold_aucs = np.array(fold_aucs, dtype=float)
    mean_auc = float(np.nanmean(fold_aucs))
    std_auc = float(np.nanstd(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
    try:
        from scipy.stats import t
        t_crit = t.ppf(0.975, df=len(fold_aucs) - 1)
    except Exception:
        t_crit = 2.776
    ci_low = max(0.0, mean_auc - t_crit * std_auc / max(np.sqrt(len(fold_aucs)), 1))
    ci_high = min(1.0, mean_auc + t_crit * std_auc / max(np.sqrt(len(fold_aucs)), 1))

    print(f"\n[Repeated Shuffle] AUCs over {len(fold_aucs)} folds: {np.round(fold_aucs, 4)}")
    print(f"[Summary] Mean AUC = {mean_auc:.4f} ± {std_auc:.4f} | 95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    # 每折 ROC
    fig = plt.figure(figsize=(6.6, 6.6))
    ax = fig.add_subplot(111)
    for i, (fpr, tpr) in enumerate(roc_curves, start=1):
        c = PALETTE[(i-1) % len(PALETTE)]
        ax.step(fpr, tpr, where="post", lw=2.0, alpha=0.95, color=c,
                label=f"Fold {i} (AUC={fold_aucs[i-1]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2, color="#888888")
    nice_axes(ax, square=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Repeated Stratified Shuffle (depth={args.depth})\n"
                 f"Mean AUC={mean_auc:.3f} ± {std_auc:.3f} | 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    ax.legend(loc="lower right", frameon=False, ncol=1)
    roc_path = os.path.join(args.outdir, "shuffle_roc_perfold.png")
    fig.tight_layout(); fig.savefig(roc_path, dpi=320); plt.close(fig)

    # AUC 柱状图
    fig = plt.figure(figsize=(7.0, 4.4))
    ax = fig.add_subplot(111)
    xs = np.arange(1, len(fold_aucs) + 1)
    bars = ax.bar(xs, fold_aucs, width=0.62, color="#A7C7E7", edgecolor="#2B547E", linewidth=1.2)
    for rect, val in zip(bars, fold_aucs):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="#333333")
    ax.hlines(mean_auc, 0.5, len(fold_aucs) + 0.5, colors="#C23B22",
              linestyles="--", linewidth=1.6, label=f"Mean={mean_auc:.3f}")
    ax.hlines([ci_low, ci_high], 0.5, len(fold_aucs) + 0.5,
              colors="#C23B22", linestyles=":", linewidth=1.4,
              label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    nice_axes(ax)
    ax.set_xticks(xs); ax.set_xticklabels([f"Fold {i}" for i in xs])
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("AUC across shuffled folds")
    auc_bar_path = os.path.join(args.outdir, "shuffle_auc_bar.png")
    fig.tight_layout(); fig.savefig(auc_bar_path, dpi=320); plt.close(fig)

    # 训练/验证曲线
    def save_curve_png(all_series, ylabel, title, fname, ylim=None):
        fig = plt.figure(figsize=(7.6, 4.8))
        ax = fig.add_subplot(111)
        for i, series in enumerate(all_series, start=1):
            c = PALETTE[(i-1) % len(PALETTE)]
            ax.plot(range(1, len(series)+1), series, lw=2.0, color=c, alpha=0.95, label=f"Fold {i}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(title)
        nice_axes(ax)
        loc = "upper right" if "Loss" in ylabel else "lower right"
        ax.legend(fontsize=9, loc=loc, frameon=False)
        path = os.path.join(args.outdir, fname)
        fig.tight_layout(); fig.savefig(path, dpi=320); plt.close(fig)
        return path

    tr_loss_png = save_curve_png(train_losses_all, "Training Loss (BCE)", "Training Loss per Fold",
                                 "train_loss_perfold.png", None)
    val_loss_png = save_curve_png(val_losses_all, "Validation Loss (BCE)", "Validation Loss per Fold",
                                  "val_loss_perfold.png", None)
    val_auc_png  = save_curve_png(val_aucs_all,  "Validation AUC", "Validation AUC per Fold",
                                  "val_auc_perfold.png", (0.0, 1.0))

    # CSV
    pd.DataFrame({"fold": [f"fold_{i}" for i in xs], "auc": fold_aucs}).to_csv(
        os.path.join(args.outdir, "shuffle_auc.csv"), index=False
    )
    def save_curve_csv(all_series, name):
        max_len = max(len(s) for s in all_series)
        df = pd.DataFrame({"epoch": np.arange(1, max_len+1)})
        for i, series in enumerate(all_series, start=1):
            arr = np.full((max_len,), np.nan, dtype=float); arr[:len(series)] = series
            df[f"fold{i}"] = arr
        df.to_csv(os.path.join(args.outdir, name), index=False)

    save_curve_csv(train_losses_all, "train_loss_perfold.csv")
    save_curve_csv(val_losses_all, "val_loss_perfold.csv")
    save_curve_csv(val_aucs_all,  "val_auc_perfold.csv")

    print(f"[OK] ROC图: {roc_path}")
    print(f"[OK] AUC柱状图: {auc_bar_path}")
    print(f"[OK] 训练损失图: {tr_loss_png}")
    print(f"[OK] 验证损失图: {val_loss_png}")
    print(f"[OK] 验证AUC图: {val_auc_png}")
    print(f"[OK] Checkpoints saved to: {ckpt_dir}")


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="包含 GATA6 标签与组学特征的 CSV")
    parser.add_argument("--outdir", type=str, required=True, help="输出目录")
    # 数据划分
    parser.add_argument("--repeats", type=int, default=10, help="随机打乱次数")
    parser.add_argument("--test_size", type=float, default=0.2, help="验证集比例")
    # 训练参数
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # 模型参数
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch_size", type=int, default=16)
    # 预处理选项（每折内拟合）
    parser.add_argument("--scaler", type=str, default="robust", choices=["standard", "robust"],
                        help="标准化器：standard=StandardScaler, robust=RobustScaler(默认)")
    parser.add_argument("--preproc", type=str, default="l1_pca", choices=["none", "pca", "l1", "l1_pca"],
                        help="是否启用 per-fold L1 选择与/或 PCA（默认 l1_pca）")
    parser.add_argument("--pca_dim", type=int, default=32, help="PCA 维度（启用 pca 或 l1_pca 时生效）")
    parser.add_argument("--l1_keep", type=int, default=0,
                        help="L1 选择保留的特征数；0 表示保留所有非零系数")
    args = parser.parse_args()

    run_repeated_shuffle(args)


if __name__ == "__main__":
    main()
