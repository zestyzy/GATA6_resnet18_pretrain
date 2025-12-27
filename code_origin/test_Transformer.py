# cv_gata6_transformer_depth3.py
# ------------------------------------------------------------
# Stratified 5-Fold 交叉验证，固定 Transformer depth=3
# - 忽略 case_id / image_path / mask_path / GATA6 四列
# - 标签为 GATA6（二分类）
# - 训练采用 AdamW + Warmup + CosineAnnealingLR
# - 加入梯度裁剪，Early Stopping
# - 输出：
#     1) 每折 ROC（标题显示 per-fold AUC 均值±SD与95%CI）
#     2) AUC 柱状图
#     3) 训练损失曲线（每折一条）+ CSV
# ------------------------------------------------------------

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# ======================= Dataset =======================

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =================== Tab-Patch Transformer ===================

class PatchEmbed(nn.Module):
    """把连续特征向量按 patch_size 分块，并线性投影到 d_model 维度"""
    def __init__(self, patch_size: int, in_feats: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        n_patches = math.ceil(in_feats / patch_size)
        self.n_patches = n_patches
        self.pad_feats = n_patches * patch_size - in_feats
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):  # x: [B, F]
        B, F = x.shape
        if self.pad_feats > 0:
            pad = x.new_zeros((B, self.pad_feats))
            x = torch.cat([x, pad], dim=1)
        x = x.view(B, self.n_patches, -1)
        x = self.proj(x)
        return x


class TabPatchTransformer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        num_classes: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        depth: int = 3,              # ★ 固定为 3 层
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        patch_size: int = 8,
    ):
        super().__init__()
        self.embed = PatchEmbed(patch_size, in_feats, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.embed.n_patches, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        tok = self.cls_token.expand(B, -1, -1)
        x = self.embed(x)
        x = torch.cat([tok, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])   # [CLS]
        logits = self.head(x).squeeze(1)
        return logits


# =================== Training / Eval utils ===================

def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    loss_vals = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss_vals.append(loss.item())
    return float(np.mean(loss_vals))


@torch.no_grad()
def infer_logits(model, loader, device):
    model.eval()
    outs = []
    for X, _ in loader:
        X = X.to(device)
        outs.append(model(X).cpu().numpy())
    return np.concatenate(outs, axis=0)


# =================== CV Loop ===================

def run_cv(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # 标签
    assert 'GATA6' in df.columns, "CSV 必须包含标签列 GATA6（0/1）"
    y_all = df['GATA6'].astype(int).values

    # —— 选特征：忽略四列，其余全用 ——
    drop_cols = {'case_id', 'image_path', 'mask_path', 'GATA6'}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("未找到组学特征列，请检查 CSV（表2）")

    # 数值化
    X_df = df[feat_cols].apply(pd.to_numeric, errors='coerce')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_aucs, roc_curves = [], []
    fold_train_losses = []             # ★ 新增：记录每折训练损失曲线

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_df, y_all), start=1):
        X_tr_df, X_te_df = X_df.iloc[tr_idx].copy(), X_df.iloc[te_idx].copy()
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # 缺失值填充
        medians = X_tr_df.median(axis=0)
        X_tr_df = X_tr_df.fillna(medians)
        X_te_df = X_te_df.fillna(medians)

        # 标准化
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df.values.astype(np.float32))
        X_te = scaler.transform(X_te_df.values.astype(np.float32))

        # Dataloader
        train_ds = TabularDataset(X_tr, y_tr)
        test_ds  = TabularDataset(X_te, y_te)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # 类别权重
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)

        # 模型
        model = TabPatchTransformer(
            in_feats=X_tr.shape[1],
            d_model=args.d_model,
            nhead=args.nhead,
            depth=3,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            patch_size=args.patch_size,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc, best_state, patience = -1.0, None, 0
        this_fold_losses = []          # ★ 新增：临时存放当前折的每 epoch 损失

        for epoch in range(1, args.epochs + 1):
            # Warmup (前10%轮数线性升温)
            warmup_epochs = max(1, int(0.1 * args.epochs))
            if epoch <= warmup_epochs:
                lr_scale = epoch / float(warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            this_fold_losses.append(tr_loss)     # ★ 记录训练损失

            # 验证
            logits = infer_logits(model, test_loader, device)
            probs = 1 / (1 + np.exp(-logits))
            try:
                fold_auc = roc_auc_score(y_te, probs)
            except ValueError:
                fold_auc = float("nan")

            print(f"[Fold {fold}] Epoch {epoch:03d} | loss={tr_loss:.4f} | valAUC={fold_auc:.4f}")

            if fold_auc > best_auc:
                best_auc = fold_auc
                best_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"[Fold {fold}] Early stop at epoch {epoch}, best AUC={best_auc:.4f}")
                    break

            scheduler.step()

        # 用最佳参数重新评估并保存 ROC
        model.load_state_dict(best_state)
        logits = infer_logits(model, test_loader, device)
        probs = 1 / (1 + np.exp(-logits))
        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_curves.append((fpr, tpr))
        fold_aucs.append(best_auc)
        fold_train_losses.append(this_fold_losses)   # ★ 保存当前折的损失序列
        print(f"[Fold {fold}] Best AUC = {best_auc:.4f}")

    # ===== 汇总（per-fold AUC 统计） =====
    fold_aucs = np.array(fold_aucs, dtype=float)
    mean_auc = float(np.nanmean(fold_aucs))
    std_auc = float(np.nanstd(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
    # 95% CI（t 分布）
    try:
        from scipy.stats import t
        t_crit = t.ppf(0.975, df=len(fold_aucs) - 1)
    except Exception:
        t_crit = 2.776  # df=4 的近似值
    ci_low = max(0.0, mean_auc - t_crit * std_auc / max(np.sqrt(len(fold_aucs)), 1))
    ci_high = min(1.0, mean_auc + t_crit * std_auc / max(np.sqrt(len(fold_aucs)), 1))

    print(f"\n[CV] 5-Fold AUCs: {np.round(fold_aucs, 4)}")
    print(f"[CV] Mean AUC (per-fold) = {mean_auc:.4f} ± {std_auc:.4f}  |  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    # ===== 图1：每折 ROC =====
    plt.figure(figsize=(6.4, 6.4))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]
    for i, ((fpr, tpr), c) in enumerate(zip(roc_curves, colors), start=1):
        plt.step(fpr, tpr, where="post", lw=1.8, alpha=0.9, color=c,
                 label=f"Fold {i} (AUC={fold_aucs[i-1]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"5-Fold ROC (depth=3)\nMean AUC={mean_auc:.3f} ± {std_auc:.3f} | 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    plt.legend(loc="lower right", fontsize=8)
    roc_path = os.path.join(args.outdir, "cv_roc_depth3_perfold.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # ===== 图2：AUC 柱状图 =====
    plt.figure(figsize=(6.5, 4.2))
    xs = np.arange(1, len(fold_aucs) + 1)
    plt.bar(xs, fold_aucs, width=0.6, color="#9ecae1", edgecolor="#2f5597")
    plt.hlines(mean_auc, 0.5, len(fold_aucs) + 0.5, colors="#d62728", linestyles="--",
               label=f"Mean={mean_auc:.3f}")
    plt.hlines([ci_low, ci_high], 0.5, len(fold_aucs) + 0.5,
               colors="#d62728", linestyles=":", label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    plt.xticks(xs, [f"Fold {i}" for i in xs])
    plt.ylabel("AUC")
    plt.ylim(0, 1.0)
    plt.title("Fold-wise AUC (depth=3)")
    plt.legend(fontsize=8, loc="lower right")
    auc_bar_path = os.path.join(args.outdir, "cv_auc_bar_depth3.png")
    plt.tight_layout()
    plt.savefig(auc_bar_path, dpi=300)
    plt.close()

    # ===== 图3：训练损失曲线（每折） =====
    plt.figure(figsize=(7.2, 4.6))
    max_len = max(len(l) for l in fold_train_losses)
    for i, (losses, c) in enumerate(zip(fold_train_losses, colors), start=1):
        plt.plot(range(1, len(losses)+1), losses, marker="", lw=1.8, color=c, alpha=0.95,
                 label=f"Fold {i} (epochs={len(losses)})")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (BCE)")
    plt.title("Training Loss per Fold")
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(fontsize=8, loc="upper right")
    loss_png = os.path.join(args.outdir, "train_loss_depth3_perfold.png")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=300)
    plt.close()

    # 保存训练损失 CSV（按列为各折，行是 epoch；不足处填 NaN）
    df_loss = pd.DataFrame({"epoch": np.arange(1, max_len+1)})
    for i, losses in enumerate(fold_train_losses, start=1):
        col = f"fold{i}"
        arr = np.full((max_len,), np.nan, dtype=float)
        arr[:len(losses)] = losses
        df_loss[col] = arr
    loss_csv = os.path.join(args.outdir, "train_loss_depth3_perfold.csv")
    df_loss.to_csv(loss_csv, index=False)

    # ===== 保存数值（AUC） =====
    pd.DataFrame({
        "fold": [f"fold_{i}" for i in xs],
        "auc": fold_aucs
    }).to_csv(os.path.join(args.outdir, "cv_auc_depth3.csv"), index=False)

    print(f"[OK] ROC图(每折): {roc_path}")
    print(f"[OK] AUC柱状图: {auc_bar_path}")
    print(f"[OK] 训练损失图: {loss_png}")
    print(f"[OK] 训练损失CSV: {loss_csv}")
    print(f"[OK] AUC明细: {os.path.join(args.outdir, 'cv_auc_depth3.csv')}")


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_cv(args)


if __name__ == "__main__":
    main()
