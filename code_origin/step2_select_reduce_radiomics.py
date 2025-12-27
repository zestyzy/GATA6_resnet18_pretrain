# select_reduce_for_table2.py
# -------------------------------------------------------------
# 适配“表2”：忽略 case_id/image_path/mask_path，标签用 GATA6
# 流程：正态性检验 -> t/U检验(p<0.05) -> 标准化 -> L1-LogisticCV 选特征
#      -> 画特征重要性与系数路径 -> PCA(20/30/40/50) 导出
# -------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA

# ========= 配置 =========
csv_path = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/ssl_136_futures_fixed.csv"
out_dir  = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/prep_table2"
os.makedirs(out_dir, exist_ok=True)

RANDOM_STATE = 2025
P_THRESH = 0.05
CS = np.logspace(-6, 6, 2000)            # L1-Logistic 的 C 网格
PCA_DIMS = [10, 20, 30, 40, 50]

# ========= 读取并确定列 =========
df = pd.read_csv(csv_path)
assert "GATA6" in df.columns, "CSV 必须包含 GATA6 标签列（0/1）"

drop_cols = {"case_id", "image_path", "mask_path", "GATA6"}
id_col = "case_id" if "case_id" in df.columns else None
label = df["GATA6"].astype(int).values

# 组学特征列（其余全部）
feat_cols = [c for c in df.columns if c not in drop_cols]
if len(feat_cols) == 0:
    raise ValueError("未找到组学特征列，请检查表2。")

# 仅保留可转为数值的列
X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
# 用中位数填充缺失
med = X_df.median(axis=0)
X_df = X_df.fillna(med)

# ========= 正态性检验 + t/U 检验 =========
normal_features, non_normal_features = [], []
for col in X_df.columns:
    vals = X_df[col].dropna()
    if vals.nunique() <= 1:
        continue
    try:
        stat, p = shapiro(vals)
        (normal_features if p > 0.05 else non_normal_features).append(col)
    except Exception:
        # 样本过少或异常，直接按非正态处理
        non_normal_features.append(col)

sig_t, sig_u = [], []
grp1 = df["GATA6"] == 1
grp0 = df["GATA6"] == 0
for col in X_df.columns:
    a = X_df.loc[grp1, col].dropna()
    b = X_df.loc[grp0, col].dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    if col in normal_features:
        _, p = ttest_ind(a, b, equal_var=False)
        if p < P_THRESH:
            sig_t.append(col)
    else:
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            if p < P_THRESH:
                sig_u.append(col)
        except ValueError:
            # 两组全相等等极端情况
            pass

significant_features = sorted(set(sig_t + sig_u))
if len(significant_features) == 0:
    # 如果单变量检验没有通过的，就使用全部特征进入下一步，但给出提示
    significant_features = list(X_df.columns)
    print("[WARN] 单变量检验无显著特征，将使用全部特征进入 L1-Logistic 选择。")

X_sig = X_df[significant_features].values

# ========= 标准化 =========
scaler = StandardScaler()
X_std = scaler.fit_transform(X_sig)

# ========= L1-Logistic CV 特征选择（分类更合理） =========
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    logi_cv = LogisticRegressionCV(
        Cs=CS, cv=10, penalty="l1", solver="liblinear",
        scoring="roc_auc", class_weight="balanced",
        max_iter=10000, n_jobs=-1, refit=True, random_state=RANDOM_STATE
    )
    logi_cv.fit(X_std, label)

best_C = float(logi_cv.C_[0])
coefs = logi_cv.coef_.ravel()
mask_nonzero = coefs != 0
selected_names = np.array(significant_features)[mask_nonzero].tolist()

print(f"[INFO] L1-Logistic 选中特征数：{len(selected_names)}（C={best_C:.4g}）")

# ========= 保存选中特征（未降维） =========
sel_df_out = pd.DataFrame(X_df[selected_names], columns=selected_names)
out_sel = pd.DataFrame(X_df[selected_names])
if id_col:
    out_sel.insert(0, id_col, df[id_col].values)
out_sel["GATA6"] = label
out_sel.to_csv(os.path.join(out_dir, "selected_features_l1logistic.csv"), index=False)

# ========= 特征重要性条形图 =========
imp_df = pd.DataFrame({"Feature": selected_names, "Importance": np.abs(coefs[mask_nonzero])})
imp_df = imp_df.sort_values("Importance", ascending=False)
plt.figure(figsize=(8, max(4, 0.25*len(imp_df))))
sns.barplot(data=imp_df, x="Importance", y="Feature")
plt.title("Feature importance (L1-Logistic)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "coef_importance.png"), dpi=180)
plt.close()

# ========= “系数路径”图（不同 C 下 Logistic 系数变化） =========
coef_paths = []
Cs_sorted = np.sort(CS)
for C in Cs_sorted:
    lr = LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
    )
    lr.fit(X_std, label)
    coef_paths.append(lr.coef_.ravel())
coef_paths = np.stack(coef_paths, axis=0)  # [nC, n_feat]
plt.figure(figsize=(9, 6))
for j in range(coef_paths.shape[1]):
    plt.plot(Cs_sorted, coef_paths[:, j], linewidth=1)
plt.axvline(best_C, color="r", linestyle="--", linewidth=1.5, label=f"Best C={best_C:.3g}")
plt.xscale("log")
plt.xlabel("C (L1-Logistic)")
plt.ylabel("Coefficient")
plt.title("Coefficient paths across C (L1-Logistic)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "coef_paths.png"), dpi=180)
plt.close()

# ========= PCA 降维（20/30/40/50）并保存 =========
# 对“选中特征矩阵”重新标准化后做 PCA（更稳妥）
X_sel = X_df[selected_names].values if len(selected_names) > 0 else X_sig
scaler_pca = StandardScaler()
X_sel_std = scaler_pca.fit_transform(X_sel)

for k in PCA_DIMS:
    k_use = min(k, X_sel_std.shape[1]) if X_sel_std.shape[1] > 0 else 0
    if k_use == 0:
        print(f"[WARN] 没有可用的选中特征，跳过 PCA{k}.")
        continue
    pca = PCA(n_components=k_use, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_sel_std)
    out_pca = pd.DataFrame(Z, columns=[f"pca_{i+1}" for i in range(k_use)])
    if id_col:
        out_pca.insert(0, id_col, df[id_col].values)
    out_pca["GATA6"] = label
    out_csv = os.path.join(out_dir, f"pca_{k_use}_from_l1selected.csv")
    out_pca.to_csv(out_csv, index=False)
    print(f"[OK] 保存：{out_csv}  (shape={out_pca.shape})")

print(f"[OK] 选中特征CSV: {os.path.join(out_dir, 'selected_features_l1logistic.csv')}")
print(f"[OK] 重要性图: {os.path.join(out_dir, 'coef_importance.png')}")
print(f"[OK] 系数路径图: {os.path.join(out_dir, 'coef_paths.png')}")
