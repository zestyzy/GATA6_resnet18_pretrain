# -*- coding: utf-8 -*-
"""
radiomics_train_test_topk.py  (drop-in replacement)

- 支持：单CSV分层切分 或 显式 train/test CSV
- 仅在训练集拟合：中位数填充 -> 常量/高相关剔除 -> 标准化 -> (可选) PCA -> 稳定RFE Top-k
- 训练集做 GridSearchCV 调参；测试集仅同步变换后评估
- 概率/分数列安全处理：
    * --no-prob 时：排除所有“像模型输出”的列（不会误删 radiomics 的 MaximumProbability 等）
    * --prob-col 指定时：仅保留该列用于训练集加权/过滤，其它“像模型输出”的列仍排除
- 伪标签置信度（新增）：
    * --weight-mode 现支持 {soft, hard, hybrid, quantile, ranksoft}
    * --conf-quantile ∈ (0,1)：按 |p-0.5| 的分位数阈值；如 0.7 表示保留前 30% 自信样本
- 重叠检测：
    * --overlap-by {case_id, image_path, image+mask}（默认 image_path）
    * --on-overlap {error, drop-from-test, drop-from-train, drop-both, ignore}
- 输出：Top-k特征、RFE选择频率、测试集各模型指标、ROC合图、最佳参数、预处理摘要、诊断信息
"""

import os, argparse, json, warnings, time, re
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, roc_curve
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------- 工具：阶段打印 -----------------
def stage(title: str):
    print("\n" + "="*74)
    print(title)
    print("="*74)

# ----------------- 标签/概率列探测 -----------------
def auto_label(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"--label-col 指定列不存在：{user_col}")
        return user_col
    for c in ["GATA6", "gata6_label", "label"]:
        if c in df.columns:
            return c
    raise ValueError("未找到标签列，请用 --label-col 指定或包含 GATA6/gata6_label/label 之一。")

def auto_prob(df: pd.DataFrame, user_col: Optional[str], no_prob: bool=False) -> Optional[str]:
    """支持 --no-prob、空串、none/null/off/disable 显式禁用"""
    if no_prob:
        return None
    if user_col is not None:
        s = str(user_col).strip().lower()
        if s in {"", "none", "null", "off", "disable"}:
            return None
        if user_col not in df.columns:
            raise ValueError(f"--prob-col 指定列不存在：{user_col}")
        return user_col
    for c in ["gata6_prob", "prob_pos", "prob", "score", "pred_prob", "prediction"]:
        if c in df.columns:
            return c
    return None

def to_binary_label(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float).values >= 0.5).astype(int)
    m = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,
         "pos":1,"neg":0,"positive":1,"negative":0,"gata6":1,"nongata6":0}
    out = [m.get(str(v).lower(), None) for v in series.values]
    if any(v is None for v in out):
        raise ValueError("标签列包含无法映射为0/1的取值，请先清洗。")
    return np.asarray(out, dtype=int)

# ----------------- 概率/分数列识别（更安全，不误杀 radiomics 特征） -----------------
# radiomics 家族关键词（检测到这些就视为“特征列”，不按模型概率处理）
_RADIO_FAMILIES = (
    "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape",
    "wavelet-", "log-sigma", "square", "squareroot", "exponential",
    "logarithm", "gradient", "lbp", "original"
)
# 明确判定为“模型输出”的常见别名
_PROB_STRICT_ALIASES = {
    "gata6_prob","prob_pos","prob","pred_prob","prediction","pred","model_prob",
    "score","pred_score","predscore","probability"
}
# token 级匹配（前后界为开头/结尾或下划线/中划线），避免命中 MaximumProbability
_TOKEN_PATTERNS = [
    re.compile(r'(^|[_\-])prob($|[_\-])', re.I),
    re.compile(r'(^|[_\-])score($|[_\-])', re.I),
    re.compile(r'(^|[_\-])pred($|[_\-])', re.I),
    re.compile(r'(^|[_\-])prediction($|[_\-])', re.I),
]

def _looks_like_model_output(colname: str) -> bool:
    c = colname.strip().lower()
    # radiomics 家族直接豁免
    if any(tok in c for tok in _RADIO_FAMILIES):
        return False
    # 严格别名
    if c in _PROB_STRICT_ALIASES:
        return True
    # token 级匹配
    return any(p.search(c) for p in _TOKEN_PATTERNS)

def find_prob_like_cols(columns: List[str]) -> List[str]:
    hits = []
    for c in columns:
        if _looks_like_model_output(c):
            hits.append(c)
    # 去重保序
    seen = set(); out = []
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
    return out

# ----------------- 预处理（仅在训练集拟合） -----------------
def variance_corr_filter(X_df: pd.DataFrame, corr_th: float=0.95) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # 常量列（或唯一值列）移除
    nunq = X_df.nunique()
    keep1 = nunq[nunq > 1].index.tolist()
    X1 = X_df[keep1]
    # 高相关去重：保留先出现的一个
    corr = X1.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_th)]
    X2 = X1.drop(columns=to_drop)
    return X2, keep1, to_drop

def fit_preprocess(X_tr_df: pd.DataFrame, corr_th: float, use_pca: bool,
                   pca_mode: str, pca_var: float, pca_n: int, pca_whiten: bool,
                   seed: int):
    t0 = time.time()
    # 1) 中位数填充
    imputer = SimpleImputer(strategy="median")
    X1 = pd.DataFrame(imputer.fit_transform(X_tr_df), columns=X_tr_df.columns, index=X_tr_df.index)
    print(f"[INFO] 中位数填充完成，训练集维度：{X1.shape}")

    # 2) 常量/高相关
    X2, kept_var, dropped_corr = variance_corr_filter(X1, corr_th=corr_th)
    print(f"[INFO] 常量特征剔除：移除 {X1.shape[1]-len(kept_var)} 列，剩余 {len(kept_var)} 列")
    print(f"[INFO] 相关性去重（阈值 {corr_th}）：移除 {len(dropped_corr)} 列，剩余 {X2.shape[1]} 列")

    # 3) 标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X2.values)
    names_after = list(X2.columns)
    print(f"[OK] 标准化完成，形状：{Xs.shape}")

    # 4) 可选 PCA
    pca = None
    if use_pca:
        if pca_mode == "var":
            pca = PCA(n_components=pca_var, svd_solver="full", whiten=pca_whiten, random_state=seed)
        else:
            pca = PCA(n_components=min(pca_n, Xs.shape[1]), svd_solver="full", whiten=pca_whiten, random_state=seed)
        Z = pca.fit_transform(Xs)
        names_for_rfe = [f"PC{i+1}" for i in range(Z.shape[1])]
        print(f"[OK] PCA 完成，保留维度：{Z.shape[1]}")
    else:
        Z = Xs
        names_for_rfe = names_after

    print(f"[OK] 预处理拟合完成，用时 {time.time() - t0:.2f}s")
    return dict(
        imputer=imputer, kept_var=kept_var, dropped_corr=dropped_corr,
        scaler=scaler, pca=pca, names_for_rfe=names_for_rfe
    ), Z

def transform_preprocess(df: pd.DataFrame, feat_cols: List[str], prep: Dict) -> np.ndarray:
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(prep["imputer"].transform(X), columns=feat_cols, index=df.index)
    # 按训练阶段留下的列顺序裁剪
    X = X[prep["kept_var"]].drop(columns=prep["dropped_corr"], errors="ignore")
    Xs = prep["scaler"].transform(X.values)
    if prep["pca"] is not None:
        Z = prep["pca"].transform(Xs)
        return Z
    return Xs

# ----------------- 置信度权重（仅训练集） -----------------
def confidence_weights(prob: Optional[np.ndarray], mode: str, min_conf: float,
                       conf_quantile: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    返回 (sample_weight, keep_mask, info)
    mode:
        soft      -> w = 2|p-0.5|, keep all
        hard      -> keep |p-0.5| >= min_conf, w = 1
        hybrid    -> keep |p-0.5| >= min_conf, w = 2|p-0.5| (else 0)
        quantile  -> keep |p-0.5| >= Q(q), w = 1
        ranksoft  -> keep |p-0.5| >= Q(q), w = 2|p-0.5| (else 0)
    说明：Q(q) 为 |p-0.5| 的 q 分位数（比如 q=0.7 -> 仅保留前 30% 最自信样本）
    """
    info = {}
    if prob is None:
        return np.ones((0,), dtype=float), np.ones((0,), dtype=bool), info

    p = prob.astype(float)
    m = np.abs(p - 0.5)               # 置信度度量
    info["mean_abs_margin"] = float(m.mean())
    info["std_abs_margin"]  = float(m.std(ddof=1)) if m.size > 1 else 0.0

    if mode in {"quantile", "ranksoft"}:
        q = conf_quantile if (conf_quantile is not None and 0.0 < conf_quantile < 1.0) else 0.7
        thr_q = float(np.quantile(m, q)) if m.size > 0 else 1.0
        mask = m >= thr_q
        w = np.ones_like(m) if mode == "quantile" else 2.0 * m
        info["quantile"] = float(q)
        info["thr_quantile_abs_margin"] = float(thr_q)
    elif mode == "soft":
        mask = np.ones_like(m, bool)
        w = 2.0 * m
    elif mode == "hard":
        thr = max(0.0, float(min_conf))
        mask = m >= thr
        w = np.ones_like(m)
        info["thr_abs_margin"] = float(thr)
    elif mode == "hybrid":
        thr = max(0.0, float(min_conf))
        mask = m >= thr
        w = np.where(mask, 2.0*m, 0.0)
        info["thr_abs_margin"] = float(thr)
    else:
        # 兜底 soft
        mask = np.ones_like(m, bool)
        w = 2.0 * m

    # 数值安全
    w = np.clip(w, 1e-3, 1.0)
    info["kept"] = int(mask.sum()); info["total"] = int(mask.size)
    info["kept_ratio"] = float(mask.mean())
    return w, mask, info

# ----------------- 稳定性 RFE Top-k（仅训练集） -----------------
def rfe_stability_topk(X_tr: np.ndarray, y_tr: np.ndarray, feat_names: List[str],
                       k: int, runs: int, seed: int, sample_weight: Optional[np.ndarray]) -> Tuple[List[str], pd.DataFrame]:
    rng = np.random.RandomState(seed)
    freq = pd.Series(0.0, index=feat_names, dtype=float)
    n = X_tr.shape[0]
    k_sel = min(max(1, k), X_tr.shape[1])

    for _ in tqdm(range(runs), desc="Stability-RFE runs", unit="run"):
        idx = rng.choice(n, size=n, replace=True)
        Xb, yb = X_tr[idx], y_tr[idx]
        sw = sample_weight[idx] if sample_weight is not None else None

        base = LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced", C=1.0, max_iter=10000, random_state=seed
        )
        selector = RFE(estimator=base, n_features_to_select=k_sel, step=1)
        if sw is not None:
            selector.fit(Xb, yb, sample_weight=sw)
        else:
            selector.fit(Xb, yb)

        chosen = np.array(feat_names)[selector.support_]
        freq.loc[chosen] += 1.0

    freq = (freq / max(1, runs)).sort_values(ascending=False).to_frame("selection_freq")
    topk = freq.head(k_sel).index.tolist()
    return topk, freq

# ----------------- 重叠检测（按键） -----------------
def _make_overlap_key(df: pd.DataFrame, mode: str) -> Optional[pd.Series]:
    mode = (mode or "").lower()
    if mode == "case_id" and "case_id" in df.columns:
        return df["case_id"].astype(str)
    if mode == "image_path" and "image_path" in df.columns:
        return df["image_path"].astype(str)
    if mode == "image+mask" and "image_path" in df.columns and "mask_path" in df.columns:
        return df["image_path"].astype(str) + "|" + df["mask_path"].astype(str)
    return None

# ----------------- 训练（GridSearchCV）并在测试集评估 -----------------
def train_and_eval_models(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                          outdir: str, n_jobs: int, seed: int, sample_weight: Optional[np.ndarray],
                          cv_folds: int, diagnostics: Dict) -> None:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def run_one(name, clf, grid, supports_weight: bool):
        print(f"[INFO] [{name}] GridSearchCV 开始…（{cv_folds}-fold, scoring=roc_auc）")
        pipe = Pipeline([("clf", clf)])
        gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, refit=True, verbose=0)
        fit_params = {}
        if supports_weight and sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        t0 = time.time()
        gs.fit(X_tr, y_tr, **fit_params)
        print(f"[OK] [{name}] GridSearchCV 完成，用时 {time.time()-t0:.2f}s；最佳参数：{gs.best_params_}")

        # 训练CV AUC（用于对比）
        best_cv_auc = float(np.max(gs.cv_results_["mean_test_score"]))
        print(f"[CV] [{name}] best_cv_auc = {best_cv_auc:.4f}")

        best = gs.best_estimator_
        if hasattr(best, "predict_proba"):
            prob = best.predict_proba(X_te)[:, 1]
        else:
            scores = best.decision_function(X_te)
            prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y_te, prob)
        ap  = average_precision_score(y_te, prob)
        acc = accuracy_score(y_te, pred)
        bacc= balanced_accuracy_score(y_te, pred)
        f1  = f1_score(y_te, pred)

        fpr, tpr, _ = roc_curve(y_te, prob)
        pd.DataFrame({"prob": prob, "pred": pred, "label": y_te}).to_csv(
            os.path.join(outdir, f"test_pred_{name}.csv"), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(gs.cv_results_).to_csv(
            os.path.join(outdir, f"cv_results_{name}.csv"), index=False, encoding="utf-8-sig"
        )
        alt_auc = 1.0 - auc
        warn_flip = alt_auc - auc > 0.05  # 1-AUC 明显更高，疑似标签取反
        if warn_flip:
            print(f"[WARN] [{name}] 1-AUC 明显更高，疑似训练/测试标签语义不一致（建议检查 High/Low / 0/1 对齐）。")
        print(f"[INFO] [{name}] TEST -> AUC={auc:.4f} | AP={ap:.4f} | ACC={acc:.4f} | BAL_ACC={bacc:.4f} | F1={f1:.4f} | (1-AUC)={alt_auc:.4f}")

        diagnostics["models"].append(dict(
            model=name, auc=auc, ap=ap, acc=acc, bal_acc=bacc, f1=f1,
            best_params=gs.best_params_, best_cv_auc=best_cv_auc,
            alt_auc=alt_auc, suspect_label_flip=bool(warn_flip)
        ))
        return dict(model=name, auc=auc, ap=ap, acc=acc, bal_acc=bacc, f1=f1, best_params=gs.best_params_), (name, fpr, tpr, auc)

    # 模型与网格
    results = []
    curves  = []

    # Logistic ENet
    log_clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                 class_weight="balanced", max_iter=50000, random_state=seed)
    log_grid = {"clf__C": np.logspace(-6, 6, 2000), "clf__l1_ratio": [0.1,0.3,0.5,0.7,0.9]}
    r, c = run_one("LogisticENet", log_clf, log_grid, supports_weight=True)
    results.append(r); curves.append(c)

    # RandomForest
    rf_clf = RandomForestClassifier(class_weight="balanced", random_state=seed)
    rf_grid = {"clf__n_estimators": [300, 600, 1000], "clf__max_depth": [None, 6, 10, 16], "clf__min_samples_leaf": [1,2,4]}
    r, c = run_one("RandomForest", rf_clf, rf_grid, supports_weight=True)
    results.append(r); curves.append(c)

    # GradientBoosting
    gbm_clf = GradientBoostingClassifier(random_state=seed)
    gbm_grid = {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2,3]}
    r, c = run_one("GradientBoosting", gbm_clf, gbm_grid, supports_weight=True)
    results.append(r); curves.append(c)

    # KNN
    knn_clf = KNeighborsClassifier()
    knn_grid = {"clf__n_neighbors": [3,5,7,9,11,15], "clf__weights": ["uniform","distance"]}
    r, c = run_one("KNN", knn_clf, knn_grid, supports_weight=False)
    results.append(r); curves.append(c)

    # XGBoost (可选)
    try:
        from xgboost import XGBClassifier
        xgb_clf = XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=seed)
        xgb_grid = {"clf__n_estimators": [300, 600, 900], "clf__max_depth": [3,4,5],
                    "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8,1.0], "clf__colsample_bytree": [0.8,1.0]}
        r, c = run_one("XGBoost", xgb_clf, xgb_grid, supports_weight=True)
        results.append(r); curves.append(c)
    except Exception:
        pass

    # 汇总
    pd.DataFrame(results).to_csv(os.path.join(outdir, "test_metrics_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({r["model"]: r["best_params"] for r in results}, f, ensure_ascii=False, indent=2)

    # ROC 合图
    plt.figure(figsize=(8,6))
    for name, fpr, tpr, auc in curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curves (all models)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves_test.png"), dpi=180)
    plt.close()
    print(f"[OK] ROC 合图已保存：{os.path.join(outdir, 'roc_curves_test.png')}")

# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser("Radiomics: Train/Test with Top-k (PCA可选) & multi-model ROC")
    # 输入：单CSV 或 train/test
    ap.add_argument("--csv", help="单CSV：按 --test-size 分层切分")
    ap.add_argument("--train-csv", help="训练CSV")
    ap.add_argument("--test-csv", help="测试CSV")

    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--label-col", default=None, help="标签列（默认自动探测）")
    ap.add_argument("--prob-col", nargs="?", default=None, const="", help="概率列；留空/none/--no-prob 可禁用")
    ap.add_argument("--no-prob", action="store_true", help="忽略概率列，完全不做置信度过滤/加权")
    ap.add_argument("--min-conf", type=float, default=0.60, help="（hard/hybrid 使用）|p-0.5| 的绝对阈值")
    # 新：权重/过滤模式 + 分位数
    ap.add_argument("--weight-mode", default="hybrid",
                    choices=["soft","hard","hybrid","quantile","ranksoft"],
                    help="伪标签处理方式：soft=加权不删；hard/hybrid=绝对阈值；quantile/ranksoft=分位数")
    ap.add_argument("--conf-quantile", type=float, default=0.0,
                    help="分位数阈值 q∈(0,1)。示例：0.7 表示保留前 30% 最自信样本（|p-0.5| ≥ Q(0.7)）。仅在 quantile/ranksoft 下使用。")

    ap.add_argument("--exclude-cols", default="", help="额外需要排除的列（逗号分隔）")
    ap.add_argument("--test-size", type=float, default=0.2, help="单CSV模式下的测试占比")
    ap.add_argument("--corr-th", type=float, default=0.95, help="相关性去重阈值")
    ap.add_argument("--topk", type=int, default=10, help="Top-k 特征/PC 数")
    ap.add_argument("--stability-runs", type=int, default=50, help="RFE 重复次数（越大越稳）")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")
    ap.add_argument("--cv", type=int, default=10, help="GridSearchCV 折数（训练集上）")
    ap.add_argument("--n-jobs", type=int, default=-1, help="并行核数")

    # PCA 可选
    ap.add_argument("--pca-before-topk", action="store_true", help="在 Top-k 之前先 PCA（默认关闭）")
    ap.add_argument("--pca-mode", default="var", choices=["var","n"], help="PCA 模式：var=按方差比；n=固定个数")
    ap.add_argument("--pca-var", type=float, default=0.95, help="var 模式保留方差比例")
    ap.add_argument("--pca-n", type=int, default=50, help="n 模式主成分个数上限")
    ap.add_argument("--pca-whiten", action="store_true", help="是否 whiten（通常不建议）")

    # 重叠检测
    ap.add_argument("--overlap-by", default="image_path",
                    choices=["case_id","image_path","image+mask"],
                    help="用于检测 train/test 重叠的键；默认 image_path。")
    ap.add_argument("--on-overlap", default="error",
                    choices=["error","drop-from-test","drop-from-train","drop-both","ignore"],
                    help="发现重叠时的处理策略；默认 error。")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    diagnostics: Dict = {"models": []}

    # 读数据 & 划分
    stage("加载数据 & 训练/测试划分")
    if args.train_csv and args.test_csv:
        df_tr = pd.read_csv(args.train_csv, encoding="utf-8-sig")
        df_te = pd.read_csv(args.test_csv,  encoding="utf-8-sig")
        label_col = auto_label(df_tr, args.label_col)
        if label_col not in df_te.columns:
            raise ValueError(f"测试集缺少标签列 {label_col}")
        print(f"[OK] 使用显式 train/test：TRAIN={df_tr.shape}, TEST={df_te.shape} | 标签列={label_col}")
    elif args.csv:
        df_all = pd.read_csv(args.csv, encoding="utf-8-sig")
        label_col = auto_label(df_all, args.label_col)
        y_all = to_binary_label(df_all[label_col])
        df_tr, df_te = train_test_split(df_all, test_size=args.test_size, random_state=args.seed, stratify=y_all)
        print(f"[OK] 单CSV分层切分：ALL={df_all.shape} -> TRAIN={df_tr.shape}, TEST={df_te.shape} | 标签列={label_col}")
    else:
        raise ValueError("请提供 --csv 或 (--train-csv 与 --test-csv)。")

    # 基本分布诊断
    tr_counts = df_tr[label_col].value_counts()
    te_counts = df_te[label_col].value_counts()
    print("[CHK] 训练集标签分布："); print(tr_counts)
    print("[CHK] 测试集标签分布："); print(te_counts)
    diagnostics["train_label_counts"] = tr_counts.to_dict()
    diagnostics["test_label_counts"]  = te_counts.to_dict()

    # 重叠检测（按 --overlap-by）
    key_tr = _make_overlap_key(df_tr, args.overlap_by)
    key_te = _make_overlap_key(df_te, args.overlap_by)
    if key_tr is not None and key_te is not None:
        inter_keys = sorted(list(set(key_tr) & set(key_te)))
        print(f"[CHK] 依据 --overlap-by={args.overlap_by} 的重叠数：{len(inter_keys)}")
        diagnostics["overlap_by"] = args.overlap_by
        diagnostics["overlap_count_before"] = len(inter_keys)
        if len(inter_keys) > 0:
            if args.on_overlap == "error":
                raise SystemExit(
                    f"检测到 train/test 重叠（依据 {args.overlap_by}）：{len(inter_keys)}。\n"
                    f"可改用 --on-overlap=drop-from-test 等策略，或把 --overlap-by 设为 image_path / image+mask。"
                )
            elif args.on_overlap == "drop-from-test":
                df_te = df_te[~key_te.isin(inter_keys)].reset_index(drop=True)
                print(f"[OK] 已从测试集移除重叠样本：{len(inter_keys)}；TEST 现为 {df_te.shape}")
                diagnostics["overlap_action"] = "drop-from-test"
                diagnostics["overlap_removed_test"] = len(inter_keys)
            elif args.on_overlap == "drop-from-train":
                df_tr = df_tr[~key_tr.isin(inter_keys)].reset_index(drop=True)
                print(f"[OK] 已从训练集移除重叠样本：{len(inter_keys)}；TRAIN 现为 {df_tr.shape}")
                diagnostics["overlap_action"] = "drop-from-train"
                diagnostics["overlap_removed_train"] = len(inter_keys)
            elif args.on_overlap == "drop-both":
                df_tr = df_tr[~key_tr.isin(inter_keys)].reset_index(drop=True)
                df_te = df_te[~key_te.isin(inter_keys)].reset_index(drop=True)
                print(f"[OK] 已从两侧移除重叠样本：{len(inter_keys)}；TRAIN {df_tr.shape} | TEST {df_te.shape}")
                diagnostics["overlap_action"] = "drop-both"
                diagnostics["overlap_removed_train"] = len(inter_keys)
                diagnostics["overlap_removed_test"]  = len(inter_keys)
            else:
                print("[WARN] 检测到重叠，但已按 --on-overlap=ignore 忽略。")
                diagnostics["overlap_action"] = "ignore"
    else:
        print(f"[CHK] --overlap-by={args.overlap_by} 对应的列在 train/test 中不完整，跳过重叠检测。")

    # 列差异/分布飘移速查
    only_tr = [c for c in df_tr.columns if c not in df_te.columns]
    only_te = [c for c in df_te.columns if c not in df_tr.columns]
    print(f"[CHK] 仅训练集有的列（前20）：{only_tr[:20]}")
    print(f"[CHK] 仅测试集有的列（前20）：{only_te[:20]}")
    diagnostics["only_train_columns_top20"] = only_tr[:20]
    diagnostics["only_test_columns_top20"]  = only_te[:20]

    # 均值差最大的特征（只看数值列&公共列，排除明显非特征）
    non_feat = {"case_id","image_path","mask_path",label_col,"GATA6","label"}
    common_cols = [c for c in df_tr.columns if (c in df_te.columns and c not in non_feat)]
    num_cols = [c for c in common_cols
                if (pd.api.types.is_numeric_dtype(df_tr[c]) or pd.api.types.is_float_dtype(df_tr[c]))]
    mean_diff = {}
    for c in num_cols:
        tr_m = pd.to_numeric(df_tr[c], errors="coerce").mean()
        te_m = pd.to_numeric(df_te[c], errors="coerce").mean()
        mean_diff[c] = abs(tr_m - te_m)
    mean_diff_top10 = pd.Series(mean_diff).sort_values(ascending=False).head(10)
    print("[CHK] 训练/测试均值差Top-10："); print(mean_diff_top10)
    diagnostics["mean_diff_top10"] = {k: float(v) for k, v in mean_diff_top10.items()}

    # 标签
    y_tr = to_binary_label(df_tr[label_col])
    y_te = to_binary_label(df_te[label_col])
    diagnostics["train_label_unique_examples"] = list(np.unique(df_tr[label_col]).astype(str)[:2])
    diagnostics["test_label_unique_examples"]  = list(np.unique(df_te[label_col]).astype(str)[:2])

    # 概率列处理
    stage("（可选）训练集伪标签置信度过滤/加权")
    prob_col = auto_prob(df_tr, args.prob_col, no_prob=args.no_prob)
    if prob_col is None:
        prob_tr = None
        print("[INFO] 已禁用概率/置信度（--no-prob 或未找到 prob 列）")
    else:
        prob_tr = df_tr[prob_col].astype(float).values
        print(f"[INFO] 使用概率列：{prob_col}（仅作用于训练集）")

    if prob_tr is not None:
        q_used = args.conf_quantile if (0.0 < args.conf_quantile < 1.0) else None
        w_tr, keep_mask, conf_info = confidence_weights(prob_tr, args.weight_mode, args.min_conf, conf_quantile=q_used)
        if keep_mask.sum() < len(keep_mask):
            print(f"[INFO] 置信度过滤：训练集保留 {keep_mask.sum()}/{len(keep_mask)} ({keep_mask.mean():.1%})")
        # 仅训练集被过滤/加权
        df_tr = df_tr.loc[keep_mask].reset_index(drop=True)
        y_tr  = y_tr[keep_mask]
        prob_tr = prob_tr[keep_mask]
        w_tr = w_tr[keep_mask]
        diagnostics["prob_strategy"] = dict(
            mode=args.weight_mode, min_conf=float(args.min_conf),
            conf_quantile=(float(q_used) if q_used is not None else None),
            **conf_info
        )
    else:
        w_tr = None
        diagnostics["prob_strategy"] = dict(mode="disabled")

    # 特征列选择（仅根据训练集决定）
    stage("构建特征矩阵（仅训练集拟合一切变换）")
    exclude = {"case_id", "image_path", "mask_path", label_col, "GATA6", "label"}
    if args.exclude_cols.strip():
        extra = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
        exclude.update(extra)
        if extra:
            print(f"[INFO] 额外排除列：{extra}")

    prob_like_train = find_prob_like_cols(list(df_tr.columns))
    if prob_col is None:
        # 完全禁用概率：训练集里所有“像模型输出”的列都排除（radiomics 特征不会被误删）
        exclude.update(prob_like_train)
        if prob_like_train:
            print(f"[INFO] 模型输出/分数列已禁用并从特征中移除：{prob_like_train}")
    else:
        # 只保留 prob_col 用于权重，其它类似列移除
        others = [c for c in prob_like_train if c != prob_col]
        exclude.update(others)
        if others:
            print(f"[INFO] 为避免泄露，除 '{prob_col}' 外的模型输出/分数列已移除：{others}")

    feat_cols = [c for c in df_tr.columns if c not in exclude]
    # 双保险：万一还有漏网（极少数命名）
    leaks = [c for c in feat_cols if _looks_like_model_output(c)]
    if leaks:
        print(f"[WARN] 发现漏网的模型输出样式列，已从特征中移除：{leaks}")
        feat_cols = [c for c in feat_cols if c not in set(leaks)]

    print(f"[INFO] 特征列数：{len(feat_cols)}")

    # 预处理拟合（仅训练集）
    stage("训练集：拟合预处理（中位数填充 → 常量特征剔除 → 相关性去重 → 标准化 → 可选PCA）")
    prep, Xtr_pp = fit_preprocess(
        df_tr[feat_cols].apply(pd.to_numeric, errors="coerce"),
        corr_th=args.corr_th,
        use_pca=args.pca_before_topk,
        pca_mode=args.pca_mode, pca_var=args.pca_var, pca_n=args.pca_n,
        pca_whiten=args.pca_whiten, seed=args.seed
    )
    Xte_pp = transform_preprocess(df_te, feat_cols, prep)
    print(f"[OK] 变换后维度：TRAIN={Xtr_pp.shape}, TEST={Xte_pp.shape} | 名称示例：{prep['names_for_rfe'][:3]}…")

    # 稳定 RFE Top-k（仅训练集）
    stage(f"训练集：稳定 RFE → Top-{args.topk}（重复 {args.stability_runs} 次）")
    t0 = time.time()
    topk_names, freq_df = rfe_stability_topk(
        Xtr_pp, y_tr, feat_names=prep["names_for_rfe"],
        k=args.topk, runs=args.stability_runs, seed=args.seed, sample_weight=w_tr
    )
    freq_df.to_csv(os.path.join(args.outdir, "rfe_frequency.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.outdir, "top_features.txt"), "w", encoding="utf-8") as f:
        for n in topk_names: f.write(n + "\n")
    print(f"[OK] RFE 完成（用时 {time.time()-t0:.2f}s），Top-{len(topk_names)}: {', '.join(topk_names)}")

    # 组装 Top-k 表示
    if args.pca_before_topk:
        idx = [int(n.replace("PC",""))-1 for n in topk_names]
        Xtr_top = Xtr_pp[:, idx]
        Xte_top = Xte_pp[:, idx]
    else:
        # names_for_rfe 与 Xtr_pp 列一一对应
        name_to_idx = {n:i for i, n in enumerate(prep["names_for_rfe"])}
        sel_idx = [name_to_idx[n] for n in topk_names]
        Xtr_top = Xtr_pp[:, sel_idx]
        Xte_top = Xte_pp[:, sel_idx]

    stage("训练模型并在测试集评估")
    print(f"[INFO] 用于建模的维度：{Xtr_top.shape[1]}")

    # 训练/评估
    stage(f"训练集：多模型超参搜索（GridSearchCV={args.cv} 折，评分=AUC）并在测试集评估")
    train_and_eval_models(
        Xtr_top, y_tr, Xte_top, y_te,
        outdir=args.outdir, n_jobs=args.n_jobs, seed=args.seed,
        sample_weight=w_tr, cv_folds=args.cv, diagnostics=diagnostics
    )

    # 预处理摘要
    stage("保存预处理摘要")
    summary = dict(
        mode="train/test",
        label_col=label_col,
        prob_col=(prob_col if prob_col is not None else "DISABLED"),
        weight_mode=args.weight_mode,
        min_conf=float(args.min_conf),
        conf_quantile=(float(args.conf_quantile) if 0.0 < args.conf_quantile < 1.0 else None),
        exclude_cols=sorted(list(exclude)),
        n_features_raw=len(feat_cols),
        kept_after_corr=len(prep["names_for_rfe"]) if not args.pca_before_topk else f"PCA:{len(prep['names_for_rfe'])}",
        topk=topk_names,
        pca=dict(enabled=args.pca_before_topk, mode=args.pca_mode, var=args.pca_var, n=args.pca_n, whiten=args.pca_whiten),
        corr_th=args.corr_th,
        seed=args.seed
    )
    with open(os.path.join(args.outdir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] preprocess_summary.json 已保存")

    # 诊断信息保存
    with open(os.path.join(args.outdir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print("[OK] diagnostics.json 已保存")

    stage("全部完成")
    print("✅ 结果已保存到：", args.outdir)


if __name__ == "__main__":
    main()
