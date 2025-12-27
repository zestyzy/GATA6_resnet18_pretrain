#!/usr/bin/env bash
set -euo pipefail

##### ==== 必改路径 ==== #####
CSV="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/prep_table2/pca_10_from_l1selected.csv"     # 你的CSV
OUTROOT="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/results/baselines_test/grid_search"   # 所有结果的根目录
SCRIPT="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/code/extra_baselines_model.py"  # 就用你现有这份脚本

##### ==== 可调参数 ==== #####
FOLDS=10            # 每个seed下的打乱次数（与脚本 --folds 对应）
TEST_SIZE=0.2       # 验证集比例
START_SEED=1     # 起始seed
N_SEEDS=100         # 连续seed数量（会跑 START_SEED..START_SEED+N_SEEDS-1）
TOPK=10             # 汇总时各模型展示的Top-K种子

# 如需指定GPU：取消下一行注释并设定
# export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUTROOT}"
export OUTROOT  # 供下方Python汇总脚本读取

echo "=== Run seeds: ${START_SEED}..$((START_SEED+N_SEEDS-1)) ==="
for (( SEED="${START_SEED}"; SEED<START_SEED+N_SEEDS; SEED++ )); do
  OUTDIR="${OUTROOT}/seed_${SEED}"
  mkdir -p "${OUTDIR}"
  echo ">>> [SEED=${SEED}] running..."
  PYTHONUNBUFFERED=1 python "${SCRIPT}" \
    --csv "${CSV}" \
    --outdir "${OUTDIR}" \
    --folds "${FOLDS}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}" 2>&1 | tee "${OUTDIR}/run.log"
done

echo "=== All seeds finished. Aggregating... ==="

# ---- 汇总：递归收集每个 seed/<model>/<model>_pseudo5fold_auc.csv ----
python - <<'PY'
import os, glob, pandas as pd, numpy as np

root = os.environ["OUTROOT"]
rows = []
# 匹配 .../seed_xxxx/<ModelName>/<ModelName>_pseudo5fold_auc.csv
for csv_path in glob.glob(os.path.join(root, "seed_*", "*", "*_pseudo5fold_auc.csv")):
    try:
        df = pd.read_csv(csv_path)
        # 目录结构解析
        model_dir = os.path.basename(os.path.dirname(csv_path))     # <ModelName>
        seed_dir  = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))  # seed_xxxx
        # 读取统计
        mean_auc = float(df["mean_auc"].iloc[0])
        std_auc  = float(df["std_auc"].iloc[0])
        ci_low   = float(df["ci_low"].iloc[0])
        ci_high  = float(df["ci_high"].iloc[0])
        seed = int(seed_dir.split("_")[-1])
        rows.append({
            "seed": seed,
            "model": model_dir,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "dir": os.path.dirname(csv_path)
        })
    except Exception as e:
        print(f"[WARN] Skip {csv_path}: {e}")

if not rows:
    print("[WARN] No per-seed results found under", root)
    raise SystemExit(0)

seeds_summary = pd.DataFrame(rows).sort_values(["model","mean_auc","std_auc"], ascending=[True, False, True])
seeds_summary_path = os.path.join(root, "seeds_summary.csv")
seeds_summary.to_csv(seeds_summary_path, index=False)
print(f"[OK] seeds_summary.csv -> {seeds_summary_path}")

# 为每个模型导出 Top-K 种子，并生成模型层面的排行榜
TOPK = int(os.environ.get("TOPK", "10"))
models = sorted(seeds_summary["model"].unique().tolist())
model_rows = []
for m in models:
    sub = seeds_summary[seeds_summary["model"] == m].copy()
    sub = sub.sort_values(["mean_auc","std_auc"], ascending=[False, True])
    topk = sub.head(min(TOPK, len(sub)))
    top_path = os.path.join(root, f"top{TOPK}_seeds_{m}.csv")
    topk.to_csv(top_path, index=False)
    best = topk.iloc[0]
    model_rows.append({
        "model": m,
        "topk_mean_of_means": float(topk["mean_auc"].mean()),
        "best_seed": int(best["seed"]),
        "best_mean_auc": float(best["mean_auc"]),
        "best_ci": f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}]",
        "count": int(len(sub))
    })

models_rank = pd.DataFrame(model_rows).sort_values(
    ["topk_mean_of_means","best_mean_auc"], ascending=[False, False]
)
models_rank_path = os.path.join(root, f"models_top{TOPK}_by_seed.csv")
models_rank.to_csv(models_rank_path, index=False)
print("\n===== Leaderboard (per model; averaged over its Top-K seeds) =====")
print(models_rank.to_string(index=False))
print(f"\n[OK] models_top{TOPK}_by_seed.csv -> {models_rank_path}")
PY

echo "=== Done. All results under: ${OUTROOT} ==="
