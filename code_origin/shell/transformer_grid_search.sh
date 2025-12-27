#!/usr/bin/env bash
set -euo pipefail

# ========= 必改路径 =========
CSV="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/prep_table2/pca_10_from_l1selected.csv"
OUTROOT="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/results/transformer_test/grid_search"  # 修正拼写
SCRIPT="/data/dataserver01/public/yzeng/6yuan_task/task3_GATA6/code/step3_train_eval.py"

# ========= 设备设置 =========
export CUDA_VISIBLE_DEVICES=1
export OUTROOT  # 供汇总Python读取

# ========= 超参网格 =========
LRS=(0.0003 0.001 0.003)
WDS=(0.00005 0.0001 0.0005)
DROPS=(0.20 0.35)
BATCH_SIZES=(12 16)
PATCHES=(8 16)
DMODELS=(96 128)
SEEDS=(2025 2026 2027)   # 新增：用于稳健性评估

# ========= 其它固定项（与脚本参数一致）=========
DEPTHS=3
EPOCHS=400
PATIENCE=15
FFN=256
REPEATS=10
TEST_SIZE=0.2

# d_model 与 nhead 的匹配（确保能整除）
NHEAD_96=6
NHEAD_128=8

mkdir -p "${OUTROOT}"

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    for DP in "${DROPS[@]}"; do
      for PS in "${PATCHES[@]}"; do
        for DM in "${DMODELS[@]}"; do
          if [ "${DM}" -eq 96 ]; then
            NHEAD=${NHEAD_96}
          else
            NHEAD=${NHEAD_128}
          fi
          for DEPTH in "${DEPTHS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
              for SEED in "${SEEDS[@]}"; do
                TAG="dm${DM}_nh${NHEAD}_depth${DEPTH}_ps${PS}_bs${BS}_lr${LR}_wd${WD}_dp${DP}_seed${SEED}"
                OUTDIR="${OUTROOT}/${TAG}"
                mkdir -p "${OUTDIR}"

                # 保存本次配置（便于复现）
                cat > "${OUTDIR}/hparams.json" <<JSON
{
  "csv": "${CSV}",
  "repeats": ${REPEATS},
  "test_size": ${TEST_SIZE},
  "seed": ${SEED},
  "batch_size": ${BS},
  "epochs": ${EPOCHS},
  "patience": ${PATIENCE},
  "lr": ${LR},
  "weight_decay": ${WD},
  "d_model": ${DM},
  "nhead": ${NHEAD},
  "depth": ${DEPTH},
  "ffn_dim": ${FFN},
  "dropout": ${DP},
  "patch_size": ${PS}
}
JSON

                echo ">>> Run ${TAG}"
                PYTHONUNBUFFERED=1 python "${SCRIPT}" \
                  --csv "${CSV}" \
                  --outdir "${OUTDIR}" \
                  --repeats "${REPEATS}" \
                  --test_size "${TEST_SIZE}" \
                  --seed "${SEED}" \
                  --batch_size "${BS}" \
                  --epochs "${EPOCHS}" \
                  --patience "${PATIENCE}" \
                  --lr "${LR}" \
                  --weight_decay "${WD}" \
                  --d_model "${DM}" \
                  --nhead "${NHEAD}" \
                  --depth "${DEPTH}" \
                  --ffn_dim "${FFN}" \
                  --dropout "${DP}" \
                  --patch_size "${PS}" \
                  2>&1 | tee "${OUTDIR}/train.log"
              done
            done
          done
        done
      done
    done
  done
done

echo "=== All runs finished. Results saved under: ${OUTROOT} ==="

# ========= 汇总：读取每个目录下的 shuffle_auc.csv，求均值排序 =========
python - <<'PY'
import os, glob, pandas as pd, numpy as np
root = os.environ["OUTROOT"]  # 关键修改：从环境变量读
rows=[]
for f in glob.glob(os.path.join(root, "**", "shuffle_auc.csv"), recursive=True):
    tag = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f)
        if "auc" in df.columns:
            mean_auc = float(np.nanmean(df["auc"].values))
            rows.append({"tag": tag, "mean_auc": mean_auc, "dir": os.path.dirname(f)})
    except Exception:
        pass
if rows:
    out = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    out.to_csv(os.path.join(root, "summary.csv"), index=False)
    print(out.head(30).to_string(index=False))
    print(f"\n[OK] Summary saved to: {os.path.join(root, 'summary.csv')}")
else:
    print("[WARN] No shuffle_auc.csv found under runs.")
PY
