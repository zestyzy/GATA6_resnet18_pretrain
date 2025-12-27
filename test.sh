#!/usr/bin/env bash
set -e

# ================= 路径参数（按你的机器改） =================
DATA_ROOT="/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/origin/中山医院有标 签136"
LABEL_CSV="/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/temp_dataset/dataset.csv"
RUNS_DIR="/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/test"

# ================= 训练公共参数（按需改） =================
EPOCHS=30
BATCH_SIZE=4         # 如果 resnet50 OOM，可改成 2
LR=1e-3
REPEATS=5
VAL_SPLIT=0.3
WORKERS=4
NORM="gn"            # 你的训练默认是 GN
WEIGHT_DECAY=1e-4
SAVE_EVERY=5
MAX_GRAD_NORM=0.0
USE_LIGHT_AUG=""     # 加 "--use-light-aug" 开启轻增广
USE_POS_WEIGHT=""    # 加 "--use-pos-weight"（通常不与采样器同用）

# 是否在 layer4 对深度（D）也降采样（你的实现里默认 false）
DOWNSAMPLE_L4_FLAG=""  # 需要的话写 "--downsample-depth-l4"

# 可选：限制可见 GPU
# export CUDA_VISIBLE_DEVICES=0

mkdir -p "$RUNS_DIR"

# 小工具：打印并执行，并把日志保存
run () {
  echo ""
  echo ">>> $*"
  echo ""
  eval "$*"
}

# ================ 实验矩阵定义 ================
# A2: bbox 裁剪 + 1 通道（只图像）
# A3: bbox 裁剪 + 2 通道（图像+mask）
# A4: full 不裁剪 + 2 通道（图像+mask）
# A4-1ch: full 不裁剪 + 1 通道（只图像）
declare -A EXP_CROP
declare -A EXP_CH


EXP_CROP["A3"]="bbox";    EXP_CH["A3"]=2


ARCHES=("resnet50")

# ================ 主循环 ================
for ARCH in "${ARCHES[@]}"; do
  for EXP in  "A3"; do
    CROP_MODE="${EXP_CROP[$EXP]}"
    IN_CH="${EXP_CH[$EXP]}"

    OUTDIR="${RUNS_DIR}/${EXP}_${ARCH}_${CROP_MODE}_${IN_CH}ch"
    mkdir -p "$OUTDIR"

    # 可针对 resnet50 单独调 batch size（如果显存吃紧）
    BS="$BATCH_SIZE"
    if [[ "$ARCH" == "resnet50" ]]; then
      BS="$BATCH_SIZE"     # 需要可改成 2，例如：BS=2
    fi

    CMD="python -m teacher_model.train \
      --data-root \"$DATA_ROOT\" \
      --label-csv \"$LABEL_CSV\" \
      --outdir \"$OUTDIR\" \
      --epochs $EPOCHS \
      --batch-size $BS \
      --lr $LR \
      --repeats $REPEATS \
      --val-split $VAL_SPLIT \
      --workers $WORKERS \
      --weight-decay $WEIGHT_DECAY \
      --max-grad-norm $MAX_GRAD_NORM \
      --save-every $SAVE_EVERY \
      --arch $ARCH \
      --in-channels $IN_CH \
      --norm $NORM \
      --crop-mode $CROP_MODE \
      $DOWNSAMPLE_L4_FLAG \
      $USE_LIGHT_AUG $USE_POS_WEIGHT | tee \"$OUTDIR/train.log\""

    run "$CMD"
  done
done

echo ""
echo "==== 全部实验提交完成 ===="
echo "输出根目录：$RUNS_DIR"
