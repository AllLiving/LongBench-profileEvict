#!/bin/bash

source config/env.sh

# 获取实验名称 (例如: baseline_full 或 ours_l2_1024)
EXP_NAME=$1

if [ -z "$EXP_NAME" ]; then
    echo "Usage: bash scripts/run_eval.sh <experiment_folder_name>"
    echo "Example: bash scripts/run_eval.sh ours_l2_1024"
    exit 1
fi

TARGET_DIR="$OUTPUT_ROOT/$EXP_NAME"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

echo "Evaluating results in: $TARGET_DIR"

# --- 关键修改 ---
# 此时 eval.py 只接受 --input-dir，请确保删除了 --model 参数
python src/eval.py --input-dir "$TARGET_DIR"
