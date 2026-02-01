#!/bin/bash

# 1. 加载公共配置
source config/env.sh

# 2. 默认参数
MODE=${1:-"baseline"}  # 默认为 baseline，可选: ours_l2, ours_richness
BUDGET=${2:-1024}      # 默认 Budget

echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Mode:  $MODE"
echo "Budget: $BUDGET"
echo "Datasets: $DEFAULT_DATASETS"
echo "=========================================="

# 3. 定义通用运行函数
run_task() {
    local EXP_NAME=$1
    local EXTRA_ARGS=$2
    
    echo ">>> Running Task: $EXP_NAME"
    
    python src/run_vllm.py \
        --model $MODEL_PATH \
        --datasets $DEFAULT_DATASETS \
        --output-dir "$OUTPUT_ROOT/$EXP_NAME" \
        --max-samples 20 \
        --cache-budget $BUDGET \
        $EXTRA_ARGS
}

# 4. 模式选择逻辑
case $MODE in
    "baseline")
        # 对应你原本的 Full Cache Baseline
        run_task "baseline_full" ""
        ;;
        
    "streaming")
        # 对应你原本的 StreamingLLM Baseline
        run_task "streaming_${BUDGET}" \
            "--enable-paged-eviction --evict-method streamingLLM --initial-blocks 16"
        ;;
        
    "ours_l2")
        # 对应你原本的 PagedEviction (Value L2)
        run_task "ours_l2_${BUDGET}" \
            "--enable-paged-eviction --evict-method global --evict-metric value_l2 --initial-blocks 4"
        ;;
        
    "ours_richness")
        # 对应你原本的 PagedEviction (Profile Richness)
        run_task "ours_richness_${BUDGET}" \
            "--enable-paged-eviction --evict-method global --evict-metric profile_richness --initial-blocks 4"
        ;;
        
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Usage: bash scripts/run_inference.sh [baseline|streaming|ours_l2|ours_richness] [budget]"
        exit 1
        ;;
esac
