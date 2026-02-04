#!/bin/bash

# ================= 配置区域 =================
# 1. 尝试加载环境配置 (如果存在)
if [ -f "config/env.sh" ]; then
    source config/env.sh
else
    # 兜底配置 (防止找不到环境变量)
    MODEL_PATH="/home/cgj/studio/vllm-ProfileEviction/models/Meta-Llama-3.1-8B-Instruct"
fi

# 2. 参数解析
# 参数1: 模式 (默认为 streamingLLM)
MODE=${1:-"streamingLLM"} 
# 参数2: Budget (默认为 4096)
BUDGET=${2:-4096}
# 参数3: 数据集列表 (默认为你列出的4个长文本数据集)
DEFAULT_SETS="gov_report multi_news qasper hotpotqa"
DATASETS=${3:-$DEFAULT_SETS}

# 3. 模式匹配与参数构建
# 目标格式: origin_模式_模型简写_Budget
MODEL_SHORT="318B" # Llama-3.1-8B 的简写

case $MODE in
    "full-cache")
        OUTPUT_DIR="results/origin_fullCache_${MODEL_SHORT}_${BUDGET}"
        PY_ARGS="" # Full Cache 不需要额外参数 (根据你的Python逻辑)
        echo ">>> [Mode] Running Standard Full Cache"
        ;;
        
    "streamingLLM")
        OUTPUT_DIR="results/origin_streamingLLM_${MODEL_SHORT}_${BUDGET}"
        PY_ARGS="--enable-paged-eviction --evict-method streamingLLM --initial-blocks 16"
        echo ">>> [Mode] Running StreamingLLM (Init: 16)"
        ;;
        
    *)
        echo "❌ Error: Unknown mode '$MODE'. Supported: full-cache, streamingLLM"
        exit 1
        ;;
esac

echo "=========================================="
echo "📜 Historical Regression Test"
echo "Model:  $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Budget: $BUDGET"
echo "Tasks:  $DATASETS"
echo "=========================================="

# ================= 执行循环 =================
mkdir -p "$OUTPUT_DIR"

for dataset in $DATASETS; do
    echo -e "\n----------------------------------------"
    echo -e "Processing dataset: \033[0;32m${dataset}\033[0m"
    
    python src/run_vllm_historical.py \
        --model "$MODEL_PATH" \
        --datasets "$dataset" \
        --output-dir "$OUTPUT_DIR" \
        --max-samples 20 \
        --cache-budget "$BUDGET" \
        --max-len 16384 \
        $PY_ARGS
        
    if [ $? -eq 0 ]; then
        echo "✅ Dataset '${dataset}' finished."
    else
        echo "❌ Dataset '${dataset}' failed."
        exit 1
    fi
done

echo -e "\n🎉 All historical tests completed."
