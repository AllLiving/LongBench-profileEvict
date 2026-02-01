#!/bin/bash

# --- 核心路径配置 ---
# 修改为你当前的 Llama-3 路径
export MODEL_PATH="/home/cgj/studio/vllm-ProfileEviction/models/Llama-3.2-1B-Instruct"

# 结果输出根目录
export OUTPUT_ROOT="results"

# --- 数据集配置 ---
# 默认测试集，可以被脚本参数覆盖
export DEFAULT_DATASETS="samsum" 
# export DEFAULT_DATASETS="narrativeqa samsum" 

# --- Python 路径修复 ---
# 确保脚本能找到 src 目录下的模块
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
