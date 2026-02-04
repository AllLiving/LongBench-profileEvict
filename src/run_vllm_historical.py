import argparse
import os
import json
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ==========================================
# 1. 任务特定的 Prompt 模板定义
# ==========================================
def build_prompt(dataset_name, context, query):
    """
    根据不同数据集，构建适配 Llama-3-Instruct 的 Prompt。
    核心目标：明确任务指令，抑制废话。
    """
    
    # 默认 System Prompt
    system_msg = "You are a helpful assistant."
    user_msg = ""
    
    # --- 针对 Passage Retrieval 的特殊优化 ---
    if "passage_retrieval" in dataset_name:
        system_msg = (
            "You are an intelligent retrieval assistant. "
            "Task: Identification. "
            "Instruction: Read the 30 paragraphs provided in the context. "
            "Determine which paragraph matches the given abstract. "
            "Output format: The output must ONLY be the paragraph name (e.g., 'Paragraph 1', 'Paragraph 15'). "
            "Do not output any other text."
        )
        # 将陈述句 query 转化为疑问句，引导模型
        user_msg = f"Context:\n{context}\n\nAbstract: {query}\n\nQuestion: Which paragraph is this abstract referring to?"
    
    # --- 针对 Summarization (如 Samsum) 的优化 ---
    elif dataset_name in ["gov_report", "samsum", "qmsum", "multi_news"]:
        system_msg = (
            "You are a summarization assistant. "
            "Instruction: Summarize the text provided efficiently. "
            "Directly output the summary without introductory phrases."
        )
        user_msg = f"Text:\n{context}\n\nQuestion: Summarize the text above."
        
    # --- 兜底通用模板 ---
    else:
        system_msg = "You are a helpful assistant. Answer the question based on the context provided."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    # Llama-3 Chat 格式封装
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

# ==========================================
# 2. 主流程
# ==========================================
def main(args):
    print(f"--- [Step 1] Initializing Model: {args.model} ---")
    # 强制覆盖配置，确保 Llama-3.2 长文本生效
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": args.max_len,     # 65536
        "max_num_batched_tokens": args.max_len,
        "enforce_eager": True,
        "dtype": "auto",
        "hf_overrides": {
            "rope_theta": 500000.0, 
            "max_position_embeddings": args.max_len
        }
    }
    
    # 如果开启 Eviction
    if args.enable_paged_eviction:
        print(f"--- [Config] Paged Eviction ENABLED (Budget: {args.cache_budget}) ---")
        engine_args.update({
            "enable_paged_eviction": True,
            "evict_method": args.evict_method,
            # "evict_metric": args.evict_metric,
            "cache_budget": args.cache_budget,
            # "initial_blocks": args.initial_blocks
        })
    else:
        print(f"--- [Config] Full Cache Mode (Standard) ---")

    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n--- [Step 2] Processing Dataset: {dataset_name} ---")
        
        # 加载数据
        data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        if args.max_samples > 0:
            data = data.select(range(min(len(data), args.max_samples)))

        input_prompts = []
        raw_examples = [] # 保存原始引用，以便后续存储结果

        # --- [Step 3] 构建 Prompt ---
        print("Building prompts...")
        for example in data:
            context = example['context']
            query = example['input']
            
            # 构建最终 Prompt
            final_prompt = build_prompt(dataset_name, context, query)
            input_prompts.append(final_prompt)
            raw_examples.append(example)

        # DEBUG: 打印第一个 Prompt 的尾部，检查格式
        print(f"\n[DEBUG] Prompt Tail (Example 0):\n...{input_prompts[0][-500:]}")
        print(f"[DEBUG] Prompt Token Len: {len(tokenizer.encode(input_prompts[0]))}\n")

        # --- [Step 4] 执行生成 ---
        print(f"Generating responses for {len(input_prompts)} samples...")
        sampling_params = SamplingParams(
            temperature=0,           # 贪婪采样，最稳
            max_tokens=200,          # 检索任务不需要生成很长
            stop=["<|eot_id|>"]      # 及时停止
        )
        
        outputs = llm.generate(input_prompts, sampling_params)
        
        # --- [Step 5] 保存结果 ---
        output_file = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
        print(f"Saving to {output_file}...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for output, example in zip(outputs, raw_examples):
                pred = output.outputs[0].text.strip()
                # 简单清洗：如果包含 "Paragraph 15"，尝试提取
                # 这里不做复杂清洗，交给 eval.py，但 Prompt 应该已经让输出很干净了
                
                json.dump({
                    "pred": pred,
                    "answers": example["answers"],
                    "all_classes": example["all_classes"],
                    "length": example["length"]
                }, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-len", type=int, default=65536)
    
    # Eviction args
    parser.add_argument("--enable-paged-eviction", action="store_true")
    parser.add_argument("--evict-method", type=str, default="global")
    #parser.add_argument("--evict-metric", type=str, default="value_l2")
    parser.add_argument("--cache-budget", type=int, default=1024)
    parser.add_argument("--initial-blocks", type=int, default=1)
    
    args = parser.parse_args()
    main(args)
