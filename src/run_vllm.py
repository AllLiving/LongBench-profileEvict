import argparse
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# LongBench 标准数据集列表
LONGBENCH_DATASETS = [
    "narrativeqa", "qasper", "wikidump", "gov_report", 
    "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

# LongBench 官方 Prompt 模板 (简化版，涵盖了常用数据集)
dataset2prompt = {
    "narrativeqa": "Read the following story and answer the question. Story: {context}\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article. Write an answer to the question based on the article. Article: {context}\n\nQuestion: {input}\n\nAnswer:",
    "wikidump": "Read the following context and answer the question. Context: {context}\n\nQuestion: {input}\n\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report. Report: {context}\n\nSummary:",
    "qmsum": "You are given a meeting transcript. Write a summary of the meeting. Meeting Transcript: {context}\n\nSummary:",
    "multi_news": "You are given several news excerpts. Write a summary of the news. News: {context}\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions and their types:\n{context}\nQuestion: {input}\nType:",
    "triviaqa": "Answer the question based on the given context. Context: {context}\n\nQuestion: {input}\n\nAnswer:",
    "samsum": "Summarize the dialogue into a few short sentences. The following is the dialogue.\n\n{context}\n\nSummary:",
    "passage_count": "There are some paragraphs below numbered 1, 2, 3... Read the paragraphs and answer the question. \n\n{context}\n\nQuestion: {input}\n\nAnswer:",
    "passage_retrieval_en": "Here are 30 paragraphs. \n\n{context}\n\nQuestion: {input}\n\nAnswer:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}Next line of code:\n"
}

# 对应不同任务的最大生成长度 (参考 LongBench 官方配置)
MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "wikidump": 128, "gov_report": 512,
    "qmsum": 512, "multi_news": 512, "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32, "lcc": 32, "repobench-p": 64
}

def format_llama3_chat(user_content):
    """
    将构建好的 LongBench Prompt 包装进 Llama-3 的 Chat 格式中。
    """
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant. Please follow the user's instructions carefully.<|eot_id|>" # System Prompt 稍微通用一点
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def get_pred(llm, examples, max_tokens, dataset_name):
    input_prompts = []
    
    # 1. 获取该数据集对应的 LongBench 原始模板
    # 如果找不到，兜底使用简单的 Context+Input 格式
    base_prompt_fmt = dataset2prompt.get(dataset_name, "{context}\n\n{input}")
    
    for example in examples:
        # 2. 第一步：构建 LongBench 原始 Prompt (包含任务指令)
        # 注意：像 samsum 这种数据集，example['input'] 可能是空的，这没关系，模板里只用了 {context}
        try:
            raw_prompt = base_prompt_fmt.format(context=example['context'], input=example['input'])
        except KeyError:
            # 防止部分数据集字段缺失导致的报错
            raw_prompt = f"{example['context']}\n\n{example.get('input', '')}"

        # 3. 第二步：套上 Llama-3 的马甲
        # 这一步是修复 "Garbage Output" 的关键
        final_prompt = format_llama3_chat(raw_prompt)
        
        input_prompts.append(final_prompt)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=False
    )
    if len(input_prompts) > 0:
        print(f"\n[DEBUG PROMPT SAMPLE] First 300 chars:\n{input_prompts[0][:300]}...")
        print(f"[DEBUG PROMPT SAMPLE] Last 300 chars:\n...{input_prompts[0][-300:]}\n")

    # 4. 执行生成
    if len(input_prompts) > 0:
        print(f"[src/run_vllm] Approx Prompt Length (Chars): {len(input_prompts[0])}")
    outputs = llm.generate(input_prompts, sampling_params)
    preds = [output.outputs[0].text.strip() for output in outputs]
    return preds

def run_eval(args):
    # 1. 准备 vLLM 参数
    print(f"--- Loading Model: {args.model} ---")
    print(f"--- PagedEviction: {args.enable_paged_eviction} | Metric: {args.evict_metric} | Budget: {args.cache_budget} ---")
    
    engine_args = {
        "model": args.model,
        "trust_remote_code": True,
        # "gpu_memory_utilization": 0.65,
        "gpu_memory_utilization": 0.9,
        # 关键点：让 vLLM 认为它能处理超长文本 (例如 32k)，
        # 实际显存占用由你的 cache_budget (例如 4k) 物理限制。
        # "max_model_len": 32000,
        "max_model_len": 65536,
        # "max_num_batched_tokens": 32000,
        "max_num_batched_tokens": 65536,
        "enforce_eager": True,
        "dtype": "auto",
        "enable_chunked_prefill": False, # 必须关闭，因为你的 Pruner 目前不支持 chunked
    }

    if args.enable_paged_eviction:
        engine_args.update({
            "enable_paged_eviction": True,
            "evict_method": args.evict_method,
            "evict_metric": args.evict_metric,
            "cache_budget": args.cache_budget,
            "initial_blocks": args.initial_blocks
        })
    print(f"\n[DEBUG CONFIG] Initial Blocks: {args.initial_blocks} (Should be ~16)")
    print(f"[DEBUG CONFIG] Cache Budget: {args.cache_budget}")

    llm = LLM(**engine_args)

    # 2. 遍历数据集
    datasets = args.datasets if args.datasets else LONGBENCH_DATASETS
    
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        # 加载数据 (THUDM/LongBench)
        data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        
        # 限制测试数量 (用于快速 Debug)
        if args.max_samples > 0:
            data = data.select(range(min(len(data), args.max_samples)))

        max_new_tokens = MAX_NEW_TOKENS.get(dataset_name, 128)
        
        preds = get_pred(llm, data, max_new_tokens, dataset_name)
        
        # 3. 保存结果
        output_file = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for example, pred in zip(data, preds):
                json.dump({
                    "pred": pred, 
                    "answers": example["answers"], 
                    "all_classes": example["all_classes"], 
                    "length": example["length"]
                }, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/Llama-3.2-1B-Instruct")
    parser.add_argument("--datasets", nargs="+", help="List of datasets to run")
    parser.add_argument("--output-dir", type=str, default="pred_e/paged_eviction")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max samples per dataset for debugging")
    
    # Eviction Params
    parser.add_argument("--enable-paged-eviction", action="store_true")
    parser.add_argument("--evict-method", type=str, default="global")
    parser.add_argument("--evict-metric", type=str, default="value_l2")
    # parser.add_argument("--cache-budget", type=int, default=1024)
    parser.add_argument("--cache-budget", type=int, default=4096)
    parser.add_argument("--initial-blocks", type=int, default=1)
    
    args = parser.parse_args()
    run_eval(args)
