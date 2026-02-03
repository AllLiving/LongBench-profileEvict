import os
import json
import argparse
import numpy as np

# 因为 metrics.py 现在就在同级目录 src/ 下，所以可以直接导入
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # 新增 input-dir 参数，用于接收 run_eval.sh 传过来的路径
    parser.add_argument('--input-dir', type=str, required=True, help="Path to the prediction directory")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "lsht"]: # No samsum
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "lsht"]: # No samsum
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    
    # 使用传入的 input-dir，不再硬编码路径
    path = args.input_dir
    if not path.endswith('/'):
        path += '/'
        
    all_files = os.listdir(path)
    print(f"Evaluating in: {path}")
    print("Files found:", all_files)
    
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    
    # 将评分结果 result.json 保存回该目录
    out_path = os.path.join(path, "result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"Evaluation finished. Results saved to {out_path}")

    # 2. [新增] 直接在终端打印格式化的结果
    print("\n" + "="*40)
    print(f"📊 Evaluation Results ({path}):")
    print("="*40)
    # 使用 indent=4 让 JSON 格式化输出，易于阅读
    print(json.dumps(scores, ensure_ascii=False, indent=4))
    print("="*40)
    print(f"✅ Results also saved to: {out_path}")
