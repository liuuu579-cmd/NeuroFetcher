# evaluate_c4_accuracy.py
import json
import numpy as np
import argparse
from tqdm import tqdm

def calculate_metrics(results_file):
    """
    读取Harness的输出文件并计算Perplexity和Top-1 Accuracy。
    """
    total_log_prob = 0
    total_eval_tokens = 0
    total_correct_predictions = 0
    
    all_results = []
    with open(results_file, 'r') as f:
        for line in f:
            all_results.append(json.loads(line))

    print(f"Loaded {len(all_results)} results from {results_file}")

    for item in tqdm(all_results, desc="Calculating Metrics"):
        request = item['request']
        result = item['result']

        # 从请求中获取上下文长度
        # 注意：这里我们假设request的结构与输入文件中的一致
        context_len = request['metadata']['context_token_len']
        
        logprobs_data = result['choices'][0]['logprobs']
        token_logprobs = logprobs_data['token_logprobs']
        top_logprobs = logprobs_data['top_logprobs']
        tokens = logprobs_data['tokens']
        
        # 评估区间是上下文之后的所有token
        # token_logprobs[0] is None, so we start from index 1, which corresponds to tokens[0]
        # if context_len is 128, we need to evaluate from token_logprobs[129] onwards
        eval_start_index = context_len
        
        # 遍历需要评估的token
        for i in range(eval_start_index, len(tokens)):
            # 确保我们有对应的logprob信息 (i+1因为logprobs列表前面有个None)
            if i < len(token_logprobs) - 1 and token_logprobs[i+1] is not None:
                # 累加log probability
                total_log_prob += token_logprobs[i+1]
                total_eval_tokens += 1
                
                # 检查Top-1准确率
                predicted_token = next(iter(top_logprobs[i+1].keys()))
                actual_token = tokens[i+1]
                
                if predicted_token == actual_token:
                    total_correct_predictions += 1

    # 计算最终指标
    mean_log_prob = total_log_prob / total_eval_tokens if total_eval_tokens > 0 else 0
    perplexity = np.exp(-mean_log_prob) if mean_log_prob != 0 else float('inf')
    top1_accuracy = total_correct_predictions / total_eval_tokens if total_eval_tokens > 0 else 0
    
    return {
        "perplexity": perplexity,
        "top1_accuracy": top1_accuracy,
        "total_eval_tokens": total_eval_tokens
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Perplexity and Accuracy on C4.")
    parser.add_argument('--results-file', type=str, required=True, help="Path to the .jsonl file from run_lm_eval_harness.py")
    args = parser.parse_args()
    
    metrics = calculate_metrics(args.results_file)
    
    print("\n--- Evaluation Results ---")
    print(f"File: {args.results_file}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4%}")
    print(f"Evaluated on {metrics['total_eval_tokens']} tokens.")