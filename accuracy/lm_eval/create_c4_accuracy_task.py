# create_c4_accuracy_task.py
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import math
import os

# --- 配置 ---
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = "/home/liushenyu/test/c4_subset_1024mb.txt"
OUTPUT_JSONL_FILE = "c4_accuracy_task.jsonl"

# --- 核心：从训练数据之后开始 ---
# (与之前脚本的计算逻辑保持一致)
TRAIN_TRAJECTORIES_COUNT = 8000
CHUNK_SIZE = 2048
START_LEN = 512
STEP = 16
SAMPLING_POINTS_PER_CHUNK = len(range(START_LEN, CHUNK_SIZE + 1, STEP))
CHUNKS_USED_FOR_TRAINING = math.ceil(TRAIN_TRAJECTORIES_COUNT / SAMPLING_POINTS_PER_CHUNK)
START_TOKEN_OFFSET = CHUNKS_USED_FOR_TRAINING * CHUNK_SIZE

print(f"--- Will start creating accuracy task from token offset: {START_TOKEN_OFFSET} ---")

# --- 任务定义 ---
CONTEXT_LENGTH = 128  # 上下文的token长度
COMPLETION_LENGTH = 128 # 我们要评估模型预测这部分内容的准确率
TOTAL_LENGTH = CONTEXT_LENGTH + COMPLETION_LENGTH
MAX_SAMPLES = 1000 # 生成的任务样本数量

def main():
    if os.path.exists(OUTPUT_JSONL_FILE):
        print(f"Output file {OUTPUT_JSONL_FILE} already exists. Skipping generation.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    
    num_samples_generated = 0
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(total=min(MAX_SAMPLES, (len(all_tokens) - START_TOKEN_OFFSET) // TOTAL_LENGTH), desc="Creating Accuracy Task")
        
        # 从计算出的偏移量开始，每次跳过一个完整的任务长度
        for i in range(START_TOKEN_OFFSET, len(all_tokens) - TOTAL_LENGTH, TOTAL_LENGTH):
            if num_samples_generated >= MAX_SAMPLES:
                break
            
            task_tokens = all_tokens[i : i + TOTAL_LENGTH]
            prompt_text = tokenizer.decode(task_tokens, skip_special_tokens=True)
            
            # 我们需要知道上下文有多少个token，以便之后进行评估
            # 重新编码以确保长度准确
            context_tokens_count = len(tokenizer.encode(tokenizer.decode(task_tokens[:CONTEXT_LENGTH]), add_special_tokens=False))

            # 创建JSON对象，包含完整的prompt和上下文长度元数据
            json_line = {
                "prompt": prompt_text,
                "metadata": {
                    "context_token_len": context_tokens_count
                }
            }
            f_out.write(json.dumps(json_line) + '\n')
            
            num_samples_generated += 1
            pbar.update(1)
        pbar.close()

    print(f"\nSuccessfully created {num_samples_generated} task samples in {OUTPUT_JSONL_FILE}")

if __name__ == '__main__':
    main()