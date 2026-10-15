# create_eval_prompts_from_offset.py
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import math

# --- 配置 ---
TOKENIZER_PATH = "/home/liushenyu/opt-13b/"
INPUT_TEXT_FILE = "/home/liushenyu/test/c4_subset_1024mb.txt"  # 还是用完整文件
OUTPUT_JSONL_FILE = "c4_eval_prompts_from_offset.jsonl"

# --- 核心参数：定义从哪里开始作为评估数据 ---
TRAIN_TRAJECTORIES_COUNT = 8000  # 您已经生成的训练样本数
CHUNK_SIZE = 2048
START_LEN = 512
STEP = 16
SAMPLING_POINTS_PER_CHUNK = len(range(START_LEN, CHUNK_SIZE + 1, STEP))

# 计算训练时用了多少个文本块 (向上取整)
CHUNKS_USED_FOR_TRAINING = math.ceil(TRAIN_TRAJECTORIES_COUNT / SAMPLING_POINTS_PER_CHUNK)
START_TOKEN_OFFSET = CHUNKS_USED_FOR_TRAINING * CHUNK_SIZE

print(f"Training used {TRAIN_TRAJECTORIES_COUNT} trajectories.")
print(f"Sampling points per chunk: {SAMPLING_POINTS_PER_CHUNK}")
print(f"Estimated chunks used for training: {CHUNKS_USED_FOR_TRAINING}")
print(f"--- Will start creating evaluation prompts from token offset: {START_TOKEN_OFFSET} ---")

# --- 生成Prompt的配置 ---
PROMPT_TOKEN_LENGTH = 1024  # prompt长度
MAX_PROMPTS = 1000 # 生成的prompt数量上限

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)

    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    all_tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    
    num_prompts_generated = 0
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(total=min(MAX_PROMPTS, (len(all_tokens) - START_TOKEN_OFFSET) // PROMPT_TOKEN_LENGTH), desc="Creating Prompts")
        
        # 从计算出的偏移量开始
        for i in range(START_TOKEN_OFFSET, len(all_tokens) - PROMPT_TOKEN_LENGTH, PROMPT_TOKEN_LENGTH):
            if num_prompts_generated >= MAX_PROMPTS:
                break
            
            prompt_tokens = all_tokens[i : i + PROMPT_TOKEN_LENGTH]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            
            json_line = {"prompt": prompt_text}
            f_out.write(json.dumps(json_line) + '\n')
            
            num_prompts_generated += 1
            pbar.update(1)
        pbar.close()

    print(f"\nSuccessfully created {num_prompts_generated} prompts in {OUTPUT_JSONL_FILE}")

if __name__ == '__main__':
    main()