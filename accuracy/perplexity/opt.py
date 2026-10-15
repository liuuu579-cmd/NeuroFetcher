import torch
import torch.nn as nn
import copy
import argparse
import math
import os
import sys
import tqdm
import json

# 新增依赖
try:
    from datasets import load_dataset
except ImportError:
    print("请安装 datasets: pip install datasets")

# ================================================================
#  SECTION 1: 辅助函数 (融合了新旧版本)
# ================================================================

# ... [set_symlink 保持不变] ...
def set_symlink(model_type, method):
    """
    根据选择的方法创建指向正确模型实现文件的符号链接。
    """
    if method == 'neural':
        fname = f"modeling_{model_type}_neu.py"
    elif method == 'infinigen':
        fname = f"modeling_{model_type}_inf.py"
    else: # 'orig' for Full Cache and H2O
        fname = f"modeling_{model_type}_orig.py"

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.join(current_script_dir, "..")
    linker_path = os.path.join(project_root_dir, "src", fname)
    model_path = os.path.join(project_root_dir, "transformers/src/transformers/models/", model_type)

    if not os.path.exists(linker_path):
        print(f"错误：源文件不存在于 {linker_path}")
        exit(1)
    if not os.path.exists(model_path):
        print(f"错误：目标目录不存在于 {model_path}")
        exit(1)
        
    curr_dir = os.getcwd()
    os.chdir(model_path)

    target_file = f'modeling_{model_type}.py'
    if os.path.lexists(target_file):
        os.remove(target_file)
        
    print(f"创建符号链接: {target_file} -> {linker_path}")
    os.symlink(os.path.abspath(linker_path), target_file)
    os.chdir(curr_dir)


# ... [opt_eval 保持不变，用于计算 PPL] ...
@torch.no_grad()
def opt_eval(model, testenc, eval_samples, ours, print_chunk=False):
    print('正在使用 opt_eval (layerwise) 方法进行 PPL 评估...')
    # [这里保持你原有的 opt_eval 代码完全不变]
    # 为了节省篇幅，假设此处是你提供的原始 opt_eval 代码
    # ...
    dev = torch.device('cuda:0')
    if hasattr(model, 'predictor') and model.predictor is not None:
        model.predictor.to(dev)
        
    testenc = testenc.input_ids
    if eval_samples:
        nsamples = eval_samples
    else:
        nsamples = min(1000, testenc.numel() // model.seqlen)
    print("nsamples: ", nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    # ===== [新增]: 将 Neural Prefetcher 的自定义 Embedding 挪到 GPU =====
    if hasattr(model, 'predictor_embed_13b') and model.predictor_embed_13b is not None:
        model.predictor_embed_13b = model.predictor_embed_13b.to(dev)
    if hasattr(model.model.decoder, 'predictor_embed_13b') and model.model.decoder.predictor_embed_13b is not None:
        model.model.decoder.predictor_embed_13b = model.model.decoder.predictor_embed_13b.to(dev)
    # ====================================================================
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
     # ===== [新增]: 将 Neural Prefetcher 的自定义 Embedding 挪回 CPU =====
    if hasattr(model, 'predictor_embed_13b') and model.predictor_embed_13b is not None:
        model.predictor_embed_13b = model.predictor_embed_13b.cpu()
    if hasattr(model.model.decoder, 'predictor_embed_13b') and model.model.decoder.predictor_embed_13b is not None:
        model.model.decoder.predictor_embed_13b = model.model.decoder.predictor_embed_13b.cpu()
    # ====================================================================
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    prev_hidden = []

    # Predictor logic
    predictor_cache = [] 
    is_neural = hasattr(model, 'predictor') and model.predictor is not None
    if is_neural:
        # print("正在为 Neural Prefetching 预计算 Logits...")
        embed_tokens_tmp = model.model.decoder.embed_tokens.to(dev)
        for j in range(nsamples):
            batch_ids = testenc[:, (j * model.seqlen):((j + 1) * model.seqlen)].to(dev)
            with torch.no_grad():
                inputs_embeds = embed_tokens_tmp(batch_ids)
                logits = model.predictor(inputs_embeds.to(torch.float32))
                predictor_cache.append(logits)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()

    density = []
    # print("逐层处理...")
    
    prefetch_layers_list = getattr(model.config, "prefetch_layers_list", None)
    
    for i in tqdm.tqdm(range(len(layers)), desc="Layerwise Eval"):
        layer = layers[i].to(dev)
        if ours and hasattr(layer.self_attn, 'partial_weight_q'):
            layer.self_attn.partial_weight_q = layer.self_attn.partial_weight_q.to(dev)
            
        for j in range(nsamples):
            if ours and not is_neural: 
                if i >= 2:
                    layer.self_attn.previous_hidden_states = prev_hidden[j]
            
            current_layer_logits = None
            if is_neural:
                sample_logits = predictor_cache[j] 
                num_pred_layers = sample_logits.shape[1]
                if i < num_pred_layers:
                    if prefetch_layers_list is None or i in prefetch_layers_list:
                        current_layer_logits = sample_logits[:, i, :]
                else:
                    current_layer_logits = sample_logits[:, num_pred_layers - 1, :]
            
            forward_kwargs = {'attention_mask': attention_mask}
            if current_layer_logits is not None:
                forward_kwargs['predictor_logits'] = current_layer_logits

            outs[j] = layer(inps[j].unsqueeze(0), **forward_kwargs)[0]

            if ours and not is_neural:
                if i >= 1:
                    cur_bsz = layer.self_attn.current_hidden_states.shape[0]
                    cur_tgt_len = layer.self_attn.current_hidden_states.shape[1]
                    cur_device = layer.self_attn.current_hidden_states.device
                    cur_dtype = layer.self_attn.current_hidden_states.dtype
                    if i == 1:
                        prev_hidden.append(torch.cat((layer.self_attn.current_hidden_states, torch.ones(cur_bsz, cur_tgt_len, 1).to(cur_device).to(cur_dtype)), dim = -1))
                    else:
                        prev_hidden[j] = torch.cat((layer.self_attn.current_hidden_states, torch.ones(cur_bsz, cur_tgt_len, 1).to(cur_device).to(cur_dtype)), dim = -1)

            if ours and hasattr(layer.self_attn, 'density') and layer.self_attn.density is not None:
                density.append(layer.self_attn.density)

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if ours and density:
        density_val = sum(density) / len(density) * 100
        print(f"InfiniGen/Neural 平均密度: {density_val:.2f}%")

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)
    
    # === 【关键修改点开始】 ===
    # 错误代码: testenc = testenc.to(dev)  <-- 这行代码会导致 arXiv OOM
    # 修正: 保持 testenc 在 CPU，下面循环里再搬运
    print("开始计算最终 PPL (Logits)...")

    nlls = []
    # 进度条有助于观察大文件的处理进度
    for i in tqdm.tqdm(range(nsamples), desc="Calc PPL"):
        # 1. 动态将当前样本搬到 GPU，而不是全部
        batch_input = inps[i].unsqueeze(0).to(dev)
        
        hidden_states = batch_input
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        
        lm_logits = model.lm_head(hidden_states)
        
        # shift logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        
        # 2. 动态获取对应的 label 并搬到 GPU
        sample_ids = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        shift_labels = sample_ids[:, 1:].to(dev) # 只把这一小段搬到 GPU
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(torch.float)
        nlls.append(neg_log_likelihood)
        
        # 及时清理不需要的显存
        del hidden_states, lm_logits, shift_logits, shift_labels, batch_input
    # === 【关键修改点结束】 ===

    nlls = torch.stack(nlls)
    if print_chunk:
        print("\n--- 分块 Perplexity ---")
        for chunk in range(int(model.seqlen / 256)):
            start = chunk * 256
            end = (chunk + 1) * 256
            ppl = torch.exp(nlls[:, start:end].sum() / (nsamples * 256))
            print(f"块 (位置 {start}-{end}): Perplexity = {ppl.item():.4f}")

    print("\n--- 最终结果 ---")
    ppl = torch.exp(nlls.sum() / (nsamples * model.seqlen))
    print(f"Dataset: PPL = {ppl.item():.4f}")
    model.config.use_cache = use_cache


# ================================================================
#  SECTION 1.5: 新增 - arXiv 数据加载与 HumanEval 评估
# ================================================================
def get_pg19_loader(model_name, seq_len):
    """
    [修复版] 手动下载 Project Gutenberg 长篇书籍替代 PG-19
    绕过 Hugging Face 的脚本错误，直接获取文学长文本用于测试
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import os
    import requests

    print("正在处理 PG-19 替代数据 (手动下载长篇小说)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 定义两本经典的超长篇小说 URL (来自 Project Gutenberg)
    # 1. Moby Dick (白鲸记) - 约 20万词
    # 2. Frankenstein (弗兰肯斯坦) - 约 7万词
    # 这两本书加起来足够测试 Long Context PPL
    books = [
        {"name": "Moby_Dick", "url": "https://www.gutenberg.org/files/2701/2701-0.txt"},
        {"name": "Frankenstein", "url": "https://www.gutenberg.org/files/84/84-0.txt"}
    ]
    
    local_file = "pg19_proxy_test.txt"

    # 1. 下载并合并书籍
    if not os.path.exists(local_file):
        print(f"本地未找到 {local_file}，正在从 Project Gutenberg 下载...")
        with open(local_file, 'w', encoding='utf-8') as outfile:
            for book in books:
                print(f"正在下载: {book['name']}...")
                try:
                    # 使用 User-Agent 防止被某些服务器拦截
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    response = requests.get(book['url'], headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    # 简单的清洗：去掉开头的 License 说明（通常 Project Gutenberg 书籍开头有大量声明）
                    text = response.text
                    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
                    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
                    
                    start_idx = text.find(start_marker)
                    if start_idx != -1:
                        # 找到换行后的正文开始
                        text = text[start_idx:]
                        # 再找下一行
                        text = text[text.find('\n', 10):] 
                    
                    # 写入文件
                    outfile.write(text)
                    outfile.write("\n\n" * 10) # 书籍之间留大空隙
                    
                except Exception as e:
                    print(f"下载 {book['name']} 失败: {e}")
                    print("如果网络不通，请手动下载 txt 文件并合并保存为 pg19_proxy_test.txt")
                    exit(1)
        print("下载完成。")
    else:
        print(f"发现本地文件 {local_file}，直接使用。")

    # 2. 加载文本
    print("正在加载文本...")
    try:
        # 使用 'text' builder 加载本地文件，安全且无需脚本
        dataset = load_dataset("text", data_files={"test": local_file}, split="test")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        exit(1)

    # 3. 拼接与 Tokenize
    print("正在拼接书籍文本...")
    text_list = dataset['text']
    full_text = "\n".join(text_list)
    
    print("正在分词 (Tokenization)...")
    enc = tokenizer(full_text, return_tensors='pt')
    print(f"PG-19 (Proxy) 总 Token 数: {enc.input_ids.shape[1]}")
    
    return enc
def get_ptb_loader(model_name, seq_len):
    """
    [修复版] 手动下载 PTB 原始文本，绕过 Hugging Face 的脚本限制
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import os
    import requests

    print("正在处理 PTB 数据集 (手动模式)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # PTB 测试集的原始下载地址 (来自 Tom Sercu/Mikolov 的标准仓库)
    url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
    file_path = "ptb.test.txt"

    # 1. 检查并下载文件
    if not os.path.exists(file_path):
        print(f"本地未找到 {file_path}，正在从 GitHub 下载...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status() # 检查请求是否成功
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("下载成功。")
        except Exception as e:
            print(f"下载 PTB 失败: {e}")
            print("请尝试手动下载该文件: https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt")
            print("并将其保存为当前目录下的 'ptb.test.txt'")
            exit(1)
    else:
        print(f"发现本地文件 {file_path}，直接使用。")

    # 2. 加载本地文本文件
    # 使用 'text' builder，这是 HF 原生支持的，绝对不会报错
    print("正在加载文本...")
    try:
        dataset = load_dataset("text", data_files={"test": file_path}, split="test")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        exit(1)

    # 3. 拼接与 Tokenize
    # PTB 的原始格式是一行一个句子
    print("正在拼接 PTB 文本...")
    # 注意：load_dataset('text') 返回的列名默认为 'text'，而不是旧版的 'sentence'
    text_list = dataset['text']
    
    # PTB 的标准处理方式通常是使用换行符连接，或者直接连起来
    # 这里我们使用换行符，因为文件本身是一行一句
    full_text = "\n".join(text_list)
    
    # 编码
    print("正在分词 (Tokenization)...")
    enc = tokenizer(full_text, return_tensors='pt')
    print(f"PTB 总 Token 数: {enc.input_ids.shape[1]}")
    
    return enc 
def get_arxiv_loader(model_name, seq_len):
    """
    加载 arXiv 数据集用于 PPL 评估
    """
    from transformers import AutoTokenizer
    print("正在加载 arXiv 数据集 (ccdv/arxiv-summarization)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 使用 HuggingFace Datasets 加载
    # 注意：arXiv 很大，这里只加载 test split 或者流式加载
    try:
        # 尝试加载一个常用的 arXiv 子集
        dataset = load_dataset("ccdv/arxiv-summarization", split="test")
    except Exception as e:
        print(f"加载 arXiv 数据集失败: {e}")
        print("尝试使用本地文件或请检查网络连接。")
        exit(1)

    print("正在对 arXiv 数据进行分词和拼接...")
    # 提取文章内容
    text = "\n\n".join(dataset['article'][:1000]) # 取前1000篇以防内存爆炸，可按需调整
    
    enc = tokenizer(text, return_tensors='pt')
    print(f"总 Token 数: {enc.input_ids.shape[1]}")
    
    # 确保能被 seq_len 整除 (简单的 drop last 处理)
    # 实际上 opt_eval 会自己处理切片，这里只需要返回 Encoded 对象
    return enc
def get_local_c4_loader(model_name, seq_len):
    """
    直接加载本地的 C4 子集 txt 文件
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import os

    file_path = "/home/liushenyu/test/c4_subset_50mb.txt"
    print(f"正在处理本地 C4 数据集: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误: 在 {file_path} 找不到文件，请检查路径。")
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print("正在加载文本...")
    try:
        # 使用 datasets 库的原生本地文本加载功能
        dataset = load_dataset("text", data_files={"test": file_path}, split="test")
    except Exception as e:
        print(f"加载本地数据集失败: {e}")
        exit(1)

    print("正在拼接 C4 文本...")
    text_list = dataset['text']
    # 用换行符将所有的行拼接起来
    full_text = "\n".join(text_list)
    
    print("正在分词 (Tokenization)... (50MB文本可能需要几十秒)")
    enc = tokenizer(full_text, return_tensors='pt')
    print(f"C4 子集总 Token 数: {enc.input_ids.shape[1]}")
    
    return enc
def eval_humaneval(model, tokenizer, k=[1], num_samples_per_task=1):
    """
    执行 HumanEval 数据集的 Pass@1 评估 (修复版)
    """
    print("\n=== 开始 HumanEval (Pass@1) 评估 ===")
    try:
        from human_eval.data import write_jsonl, read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("错误: 缺少 human-eval 库。")
        return

    problems = read_problems()
    samples = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将模型移动到 {device} 进行生成...")
    model.to(device)

    # 定义停止词，防止模型生成多余内容导致语法错误
    # OPT 可能会生成 "\nclass", "\ndef", "\nif" 等新开头，我们应该在那之前截断
    stop_words = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    
    pbar = tqdm.tqdm(problems.items(), desc="Generating HumanEval")
    
    for task_id, problem in pbar:
        prompt = problem["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = {
            "do_sample": False, # Pass@1 使用 Greedy Search 
            "max_new_tokens": 512,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                **gen_kwargs
            )
        
        # === 关键修复：精确去除 Prompt ===
        # 只解码新生成的 token
        generated_ids = outputs[0][input_len:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # === 关键修复：手动截断 ===
        # 如果模型开始写新的函数或类，强制截断，保留第一个函数体
        for stop_word in stop_words:
            if stop_word in generated_code:
                generated_code = generated_code.split(stop_word)[0]

        # 组合结果 (HumanEval 有时需要 prompt + completion，有时只需要 completion，取决于具体的评测实现)
        # 标准 HumanEval 评测只需要 completion 部分，只要它是合法的后续代码
        samples.append(dict(task_id=task_id, completion=generated_code))

    output_file = "humaneval_samples.jsonl"
    write_jsonl(output_file, samples)
    print(f"生成完成，结果已保存至 {output_file}")

    print("正在计算 Pass@1 指标...")
    # 注意：在本地运行 generated code 有风险
    results = evaluate_functional_correctness(output_file, k=k, n_workers=1, timeout=5.0)
    print("========================================")
    print(f"HumanEval Results: {results}")
    print("========================================")


# ================================================================
#  SECTION 2: 主逻辑
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 通用模型和数据集参数 ---
    parser.add_argument('--model', type=str, required=True, help="模型路径")
    parser.add_argument('--eval_dataset', type=str, required=True, help="评估数据集: 'wikitext2', 'ptb', 'c4', 'arxiv', 'humaneval'")
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--eval_samples', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b')
    parser.add_argument('--print_blk_ppl', action='store_true')
    parser.add_argument('--model-type', type=str, default='opt')

    # --- H2O 参数 ---
    parser.add_argument('--enable-small-cache', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    # --- InfiniGen & Neural 共享参数 ---
    parser.add_argument('--ours', action='store_true')
    parser.add_argument('--strategy', type=str, default='infinigen', choices=['infinigen', 'neural'])

    # --- InfiniGen 特定参数 ---
    infinigen_group = parser.add_argument_group('InfiniGen Strategy Arguments')
    infinigen_group.add_argument("--partial_weight_ratio", type=float, default=0.1)
    infinigen_group.add_argument("--partial_weight_path", type=str, default=None)
    infinigen_group.add_argument("--alpha", type=float, default=5.0)
    infinigen_group.add_argument("--capacity", type=float, default=1.0)
    infinigen_group.add_argument("--budget", type=float, default=0.2)
    infinigen_group.add_argument("--eviction_policy", type=str, default="lru")

    # --- Neural Prefetching 特定参数 ---
    neural_group = parser.add_argument_group('Neural Prefetcher Arguments')
    neural_group.add_argument('--predictor-path', type=str, default=None)
    neural_group.add_argument('--prefetch-budget-ratio', type=float, default=0.3)

    args = parser.parse_args()

    # 1. 设置符号链接
    if args.ours:
        print(f"启用自定义模型，策略: {args.strategy.upper()}")
        set_symlink(args.model_type, args.strategy)
    else:
        print("使用原始模型实现 (Full Cache or H2O)")
        set_symlink(args.model_type, 'orig')

    from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer

    # 2. 加载配置
    print("加载模型配置...")
    config = AutoConfig.from_pretrained(args.model_name)
    
    if args.ours and args.strategy == 'neural':
        print("[INFO] 为 Neural Prefetching 策略配置模型...")
        if not args.predictor_path:
            raise ValueError("--predictor-path 是 'neural' 策略的必需参数。")
        config.prefetch_strategy = 'neural'
        config.prefetch_budget_ratio = args.prefetch_budget_ratio
        config.num_predict_layers = 5 
        config.predictor_nhead = 4
        config.predictor_encoder_layers = 1
        config.predictor_d_hid = 2048
        config.prefetch_layers_list = [0,1,2,3,4]
        print(f"  - 预取预算比例: {config.prefetch_budget_ratio * 100:.1f}%")

    # 3. 加载模型
    print(f"加载模型: {args.model}")
    model = OPTForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map='cpu', # opt_eval 会手动管理设备，HumanEval 时我们会手动挪到 GPU
        config=config, 
        ignore_mismatched_sizes=True if args.ours and args.strategy == 'neural' else False
    )
    model.seqlen = args.seq_len
    
    # 4. 策略特定配置
    if args.enable_small_cache:
        print('\n--- 配置 H2O (Small Cache Size) ---')
        sys.path.append("../lm_eval")
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
        model.config.heavy_ratio = args.heavy_ratio
        model.config.recent_ratio = args.recent_ratio
        model = convert_kvcache_opt_heavy_recent(model, model.config)
        model = model.to(torch.float16)
        print(f"H2O 配置完成")

    elif args.ours:
        if args.strategy == 'infinigen':
            print("\n--- 配置 InfiniGen 模型层 ---")
            for layer in model.model.decoder.layers:
                layer.self_attn.partial_weight_ratio = args.partial_weight_ratio
                
                # === [修改开始] 确保加载后的权重在正确的设备上 ===
                # 读取权重
                pw_q = torch.load(os.path.join(args.partial_weight_path, f"partial_weight_q_{layer.self_attn.layer_idx}.pt"), map_location='cpu')
                
                # 如果是运行 HumanEval，模型之后会被移到 GPU，这里我们需要确保权重是 nn.Parameter
                # 或者手动管理。最稳妥的方法是将其注册为 buffer，这样 model.to(device) 会自动处理它。
                if not hasattr(layer.self_attn, 'partial_weight_q'):
                    # 注册为 buffer，这样调用 model.to(device) 时它会自动移动
                    layer.self_attn.register_buffer('partial_weight_q', pw_q)
                else:
                    # 如果已经存在（可能是 None），则赋值并确保它是 Tensor
                    layer.self_attn.partial_weight_q = pw_q
                
                # 如果当前是在运行 opt_eval (手动分层)，可能需要保持在 CPU 直到手动 .to(dev)
                # 但对于 model.generate，必须跟随模型。
                # 为了兼容两者，register_buffer 是最佳实践。
                # === [修改结束] ===

                layer.self_attn.alpha = args.alpha
                layer.self_attn.capacity = args.capacity
                layer.self_attn.budget = args.budget
                layer.self_attn.eviction_policy = args.eviction_policy
            print(f"InfiniGen 配置完成")

        elif args.strategy == 'neural':
            print("\n--- 配置神经网络预取器 ---")
            if not hasattr(model, 'predictor') or model.predictor is None:
                raise TypeError("加载的模型没有 'predictor' 属性。")
            print(f"加载预测器权重: {args.predictor_path}")
            predictor_state_dict = torch.load(args.predictor_path, map_location='cpu')
            model.predictor.load_state_dict(predictor_state_dict)
            model.predictor.eval()
            model.predictor.to(torch.float32)
            print("神经网络预测器配置成功。")
    
    model.eval()

    
    # 5. 根据数据集分流逻辑
    if args.eval_dataset == 'humaneval':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        eval_humaneval(model, tokenizer)
        
    else:
        # PPL 逻辑 (opt_eval)
        if args.eval_dataset == 'arxiv':
            testloader = get_arxiv_loader(args.model_name, model.seqlen)
        
        # === [新增] PG-19 分支 ===
        elif args.eval_dataset == 'pg19':
            testloader = get_pg19_loader(args.model_name, model.seqlen)
        # ========================
        elif args.eval_dataset == 'ptb':
            testloader = get_ptb_loader(args.model_name, model.seqlen)
        elif args.eval_dataset == 'c4':
            testloader = get_local_c4_loader(args.model_name, model.seqlen)
        
        else:
            # 原有的 WikiText2 / PTB 加载逻辑
            from datautils import get_loaders
            print(f"\n为 '{args.eval_dataset}' 数据集加载数据 (Legacy)...")
            testloader = get_loaders(args.eval_dataset, model=args.model_name, seqlen=model.seqlen)

        is_infinigen_for_eval = args.ours and args.strategy == 'infinigen'
        # 建议在跑 PG-19 时加上 print_chunk 看看分块效果
        opt_eval(model, testloader, args.eval_samples, is_infinigen_for_eval, args.print_blk_ppl)