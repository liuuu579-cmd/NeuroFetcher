import argparse
import json
import tqdm
import torch
import copy
import os
import sys
import math
import time

def set_symlink(model_type, method):
    """
    根据选择的方法创建指向正确模型实现文件的符号链接。
    - method 'neural': 链接 modeling_{model_type}_neu.py
    - method 'infinigen': 链接 modeling_{model_type}_inf.py
    - method 'orig': 链接 modeling_{model_type}_orig.py (用于 full cache 和 H2O)
    """
    if method == 'neural':
        fname = f"modeling_{model_type}_neu.py"
    elif method == 'infinigen':
        #fname = f"modeling_{model_type}_inf.py"
        fname = f"modeling_opt_ours_v1.py"
    else: # 'orig'
        fname = f"modeling_{model_type}_orig.py"

    # 使用相对路径构建更稳健的路径
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 'src' 或类似目录下，需要根据你的项目结构调整 '..' 的数量
    base_project_dir = os.path.join(src_dir, "..") 
    model_path = os.path.join(base_project_dir, "transformers/src/transformers/models/", model_type)
    linker_path = os.path.realpath(os.path.join(base_project_dir, "src/", fname))

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
    os.symlink(linker_path, target_file)
    os.chdir(curr_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Unified Language Model Evaluation Runner',
        description='Run language model evaluation with various KV cache strategies.'
    )

    # --- 通用参数 (来自原版) ---
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='opt')

    # --- 量化参数 (来自原版) ---
    parser.add_argument('--enable-quant', action='store_true')
    parser.add_argument("--qbits", type=int, default=8)

    # --- H2O 参数 (来自原版) ---
    parser.add_argument('--enable-small-cache', action='store_true', help="启用 H2O (heavy-hitter) 缓存策略")
    parser.add_argument("--heavy-ratio", type=float, default=0.1)
    parser.add_argument("--recent-ratio", type=float, default=0.1)

    # --- InfiniGen & Neural 共享参数 ---
    parser.add_argument('--ours', action='store_true', help="启用自定义模型 (InfiniGen 或 Neural)")
    
    # --- 新增: 策略选择参数 (仅当 --ours 启用时有效) ---
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='infinigen', 
        choices=['infinigen', 'neural'],
        help="当启用 '--ours' 时，选择具体策略。默认为 'infinigen' 以保持向后兼容。"
    )

    # --- InfiniGen 特定参数 (来自原版) ---
    infinigen_group = parser.add_argument_group('InfiniGen Strategy Arguments')
    infinigen_group.add_argument("--partial-weight-ratio", type=float, default=0.1)
    infinigen_group.add_argument("--partial-weight-path", type=str)
    infinigen_group.add_argument("--skewing-matrix-path", type=str)
    infinigen_group.add_argument("--alpha", type=int, default=5)
    infinigen_group.add_argument("--capacity", type=float, default=1.0)
    infinigen_group.add_argument("--budget", type=float, default=0.2)
    infinigen_group.add_argument("--prefetch-start-layer",type=int, default=1)
    infinigen_group.add_argument("--prefetch-interval",type=int, default=1)

    # --- Neural Prefetching 特定参数 (来自神经网络版) ---
    neural_group = parser.add_argument_group('Neural Prefetcher Arguments')
    neural_group.add_argument('--predictor-path', type=str, default=None, help="[Neural] 神经网络预测器的权重路径")
    neural_group.add_argument('--prefetch-budget-ratio', type=float, default=0.3, help="[Neural] 预取预算比例")

    args = parser.parse_args()
    
    # 1. 根据策略设置符号链接
    if args.ours:
        print(f"启用自定义模型，策略: {args.strategy.upper()}")
        # 当 --ours 启用时，根据 --strategy 链接 'neu' 或 'inf' 文件
        set_symlink(args.model_type, args.strategy)
    else:
        # 当 --ours 未启用时 (full cache 或 H2O)，链接 'orig' 文件
        set_symlink(args.model_type, 'orig')

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # 2. 加载和配置模型
    print("加载模型配置...")
    config = AutoConfig.from_pretrained(args.model_name)

    # 如果是神经网络策略，需要修改config以初始化模型内的预测器结构
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
        print(f"  - Budget Ratio: {config.prefetch_budget_ratio * 100:.1f}%")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, device_map='auto', torch_dtype=torch.float16)
    
    print("\n加载模型...")
    model_load_path = args.model_path if args.model_path else args.model_name
    ignore_mismatched = True if args.ours and args.strategy == 'neural' else False
    
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, 
        device_map={"": 0}, 
        torch_dtype=torch.float16, 
        config=config,
        ignore_mismatched_sizes=ignore_mismatched
    )

    # 3. 模型加载后的特定修改 (保持原版逻辑结构)
    if args.enable_quant:
        print(f"启用 {args.qbits}-bit 量化...")
        model_layers = model.model.decoder.layers if args.model_type == "opt" else model.model.layers
        for i, layer in enumerate(model_layers):
            if i >= 2:
                layer.self_attn.enable_quant = True
                layer.self_attn.qbits = args.qbits

    elif args.enable_small_cache:
        print('启用 H2O (Small Cache Size)...')
        from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
        from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
        
        ENABLE_Heavy_Hitter_FUNCTIONS = {
            "llama": convert_kvcache_llama_heavy_recent,
            "opt": convert_kvcache_opt_heavy_recent,
            "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
        }
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        clean_path = args.model_name.rstrip(os.sep) 
        base_path = os.path.basename(clean_path)
        h2o_model_dir = "../h2o_model"
        os.makedirs(h2o_model_dir, exist_ok=True)
        h2o_checkpoint_path = os.path.join(h2o_model_dir, f"{base_path}.pt")

        if not os.path.exists(h2o_checkpoint_path):
            checkpoint = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(checkpoint, h2o_checkpoint_path)
        
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        model.load_state_dict(torch.load(h2o_checkpoint_path))
        model = model.to('cuda')        

    elif args.ours:
        model_layers = model.model.decoder.layers if args.model_type == "opt" else model.model.layers
        
        if args.strategy == 'infinigen':
            print("\n--- 配置 InfiniGen 模型层 ---")
            if args.model_type == "llama" and args.skewing_matrix_path:
                A = torch.load(args.skewing_matrix_path)
            
            for i, layer in enumerate(model_layers):
                layer.self_attn.partial_weight_ratio = args.partial_weight_ratio
                layer.self_attn.partial_weight_q = torch.load(os.path.join(args.partial_weight_path, f"partial_weight_q_{i}.pt"))
                layer.self_attn.alpha = args.alpha
                layer.self_attn.capacity = args.capacity
                layer.self_attn.budget = args.budget
                if args.model_type == "llama" and args.skewing_matrix_path:
                    layer.self_attn.skewing_matrix = A[i]

            # InfiniGen的隔层预取调度
            model.config.prefetch_start_layer = args.prefetch_start_layer
            model.config.prefetch_interval = args.prefetch_interval
            print(f"[DEBUG] Config Updated -> Start: {model.config.prefetch_start_layer}, Interval: {model.config.prefetch_interval}")

        elif args.strategy == 'neural':
            print("\n--- 配置神经网络预取器 ---")
            if not hasattr(model, 'predictor') or model.predictor is None:
                raise TypeError("加载的模型没有 'predictor' 属性。请确保链接了正确的 'neu' 模型文件。")
            
            print(f"从以下路径加载神经网络预测器权重: {args.predictor_path}")
            predictor_state_dict = torch.load(args.predictor_path, map_location='cpu')
            model.predictor.load_state_dict(predictor_state_dict)
            model.predictor.eval()
            print("神经网络预测器配置并加载成功。")

    # 4. 模型评估
    model.eval()
    if args.ours and args.strategy == 'neural':
        model.predictor.to(model.device, dtype=torch.float32)

    requests = []
    with open(args.input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results, density = [], []
    print("\n开始评估循环...")
    start_time = time.time()
    
    with torch.no_grad():
        for request in tqdm.tqdm(requests, desc="处理请求"):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            logits = model(input_ids).logits.log_softmax(dim=-1)
            
            # 仅在InfinGen模式下计算密度
            if args.ours and args.strategy == 'infinigen':
                density.append(model.get_density())

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:]
            logprobs = [None] + torch.gather(logits[:, :-1, :], -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1)[:-1], indices.squeeze(-1)[:-1])]
            
            result['result'] = {
                "choices": [{"text": prompt, "logprobs": {"tokens": tokens, "token_logprobs": logprobs, "top_logprobs": top_logprobs, "text_offset": []}, "finish_reason": "length"}],
                "request_time": {"batch_time": 0, "batch_size": 1}
            }
            results.append(result)
            
            # 仅在InfinGen模式下重置状态
            if args.ours and args.strategy == 'infinigen':
                model_layers = model.model.decoder.layers if args.model_type == "opt" else model.model.layers
                for layer in model_layers:
                    if hasattr(layer.self_attn, 'previous_hidden_states'):
                        layer.self_attn.previous_hidden_states = None

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总执行时间: {elapsed_time:.2f} 秒，处理了 {len(requests)} 个请求。")
    print(f"吞吐量: {len(requests) / elapsed_time:.2f} 请求/秒。")

    # 5. 结果后处理和保存
    if args.ours and args.strategy == 'infinigen':
        avg_density = sum(density) / len(density) * 100
        retain_ratio = (1 - math.sqrt(1 - avg_density / 100)) * 100
        print(f"\n[InfiniGen] 平均密度: {avg_density:.2f}%")
        print(f"[InfiniGen] 保留率: {retain_ratio:.2f}%\n")

    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
    print(f"结果已保存到 {args.output_path}")