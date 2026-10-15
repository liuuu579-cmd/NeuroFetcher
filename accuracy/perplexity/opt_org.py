import torch
import torch.nn as nn
import copy
import argparse
import math
import os
import sys


# ================================================================
#  SECTION 1: 辅助函数 (融合了新旧版本)
# ================================================================

# ## 升级 ##: 从新版引入了更通用的 set_symlink 函数
def set_symlink(model_type, method):
    """
    根据选择的方法创建指向正确模型实现文件的符号链接。
    - method 'neural': 链接 modeling_{model_type}neu.py
    - method 'infinigen': 链接 modeling_{model_type}inf.py
    - method 'orig': 链接 modeling_{model_type}_orig.py (用于 full cache 和 H2O)
    """
    if method == 'neural':
        fname = f"modeling_{model_type}_neu.py"
    elif method == 'infinigen':
        fname = f"modeling_{model_type}_inf.py"
    else: # 'orig' for Full Cache and H2O
        fname = f"modeling_{model_type}_orig.py"

    # --- 路径修正 (来自新版) ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.join(current_script_dir, "..")
    linker_path = os.path.join(project_root_dir, "src", fname)
    model_path = os.path.join(project_root_dir, "transformers/src/transformers/models/", model_type)
    # --- 修正结束 ---

    if not os.path.exists(linker_path):
        print(f"错误：源文件不存在于 {linker_path}")
        print("请确认 'src' 文件夹和 modeling 文件在项目根目录下。")
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


# ## 保留 ##: 旧版的模型加载函数
def get_opt_base(model, seq_len):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    # 延迟导入，确保符号链接已设置
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map = 'cpu')

    model.seqlen = seq_len
    return model


@torch.no_grad()
def opt_eval(model, testenc, eval_samples, ours, print_chunk=False):
    print('正在使用 opt_eval (layerwise) 方法进行评估...')

    dev = torch.device('cuda:0')
    # 确保 predictor 在 GPU 上
    if hasattr(model, 'predictor') and model.predictor is not None:
        print(f"将神经网络预测器移动到设备: {dev}")
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
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    prev_hidden = []

    # ================== [修复开始] 预计算 Predictor Logits ==================
    # 因为 opt_eval 是逐层跑的，不会触发 OPTModel.forward 里的逻辑，
    # 我们必须在这里手动运行 Predictor 并缓存结果。
    predictor_cache = [] # 存储每个样本的预测结果
    
    is_neural = hasattr(model, 'predictor') and model.predictor is not None
    if is_neural:
        print("正在为 Neural Prefetching 预计算 Logits...")
        # 临时把 embedding 层放回 GPU 用于计算 predictor 输入
        embed_tokens_tmp = model.model.decoder.embed_tokens.to(dev)
        
        for j in range(nsamples):
            # 获取原始 token id
            batch_ids = testenc[:, (j * model.seqlen):((j + 1) * model.seqlen)].to(dev)
            with torch.no_grad():
                # Predictor 需要原始 Embedding (不带位置编码)
                # 注意：Modeling 文件中是 predictor(inputs_embeds.to(float32))
                inputs_embeds = embed_tokens_tmp(batch_ids)
                
                # 运行 Predictor
                # 输出 shape: [1, Num_Predict_Layers, Seq_Len]
                logits = model.predictor(inputs_embeds.to(torch.float32))
                predictor_cache.append(logits)
        
        # 用完放回 CPU (虽然 embed_tokens 占用不大，保持逻辑一致)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    # ================== [修复结束] ==================

    density = []
    print("逐层处理...")
    import tqdm
    
    # 获取配置好的预取层列表
    prefetch_layers_list = getattr(model.config, "prefetch_layers_list", None)
    
    for i in tqdm.tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        if ours and hasattr(layer.self_attn, 'partial_weight_q'):
            layer.self_attn.partial_weight_q = layer.self_attn.partial_weight_q.to(dev)
            
        for j in range(nsamples):
            # InfiniGen 逻辑
            if ours and not is_neural: 
                if i >= 2:
                    layer.self_attn.previous_hidden_states = prev_hidden[j]
            
            # ================== [修复开始] 准备当前层的 Logits ==================
            current_layer_logits = None
            if is_neural:
                # 获取当前样本的预测结果 [1, Num_Pred_Layers, Seq]
                sample_logits = predictor_cache[j] 
                num_pred_layers = sample_logits.shape[1]
                
                # 复制 modeling_opt_neu.py 中的分发逻辑
                if i < num_pred_layers:
                    # 如果定义了 list，只在 list 中的层启用；否则全部启用
                    if prefetch_layers_list is None or i in prefetch_layers_list:
                        current_layer_logits = sample_logits[:, i, :]
                else:
                    # 超过预测层数的后续层，复用最后一层的预测结果
                    current_layer_logits = sample_logits[:, num_pred_layers - 1, :]
            
            # 构造参数字典，如果 logits 存在则传入
            forward_kwargs = {'attention_mask': attention_mask}
            if current_layer_logits is not None:
                forward_kwargs['predictor_logits'] = current_layer_logits

            # 调用层，传入 predictor_logits
            outs[j] = layer(inps[j].unsqueeze(0), **forward_kwargs)[0]
            # ================== [修复结束] ==================

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
        retain_ratio = (1 - math.sqrt(1 - (density_val / 100))) * 100
        print(f"InfiniGen/Neural 平均密度: {density_val:.2f}%")
        # print(f"InfiniGen 保留率: {retain_ratio:.2f}%")

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(torch.float)
        nlls.append(neg_log_likelihood)
        
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
    print(f"总 Perplexity: {ppl.item()}")
    model.config.use_cache = use_cache


# ================================================================
#  SECTION 2: 主逻辑
# ================================================================
if __name__ == '__main__':
    # ## 升级 ##: 使用新版的参数解析器，支持所有策略
    parser = argparse.ArgumentParser()
    # --- 通用模型和数据集参数 ---
    parser.add_argument('--model', type=str, required=True, help="模型路径 (本地或Hugging Face Hub)")
    parser.add_argument('--eval_dataset', type=str, required=True, help="评估数据集名称 (e.g., 'wikitext2', 'c4')")
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--eval_samples', type=int, default=0, help="评估样本数 (0表示使用所有样本)")
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b', help="用于分词器和配置的HF模型名称")
    parser.add_argument('--print_blk_ppl', action='store_true', help="分块打印PPL")
    parser.add_argument('--model-type', type=str, default='opt', help="模型架构类型 (e.g., 'opt', 'llama')")

    # --- H2O 参数 ---
    parser.add_argument('--enable-small-cache', action='store_true', help="启用 H2O (heavy-hitter) 缓存策略")
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    # --- InfiniGen & Neural 共享参数 ---
    parser.add_argument('--ours', action='store_true', help="启用自定义模型 (InfiniGen 或 Neural)")
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='infinigen', 
        choices=['infinigen', 'neural'],
        help="当启用 '--ours' 时，选择具体策略"
    )

    # --- InfiniGen 特定参数 ---
    infinigen_group = parser.add_argument_group('InfiniGen Strategy Arguments')
    infinigen_group.add_argument("--partial_weight_ratio", type=float, default=0.1)
    infinigen_group.add_argument("--partial_weight_path", type=str, default=None)
    infinigen_group.add_argument("--alpha", type=float, default=5.0)
    infinigen_group.add_argument("--capacity", type=float, default=1.0)
    infinigen_group.add_argument("--budget", type=float, default=0.2)
    infinigen_group.add_argument("--eviction_policy", type=str, default="lru", help="[InfiniGen] 驱逐策略")

    # --- Neural Prefetching 特定参数 ---
    neural_group = parser.add_argument_group('Neural Prefetcher Arguments')
    neural_group.add_argument('--predictor-path', type=str, default=None, help="[Neural] 神经网络预测器的权重路径")
    neural_group.add_argument('--prefetch-budget-ratio', type=float, default=0.3, help="[Neural] 预取预算比例")

    args = parser.parse_args()

    # 步骤 1. 根据策略设置符号链接
    if args.ours:
        print(f"启用自定义模型，策略: {args.strategy.upper()}")
        set_symlink(args.model_type, args.strategy)
    else:
        print("使用原始模型实现 (Full Cache or H2O)")
        set_symlink(args.model_type, 'orig')

    # 延迟导入，确保符号链接设置后才导入相关模块
    from transformers import AutoConfig, OPTForCausalLM

    # --- 关键修改：提前加载和修改配置 ---
    print("加载模型配置...")
    # 使用 model_name 加载基础配置，因为这里包含了架构信息
    config = AutoConfig.from_pretrained(args.model_name)
    
    # 如果是 'neural' 策略，在加载模型前向配置中注入必要参数
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

    # --- 关键修改：使用修改后的配置加载模型 ---
    # 不再使用 get_opt_base 函数
    print(f"加载模型: {args.model}")
    model = OPTForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map='cpu',
        config=config,  # 传入修改后的配置
        ignore_mismatched_sizes=True if args.ours and args.strategy == 'neural' else False # 忽略predictor权重的缺失
    )
    model.seqlen = args.seq_len
    
    # 步骤 3. 模型加载后的特定修改 (融合新旧逻辑)
    if args.enable_small_cache:
        print('\n--- 配置 H2O (Small Cache Size) ---')
        # ... (这部分逻辑保持不变)
        sys.path.append("../lm_eval")
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
        
        model.config.heavy_ratio = args.heavy_ratio
        model.config.recent_ratio = args.recent_ratio
        model = convert_kvcache_opt_heavy_recent(model, model.config)
        model = model.to(torch.float16)
        print(f"H2O 配置完成: heavy_ratio={model.config.heavy_ratio}, recent_ratio={model.config.recent_ratio}")

    elif args.ours:
        if args.strategy == 'infinigen':
            print("\n--- 配置 InfiniGen 模型层 ---")
            # ... (这部分逻辑保持不变)
            for layer in model.model.decoder.layers:
                layer.self_attn.partial_weight_ratio = args.partial_weight_ratio
                layer.self_attn.partial_weight_q = torch.load(os.path.join(args.partial_weight_path, f"partial_weight_q_{layer.self_attn.layer_idx}.pt"), map_location='cpu')
                layer.self_attn.alpha = args.alpha
                layer.self_attn.capacity = args.capacity
                layer.self_attn.budget = args.budget
                layer.self_attn.eviction_policy = args.eviction_policy
            print(f"InfiniGen 配置完成: budget={args.budget}, alpha={args.alpha}")

        elif args.strategy == 'neural':
            print("\n--- 配置神经网络预取器 ---")
            # 此时 model.predictor 应该已经被成功创建
            if not hasattr(model, 'predictor') or model.predictor is None:
                # 如果这里仍然报错，说明 modeling_opt_neu.py 的 __init__ 方法可能有问题
                raise TypeError("加载的模型仍然没有 'predictor' 属性。请检查 'modeling_opt_neu.py' 的实现。")

            print(f"从以下路径加载神经网络预测器权重: {args.predictor_path}")
            predictor_state_dict = torch.load(args.predictor_path, map_location='cpu')
            model.predictor.load_state_dict(predictor_state_dict)
            model.predictor.eval()
            model.predictor.to(torch.float32)
            print("神经网络预测器配置并加载成功。")
    else:
        print("\n--- 运行 Full Cache (Baseline) 模式 ---")

    model.eval()
    from datautils import *
    # 步骤 4. 使用旧版方式加载数据
    print(f"\n为 '{args.eval_dataset}' 数据集加载数据...")
    testloader = get_loaders(args.eval_dataset, model=args.model_name, seqlen=model.seqlen)

    # 步骤 5. 调用旧版评估函数
    is_infinigen_for_eval = args.ours and args.strategy == 'infinigen'
    
    opt_eval(model, testloader, args.eval_samples, is_infinigen_for_eval, args.print_blk_ppl)