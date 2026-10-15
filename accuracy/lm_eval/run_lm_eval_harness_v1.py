# run_lm_eval_harness.py (MODIFIED for ALPHA + BUDGET Neural Prefetching + Interval Prefetching)
import argparse
import json
import tqdm
import torch
import os
import time

def set_symlink(model_type, fname):
    # (This function remains unchanged, no need to copy it here)
    model_path = os.path.join(os.path.dirname(__file__), "..", "transformers/src/transformers/models/", model_type)
    linker_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src/", fname))
    
    if not os.path.exists(linker_path):
        print(f"Error: Source file for symlink does not exist at {linker_path}")
        exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Target directory for symlink does not exist at {model_path}")
        exit(1)
        
    curr_dir = os.getcwd()
    os.chdir(model_path)
    
    target_file = f'modeling_{model_type}.py'
    if os.path.lexists(target_file): # Use lexists for symlinks
        os.remove(target_file)
        
    print(f"Creating symlink: {target_file} -> {linker_path}")
    os.symlink(linker_path, target_file)
    os.chdir(curr_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LM Eval Harness with Prefetching',
        description='Run language model evaluation with different KV cache prefetching strategies.'
    )

    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--ours', action='store_true', help="Enable custom model implementation (_ours.py)")

    # Group for prefetching strategies
    prefetch_group = parser.add_argument_group('Prefetching Strategies')
    prefetch_group.add_argument('--prefetch-strategy', type=str, default=None, choices=['infinigen', 'neural'],
                                help="Prefetching strategy to use.")

    # --- <<< MODIFICATION 1: Update arguments for Neural Prefetching >>> ---
    neural_group = parser.add_argument_group('Neural Prefetcher Arguments')
    neural_group.add_argument('--predictor-path', type=str, default=None,
                              help="[Neural] Path to the pretrained weights of the neural prefetch predictor.")
    neural_group.add_argument('--prefetch-budget-ratio', type=float, default=0.3,
                              help="[Neural] The ratio of sequence length to prefetch, constrained by budget (e.g., 0.3 for 30%).")
    # ALPHA is now part of the neural strategy's configuration, but it's defined in the model code,
    # as the predictor was trained with a specific ALPHA. We can add it here if we want to override.

    # --- Arguments for original InfiniGen strategy (unchanged) ---
    infinigen_group = parser.add_argument_group('InfiniGen Arguments')
    infinigen_group.add_argument("--alpha",type=float, default=5, help="[InfiniGen] Alpha value for attention thresholding.")
    infinigen_group.add_argument("--capacity",type=float, default=1.0, help="[InfiniGen] KV cache capacity.")
    infinigen_group.add_argument("--budget",type=float, default=0.2, help="[InfiniGen] Prefetching budget.")
    # (other infinigen args can be added here if needed)
    infinigen_group.add_argument("--prefetch-start-layer",type=int, default=1, help="[InfiniGen] Layer index to start infinigen prefetching from. Before this layer, full KV cache is used.")
    infinigen_group.add_argument("--prefetch-interval",type=int, default=1, help="[InfiniGen] The interval between layers that perform prefetching.")
    # (Other arguments like quantization remain unchanged)
    # ...
    
    args = parser.parse_args()
    #set_symlink(args.model_type, f"modeling_{args.model_type}_orig_v1.py")
    set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")
    #set_symlink(args.model_type, f"modeling_{args.model_type}_ours_v1.py")
    # Symlink logic remains the same. Use the _ours file if --ours is specified.
    

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print("Loading model config...")
    config = AutoConfig.from_pretrained(args.model_name)

    # --- <<< MODIFICATION 2: Update Config setup logic >>> ---
    if args.prefetch_strategy:
        print(f"\n[INFO] Prefetching strategy enabled: {args.prefetch_strategy.upper()}")
        config.prefetch_strategy = args.prefetch_strategy
        
        if args.prefetch_strategy == 'neural':
            # Pass the budget ratio to the model config
            config.prefetch_budget_ratio = args.prefetch_budget_ratio
            print(config.prefetch_budget_ratio)
            # These are fixed based on the predictor's architecture
            config.num_predict_layers = 5 
            
            config.predictor_nhead = 4
            config.predictor_encoder_layers = 1
            config.predictor_d_hid = 2048
            '''
            config.predictor_nhead = 16
            config.predictor_encoder_layers = 8
            config.predictor_d_hid = 4096
            '''
            config.prefetch_layers_list = [0,1,2,3,4] 
            config.alpha = args.alpha
            config.capacity = args.capacity
            config.budget = args.budget
            print(f"  - Strategy: NEURAL")
            print(f"  - Budget Ratio: {config.prefetch_budget_ratio * 100:.1f}%")
            if not args.predictor_path:
                raise ValueError("--predictor-path must be provided for the 'neural' strategy.")

        elif args.prefetch_strategy == 'infinigen':
            # Pass InfiniGen specific parameters to the config
            config.alpha = args.alpha
            config.capacity = args.capacity
            config.budget = args.budget
            print(f"  - Strategy: INFINIGEN")
            print(f"  - Alpha: {config.alpha}")
            print(f"  - Budget: {config.budget}")
            print(f"  - Capacity: {config.capacity}")
            # --- <<< 新增逻辑开始 >>> ---
            # Also print the new scheduling parameters
            print(f"  - Prefetch Start Layer: {args.prefetch_start_layer}")
            print(f"  - Prefetch Interval: {args.prefetch_interval}")
            # --- <<< 新增逻辑结束 >>> ---

    else:
        print("\n[INFO] No prefetching strategy specified.")
    # --- <<< END MODIFICATION 2 >>> ---

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    print("\nLoading model with modified config...")
    model_load_path = args.model_path if args.model_path else args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, 
        torch_dtype=torch.float16, 
        config=config,
        ignore_mismatched_sizes=True # Important for loading models with custom heads/modules
    )
    
    # --- <<< MODIFICATION 3: Update model configuration logic >>> ---
    if args.ours and args.prefetch_strategy == 'neural':
        print("\n--- Configuring model for NEURAL PREDICTOR strategy ---")
        
        # Check if the model has the predictor attribute (defined in our modified modeling_opt.py)
        if not hasattr(model, 'predictor') or model.predictor is None:
             raise TypeError("The loaded model does not have a 'predictor' attribute. Make sure you are using the modified modeling script (e.g., modeling_opt_ours.py) which initializes it.")

        print(f"Loading neural predictor weights from: {args.predictor_path}")
        # Load the pretrained predictor weights.
        # It's good practice to keep the predictor in float32 for better precision.
        predictor_state_dict = torch.load(args.predictor_path, map_location='cpu')
        model.predictor.load_state_dict(predictor_state_dict) 
        model.predictor.eval() # Set predictor to evaluation mode
        print("Neural predictor configured and loaded successfully.")

    elif args.ours and args.prefetch_strategy == 'infinigen':
        print("\n--- Configuring model for original INFINIGEN strategy ---")
        
        # --- <<< 新增逻辑开始: 配置隔层预取调度 >>> ---
        start_layer = args.prefetch_start_layer
        interval = args.prefetch_interval
        num_layers = model.config.num_hidden_layers

        print("Applying custom prefetching schedule...")
        
        # The logic iterates through each layer `i` and sets a flag that will control the *next* layer's (`i+1`) behavior.
        for i in range(num_layers):
            # The layer whose behavior we are about to determine (the one that might prefetch).
            layer_to_be_controlled_idx = i + 1
            
            enable_prefetch_for_next_layer = False
            
            # Condition to enable prefetching for the next layer:
            # 1. The next layer's index must be at or after the start_layer.
            # 2. The distance from the start_layer must be a multiple of the interval.
            if layer_to_be_controlled_idx >= start_layer:
                if (layer_to_be_controlled_idx - start_layer) % interval == 0:
                    enable_prefetch_for_next_layer = True
            
            # The `enable_prefetching` flag on layer `i` controls layer `i+1`.
            # We assume an OPT-like model structure: model -> model -> decoder -> layers
            # You might need to adjust this path for other model architectures.
            try:
                # This flag exists in our custom `modeling_opt_ours_v1.py`
                model.model.decoder.layers[i].self_attn.enable_prefetching = enable_prefetch_for_next_layer
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not set prefetching flag for layer {i}. "
                      f"Ensure the model structure is correct. Error: {e}")
                break # Stop if the structure is not as expected.

        # For verification, let's print which layers will actually perform prefetching.
        prefetching_layers = []
        for i in range(1, num_layers): # Check from layer 1 onwards
            # Layer `i` prefetches if the flag on the *previous* layer (`i-1`) is True.
            if model.model.decoder.layers[i-1].self_attn.enable_prefetching:
                prefetching_layers.append(i)
        
        if not prefetching_layers:
            print("  - No layers are scheduled for prefetching based on the provided settings.")
        else:
            print(f"  - Layers scheduled to perform prefetching: {prefetching_layers}")
        # --- <<< 新增逻辑结束 >>> ---

    # --- <<< END MODIFICATION 3 >>> ---

    # Move model to GPU
    model.eval().cuda()
    if args.ours and args.prefetch_strategy == 'neural':
        # Also move the predictor to the same device as the main model
        model.predictor.to(model.device, dtype=torch.float32)

    # --- Evaluation loop (remains the same) ---
    print("\nStarting evaluation loop...")
    start_time = time.time()
    requests = []
    with open(args.input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []
    for i, request in enumerate(tqdm.tqdm(requests, desc="Processing requests")):
        result = {'request': request, 'result': {}}
        prompt = request['prompt']
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        
        if input_ids.shape[1] > tokenizer.model_max_length:
            input_ids = input_ids[:, :tokenizer.model_max_length]
        
        # When model is called, our prefetching logic inside modeling_opt.py will be triggered
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs.logits.log_softmax(dim=-1)

        values, indices = logits.squeeze(0).topk(dim=-1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        gold_indices = input_ids[:, 1:]
        logprobs = [None] + torch.gather(logits[:, :-1, :], -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).cpu().tolist()
        top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1)[:-1], indices.squeeze(-1)[:-1])]
        
        result['result'] = {
            "choices": [
                {"text": prompt, "logprobs": {"tokens": tokens, "token_logprobs": logprobs, "top_logprobs": top_logprobs, "text_offset": []}, "finish_reason": "length"}
            ], 
            "request_time": {"batch_time": 0, "batch_size": 1}
        }
        results.append(result)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds for {len(requests)} requests.")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/second.")
    
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
    print(f"\nResults saved to {args.output_path}")
    
    