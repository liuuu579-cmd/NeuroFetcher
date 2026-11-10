import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor


DATASET_CACHE_LAYOUTS = {
    "openbookqa": {
        "directory": "openbookqa/main/0.0.0/*",
        "prefix": "openbookqa",
        "splits": ("train", "validation", "test"),
    },
    "winogrande": {
        "directory": "winogrande/winogrande_xl/{version}/*",
        "prefix": "winogrande",
        "splits": ("train", "validation"),
        "versions": ("0.0.0", "1.1.0"),
    },
    "copa": {
        "directory": "super_glue/copa/0.0.0/*",
        "prefix": "super_glue",
        "splits": ("train", "validation"),
    },
    "rte": {
        "directory": "glue/rte/0.0.0/*",
        "prefix": "glue",
        "splits": ("train", "validation"),
    },
}


def read_arrow_rows(path):
    """Read a cached Hugging Face Arrow stream without resolving a Hub dataset."""
    with pa.memory_map(str(path), "r") as source:
        return pa_ipc.open_stream(source).read_all().to_pylist()


def local_task_dataset(task_name, cache_root):
    """Load task splits from the local datasets cache only."""
    layout = DATASET_CACHE_LAYOUTS.get(task_name)
    if layout is None:
        raise ValueError(f"No offline dataset-cache mapping for task {task_name!r}")

    versions = layout.get("versions", (None,))
    last_missing = None
    for version in versions:
        relative_dir = layout["directory"].format(version=version) if version else layout["directory"]
        directories = sorted(cache_root.glob(relative_dir))
        for directory in directories:
            files = {
                split: directory / f"{layout['prefix']}-{split}.arrow"
                for split in layout["splits"]
            }
            missing = [path for path in files.values() if not path.is_file()]
            if missing:
                last_missing = missing[0]
                continue
            return {split: read_arrow_rows(path) for split, path in files.items()}

    details = f"; last missing file: {last_missing}" if last_missing else ""
    raise FileNotFoundError(
        f"Cached data for {task_name!r} was not found under {cache_root}{details}"
    )


def load_local_task(task_name, cache_root):
    """Instantiate an lm_eval task with cached Arrow splits and no Hub access."""
    task_class = tasks.get_task(task_name)

    def download_from_cache(instance, *args, **kwargs):
        instance.dataset = local_task_dataset(task_name, cache_root)

    local_task_class = type(
        f"Local{task_class.__name__}", (task_class,), {"download": download_from_cache}
    )
    return {task_name: local_task_class()}


def json_to_key(obj):
    return json.dumps(obj)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='result.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--tokenizer-path', type=str, default='/home/liushenyu/opt-13b')
    parser.add_argument('--seq', type=int, default=2048)
    parser.add_argument(
        '--datasets-cache-dir',
        type=Path,
        default=Path(os.environ.get('HF_DATASETS_CACHE', Path.home() / '.cache/huggingface/datasets')),
        help='Local Hugging Face datasets cache; evaluation does not access the Hub.',
    )
    parser.add_argument('--output-file', type=Path)
    args = parser.parse_args()
    
    if args.tokenizer_path:
        os.environ['MODEL_NAME'] = args.tokenizer_path
    elif args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "huggyllama/llama-7b"
    else:
        assert False

    seq = args.seq
    total_batch = 1
    pe = 'fixed'

    class RealRunner:
        
        def __init__(self, args):
            
            self.results = {}
            
            with open(args.result_file, 'r') as f:
                
                for line in f:
                    if line.strip() == '':
                        continue
                    
                    item = json.loads(line)
                    
                    request = item['request']
                    result = item['result']
                    
                    self.results[json_to_key(request)] = result
            
            print(f"{len(self.results)} items in the cache")
        
        def eval(self, batch):
            
            from tasks.eval_harness import tokenizer
            
            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch['text']):
                
                request = {
                        "best_of": 1, 
                        "echo": True, 
                        "logprobs": 1, 
                        "max_tokens": 0, 
                        "model": "x", 
                        "n": 1, 
                        "prompt": text, 
                        "request_type": "language-model-inference", 
                        "stop": None, 
                        "temperature": 0, 
                        "top_p": 1
                    }
                
                key = json_to_key(request)
                
                correct = True
                
                if key in self.results:
                    result = self.results[key]
                    
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    obs = batch['obs'][i]
                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            
                            if i+1 >= len(tokens):
                                break
                            
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    avg_logprob = sum_lobprob / n_positive
                    
                    mask_loss.append( - avg_logprob)
            
                    each_correct.append( correct )
                    
                else:
                    assert False
                

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            
            
            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    try:
        task_dict = load_local_task(args.task_name, args.datasets_cache_dir)
        results = evaluator.evaluate(adaptor, task_dict, num_fewshot=args.num_fewshot)
    finally:
        adaptor.pool.close()
        adaptor.pool.join()
    
    dumped = json.dumps(results, indent=2)
    print(dumped)
    output_file = args.output_file or Path(args.result_file).with_suffix('.accuracy.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(dumped + '\n', encoding='utf-8')
    print(f'accuracy_results={output_file}')
