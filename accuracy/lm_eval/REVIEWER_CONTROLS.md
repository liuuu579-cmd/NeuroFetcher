# Reviewer controls

These files implement the two controls requested in the reviews without
changing the existing evaluation scripts.

`previous_step` reuses the selected KV indices from the preceding query step,
retains the configured recent window, and keeps the same dynamic
`ceil(history_length * budget_ratio)` total budget. `oracle` selects the
current step's true attention top-k from the non-recent positions and includes
the same recent window. The default window is one token; set `RECENT_WINDOW`
to the exact fixed window used for training labels when that value is known.
The oracle is an upper-bound quality control and must be labelled as
non-implementable in the paper.

Run from this directory in the `infinigen` environment. The default model and
tokenizer path is `/home/liushenyu/opt-13b`, the standard OPT-13B checkpoint;
set `MODEL_PATH` and `TOKENIZER_PATH` if using a different standard OPT
checkpoint. The InfiniGen-modified checkpoint has a different projection-layer
shape and is not compatible with these reviewer controls.

```bash
conda run -n infinigen bash reviewer_controls.sh previous_step
conda run -n infinigen bash reviewer_controls.sh oracle
# Full matched-budget suite (runs serially to avoid GPU memory contention)
conda run -n infinigen bash run_reviewer_controls_suite.sh
# Continue after completed conditions; keep the same experiment settings
RESUME=1 conda run -n infinigen bash run_reviewer_controls_suite.sh
```

The evaluator reads the four task splits from the local Hugging Face Arrow
dataset cache (override with `DATASETS_CACHE`), so scoring does not require Hub access. It writes an
`.accuracy.json` file beside each result JSONL. The batch script supports
`RESUME=1`: it skips a condition only when its result, inference metrics, and
accuracy JSON are all present.

The wrapper exposes GPUs 0--3 and limits OpenMP/MKL to 42 threads. Set
`TASK`, `BUDGET_RATIO`, `LIMIT`, `DEVICE`, and `DTYPE` to reproduce a specific
condition. Existing JSONL inputs are used, and outputs remain in `results/`:
the JSONL contains model log probabilities, while the `.metrics.json` file
records token counts, elapsed time, actual budget, and previous-step IoK.

The current repository has task inputs for OpenBookQA, Winogrande, COPA, and
RTE. The checked-in PIQA input is empty and should be regenerated before using
that task.
