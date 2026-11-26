# Causal tracing

This module runs ROME's corrupt-and-restore causal tracing on definitions. It measures how much each hidden state, MLP output and attention output carries the definition's meaning towards the definiendum.

The module is imported as `tracing`.

## Protocol

For each fact, one batch of 11 runs of the prompt:

1. **Clean run** (row 0): the prompt as it is; every internal state is kept.
2. **Corrupted runs** (rows 1-10): Gaussian noise added to the embeddings of the definition tokens (the fact's `subject`).
3. **Patched runs:** the corrupted runs again, with one clean state restored.

The probability of the definiendum (the clean argmax token) is averaged over the corrupted rows. A fact is traced only if the clean argmax token is its definiendum (single-token correctness); otherwise its npz records `correct_prediction=False`.

Three trace kinds:

| Kind | Restored per cell (token *i*, layer *j*) | File suffix |
| --- | --- | --- |
| `hidden` | the hidden state at (*i*, *j*) | none |
| `mlp` | the MLP output at token *i* for the `window` layers centred on *j* | `_mlp` |
| `attn` | the attention output at token *i* for the `window` layers centred on *j* | `_attn` |

## Noise

| Rule | Noise |
| --- | --- |
| float | N(0, s²) with the given scale *s* |
| `s<k>` | *k* times the standard deviation of the embeddings of the traced definitions; `s3` is the ROME 3σ rule |
| `m` | multivariate Gaussian matched to the embedding covariance (estimated on WikiText-103) |
| `t<df>` | Student-t with `df` degrees of freedom and the embedding covariance |
| `u<scale>` | uniform noise of the given scale |

For GPT-J-6B, `s3` gives 3σₑ = 0.12922; `TraceConfig.gptj()` uses that value directly. The noise draws use a fixed `RandomState(1)`, so traces are reproducible.

## Window

The MLP and attention window keeps GPT-J's window-to-depth ratio of 10/28:

```
w = 10                          for L >= 28
w = max(2, round(10 * L / 28))  for L <  28
```

```python
from tracing import window_for

window_for(18)   # 6
```

A window below 2 restores no layers, so the MLP and attention maps would come out flat at the corrupted baseline; `TraceConfig` refuses it.

## Configuration

```python
from tracing import TraceConfig

config = TraceConfig(
    model_name="gpt2-large",
    noise="s3",
    samples=10,                      # corrupted runs per input
    window=10,
    kinds=("hidden", "mlp", "attn"),
    replace=False,                   # replace the embeddings with noise instead of adding it
    torch_dtype="auto",              # "auto", "float32", "float16" or "bfloat16"
    device="cuda",                   # default: cuda if available
    num_cases=-1,                    # -1 traces every fact
    max_plot_id=200,                 # heatmaps for known_id <= 200
    output_dir="results/{run_name}/causal_trace",
)
config.run_name                      # "ns3_r0_gpt2-large"
```

Presets:

```python
TraceConfig.gptj()                                # GPT-J-6B as in the paper
TraceConfig.for_model("google/gemma-2b", 18)      # 3σ noise, window scaled to 18 layers
```

## Tracing a fact file

```python
from tracing import TraceConfig, load_facts, trace_facts

config = TraceConfig.for_model("gpt2-large", 36)
correct = trace_facts(config, load_facts("data/facts_gpt2-large_n800.json"))
```

`trace_facts(config, facts, mt=None)` returns the `known_id` of every correctly predicted fact. Existing npz files are reused, so an interrupted run resumes where it stopped.

From the command line, from the repository root:

```bash
python -m tracing.trace --model_name gpt2-large --fact_file data/facts_gpt2-large_n800.json

# GPT-J-6B with the paper's fixed noise
python -m tracing.trace --model_name EleutherAI/gpt-j-6B \
    --fact_file data/known_data_wordNet.json --noise_level 0.12922460585832596
```

| Argument | Purpose |
| --- | --- |
| `--model_name` | Hugging Face id (default: `EleutherAI/gpt-j-6B`). |
| `--fact_file` | Fact records (default: `data/known_data_wordNet.json`). |
| `--output_dir` | Results folder; `{model_name}` becomes `n<noise>_r<replace>_<model>`. |
| `--noise_level` | Noise rule (default: `s3`). |
| `--replace` | `1` replaces the definition embeddings with noise. |
| `--num_cases` | Trace only the first N facts. |
| `--torch_dtype` | `auto`, `float32`, `float16` or `bfloat16`. |
| `--window` | Severed-window width (default: the depth rule above). |
| `--samples` | Corrupted runs per input (default: `10`). |
| `--kinds` | Trace kinds to run (default: all three). |
| `--max_plot_id` | Draw heatmaps for `known_id` up to this (default: `200`). |
| `--device` | Torch device (default: `cuda` if available). |

## Outputs

Under `results/n<noise>_r<replace>_<model>/causal_trace/`:

- `cases/knowledge_<id>.npz`, `knowledge_<id>_mlp.npz`, `knowledge_<id>_attn.npz` with `scores` (tokens x layers grid of patched probabilities), `low_score` (corrupted), `high_score` (clean), `input_ids`, `input_tokens`, `subject_range` (the corrupted span), `answer`, `window`, `kind` and `correct_prediction`. bfloat16 runs are saved as float32.
- `pdfs/<answer>_<id>[_mlp|_attn].pdf`: heatmaps of `scores` with `vmin` at `low_score` and a `p(answer)` colourbar; corrupted tokens are marked `*`.

Every npz stores the `window` the run was launched with, including hidden-state traces, which never use it; judge a run's window by its `_mlp` and `_attn` files.

## Single traces

```python
from tracing import ModelAndTokenizer, calculate_hidden_flow, plot_trace_heatmap

mt = ModelAndTokenizer("gpt2-large", device="cuda")
prompt = 'a building where books are kept is called a "'
result = calculate_hidden_flow(mt, prompt, "a building where books are kept",
                               noise=0.1, window=10, kind="mlp")
plot_trace_heatmap(result, savepdf="library_mlp.pdf")
```

## Supported models

`layername` recognises three module trees: GPT-2 / GPT-J (`transformer.h`), GPT-NeoX / Pythia (`gpt_neox.layers`) and the Llama family, which also covers Qwen2, Gemma and Mistral (`model.layers`). For SentencePiece tokenisers (Gemma), whose per-token decoding drops word-boundary spaces, the corrupted span is found on incremental decodes. Llama and Gemma tokenisers prepend a BOS token, which appears as an extra first row in the heatmaps; the corrupted span covers only the definition tokens.

The ROME helpers the tracer uses (`nethook`, `runningstats`, `tok_dataset`) are kept unchanged in `legacy/rome`.
