# Cross-model runs

The paper traces GPT-J-6B (English) and BERTIN GPT-J-6B (Spanish). The launcher `run_dsra_models.sh` repeats the tracing on pretrained models from other families, to test whether the localisation profiles reported for GPT-J (bimodal MLP recovery, middle-to-upper attention recovery, upper-layer hidden-state recovery at the final token) hold across architectures. All models are base checkpoints.

## Models

Default group (the CAP model set):

| Model | Hugging Face id | Layers | Window | Template |
| --- | --- | --- | --- | --- |
| GPT-2 Large | `gpt2-large` | 36 | 10 | `{} is called a "` |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | 36 | 10 | `{} is commonly known as "` |
| Gemma-2B | `google/gemma-2b` | 18 | 6 | `{} is called a "` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | 28 | 10 | `{} is called a "` |

`ch6` group (the COMPASS models at about 1B scale), plus two Pythia checkpoints that run only when named:

| Model | Hugging Face id | Layers | Window | Template | In `ch6` |
| --- | --- | --- | --- | --- | --- |
| Pythia-1B | `EleutherAI/pythia-1b` | 16 | 6 | `{} is called a "` | yes |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | 16 | 6 | `{} is called a "` | yes |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 28 | 10 | `{} is commonly known as "` | yes |
| Pythia-410M | `EleutherAI/pythia-410m` | 24 | 9 | `{} is called a "` | no |
| Pythia-14M | `EleutherAI/pythia-14m` | 6 | 2 | `{} is called a "` | no |

The templates are the per-family prompts of the [prompt registry](prompts.md). All models run in float32 by default; the largest is about 13 GB of weights.

## Environment

The environment of `environment.yml` (transformers 4.52.4) supports every model above; Llama-3.2 needs transformers 4.43 or later, which the launcher checks before tracing it. `google/gemma-2b` and the Llama-3.2 models are gated: accept their licence on Hugging Face and log in (`huggingface-cli login`) or export `HF_TOKEN`.

## Running

From the repository root:

```bash
N_FACTS=25 bash run_dsra_models.sh          # pilot, the default group
bash run_dsra_models.sh                     # full runs, 800 screened facts per model
bash run_dsra_models.sh Qwen/Qwen2.5-3B     # one model
bash run_dsra_models.sh ch6                 # the ch6 group
bash run_dsra_models.sh EleutherAI/pythia-410m EleutherAI/pythia-14m
```

| Variable | Purpose |
| --- | --- |
| `N_FACTS` | Facts per model (default: `800`). |
| `NOISE` | Noise rule (default: `s3`). |
| `DTYPE` | Precision for screening and tracing (default: per model, `float32`). On a memory-constrained GPU use `bfloat16`, which is safer than `float16` for Gemma; bfloat16 traces are saved as float32. |
| `WINDOW` | Severed-window width for **every** model of the run, replacing the per-model values; leave it unset unless that is what you mean. |

For each model the launcher builds `data/facts_<model>_n<N>.json` with `python -m dataset_construction.facts.build_facts` (screened, reused if present) and traces it with `python -m tracing.trace`. Logs go to `logs/`. Fact files are named by size, so a pilot file is never reused for a full run.

Outputs land in `results/ns3_r0_<model>/causal_trace/`, with the npz schema and heatmaps described in [tracing](tracing.md).

## Runtime

Each case runs three grids (hidden, MLP, attention) of roughly tokens x layers patched forward passes at batch 11. On an RTX A6000 in float32, expect about 1-3 minutes per case for the 2B-3B models (16-32 hours for 800 cases), tens of seconds for the 1B models, and seconds for Pythia-14M. `bfloat16` roughly halves the time; `N_FACTS` scales it linearly.

## Protocol notes

- **Noise.** `s3` recomputes 3σ from the embedding spread of each model's own definitions, the rule that gives 3σₑ = 0.12922 for GPT-J. The tracer prints the value; record it per model.
- **Correct-prediction conditioning.** Fact files are screened, so every traced case is baseline-correct, as in the GPT-J protocol.
- **Window.** The severed window keeps GPT-J's window-to-depth ratio of 10/28 = 0.36 (`w = 10` for `L >= 28`, `max(2, round(10 * L / 28))` below), because the window sets how much the layer distribution is smoothed. Report each model's window with `w / L`. An earlier rule cut over at 18 layers and left Gemma-2B at `w = 10` (`w / L = 0.56`); Gemma's MLP and attention traces were rerun at `w = 6`, and any Gemma `_mlp` / `_attn` npz stamped `window=10` predates that correction.
- **Stored windows.** Every npz stores the launch window, including hidden-state traces, which never use it. Some hidden-state files of Pythia-1B, Llama-3.2-1B and Qwen2.5-1.5B carry a stale `window=0` stamp from an earlier launch; their scores are valid. Judge a run's window by its `_mlp` / `_attn` files, as `analysis` does.
- **Window below 2.** Both the launcher and the tracer refuse it: such a window restores no layers, and the MLP / attention maps come out flat at `low_score` while still writing a full set of npz.
- **BOS tokens.** Llama and Gemma tokenisers prepend a BOS token, an extra first row in token-level heatmaps; GPT-J, GPT-2 and Pythia have none. The corrupted span covers only the definition tokens.
- **Comparability.** Compare localisation across models in normalised depth (`layer / (L - 1)`).
- **Pythia-14M.** Its zero-shot accuracy is very low, so the builder may exhaust the pool before 800 facts; report the obtained N, and treat its MLP / attention decomposition (`w = 2`) as indicative only.

## After a pilot

```bash
python -m analysis.summarise check --results results
```

Expect every npz to have a correct clean prediction (screened facts), corruption to lower `p(answer)` in nearly all cases, one window per MLP / attention run and no flat maps. The facts log reports the screening yield, roughly 10-35 % for the 1B-3B models. In the heatmaps, the definition tokens are marked `*` and recovery should appear at the final token in late layers.

Then aggregate:

```bash
python -m analysis.summarise cross-model --results results
python -m analysis.summarise topk-mass --results results
python -m analysis.summarise plot
```
