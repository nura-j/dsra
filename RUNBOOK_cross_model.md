# Runbook: cross-model DSRA causal tracing

Purpose: test whether the DSRA localisation profiles reported for GPT-J-6B
(Chapter 6, RQ6.1/RQ6.2 — bimodal MLP recovery, middle-to-upper attention
recovery, upper-layer hidden-state recovery at the final token) generalise
across model families. Target models (all base checkpoints, matching the
thesis model set):

| Model | HF id | Layers | Window | IDM template (from CAP/CARMA) | dtype |
|---|---|---|---|---|---|
| GPT-2 Large | `gpt2-large` | 36 | 10 | `{} is called a "` | float32 |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | 36 | 10 | `{} is commonly known as "` | float32 |
| Gemma-2B | `google/gemma-2b` | 18 | 6 | `{} is called a "` | float32 |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | 28 | 10 | `{} is called a "` | float32 |

Templates are the per-family IDM prompts already selected for CAP and CARMA.
All models run in float32 by default: the largest is about 13 GB of weights,
comfortably within the A6000's 48 GB, and float32 avoids any low-precision
concern. On a memory-constrained GPU, rerun with `DTYPE=bfloat16` (preferred
over float16, since Gemma was trained in bfloat16 and float16 risks overflow;
the tracing code saves bfloat16 tensors correctly by casting to float32 at
save time). Screening and tracing always use the same dtype, and fact files
are named by size (`facts_<model>_n<N>.json`), so pilot files are never
silently reused for full runs.

## 1. Environment

The existing ROME conda environment (Python 3.11.7, torch 2.2.2,
transformers 4.39.0.dev0) runs **GPT-2 Large, Qwen2.5-3B, and Gemma-2B**
as-is (Qwen2 support arrived in transformers 4.37, Gemma in 4.38).

**Llama-3.2 needs transformers >= 4.43**: its config declares
`rope_scaling.rope_type = "llama3"`, which older versions reject at load
time. To avoid disturbing the ROME environment, clone it once:

```bash
conda create -n rome-llama --clone <your_rome_env>
conda activate rome-llama
pip install -U "transformers==4.46.3"
```

torch 2.2.2 is compatible with transformers 4.46.3, and pip upgrades
`tokenizers` to the matching version automatically. The launcher script
checks the installed version and skips Llama-3.2 with a message when it is
too old, so the other three models can run in the original environment
either way.

**Gated models**: `google/gemma-2b` and `meta-llama/Llama-3.2-3B` require an
accepted licence on their Hugging Face pages and an authenticated session
(`huggingface-cli login`, or export `HF_TOKEN`).

## 2. What changed in this repository

Model support:
- `layername()` (in `experiments/causal_trace_wordnet.py` and
  `experiments/causal_trace.py`) now recognises the Llama-family module tree
  (`model.embed_tokens`, `model.layers.{n}.self_attn`, `model.layers.{n}.mlp`),
  which covers Llama-3.2, Qwen2.5, and Gemma. `ModelAndTokenizer` counts
  layers for that tree as well.
- `find_token_range()` gained a fallback that locates the corrupted span via
  incremental decoding, needed for SentencePiece tokenisers (Gemma) whose
  per-token decode drops word-boundary spaces.

Runner fixes (`experiments/causal_trace_wordnet.py`):
- `--fact_file` is honoured (a hardcoded `known_data_wordNet.json` override
  previously shadowed it, and the default path now points into `data/`).
- The `knowns[:10]` debugging cap is removed; `--num_cases` limits the run.
- `--model_name` accepts any supported HF id, and `--torch_dtype
  {auto,float32,float16,bfloat16}` controls precision.
- The single-token `predict_from_input` is the active definition again; an
  experimental greedy multi-token variant had shadowed it in this copy of the
  code (it is kept as `predict_from_input_greedy` for reference). The
  single-token form is the thesis protocol: a case counts as correct only if
  the argmax next token decodes to the definiendum.

New: `dataset_construction/build_dsra_facts.py` builds a per-model fact file
from `data/wordnet_data_definitions.json` (84,423 entries) with the DSRA pool
conventions: alphabetic definiendum, first definition per word, definitions
of at most 25 words, definiendum absent from its own definition, and the
definiendum forming a single continuation token after the model's prompt.
With `--screen`, candidates are kept only if the model answers them correctly
zero-shot, so every traced case is productive; selection order is shuffled
with seed 42.

## 3. Running

```bash
cd DSRA
# Pilot first (about 15-30 minutes per model):
N_FACTS=25 bash run_dsra_models.sh
# Full runs (800 screened cases per model):
bash run_dsra_models.sh
# Or one model at a time:
bash run_dsra_models.sh Qwen/Qwen2.5-3B
```

Outputs land in `results/ns3_r0_<model>/causal_trace/`:
- `cases/knowledge_<id>.npz` (hidden states), `knowledge_<id>_mlp.npz`,
  `knowledge_<id>_attn.npz` — the same schema as the GPT-J archive
  (`scores`, `low_score`, `high_score`, `input_tokens`, `subject_range`,
  `answer`, `correct_prediction`), so the existing aggregation code applies.
- `pdfs/` — per-case heatmaps for the first 200 ids (colourbar `p(answer)`,
  `vmin` at the corrupted baseline, matching the thesis convention).

Runtime on an RTX A6000: each case runs three grids (hidden/MLP/attention) of
roughly tokens x layers patched forward passes at batch 11 — expect in the
order of 1–3 minutes per case in float32, that is roughly 16–32 hours per
model for 800 cases (`DTYPE=bfloat16` roughly halves this). `N_FACTS` scales
this linearly.

## 4. Protocol notes (for the thesis write-up)

- **Noise**: `--noise_level s3` recomputes 3 sigma from the embedding spread
  of each model's own definition subjects — the same rule that produced
  3 sigma_e = 0.12922 for GPT-J. Record the printed per-model value.
- **Correct-prediction conditioning**: fact files are pre-screened, so traced
  cases are all baseline-correct, mirroring the ~800-case GPT-J protocol.
- **Window**: the MLP/attention severed-window is set per model to preserve
  GPT-J's window-to-depth ratio (Section 5). For the CAP set that is 10 layers
  for GPT-2 Large, Qwen2.5-3B and Llama-3.2-3B (L >= 28) and 6 for Gemma-2B
  (L = 18). State the per-model window in the chapter alongside w/L.
- **BOS tokens**: Llama-3.2 and Gemma tokenisers prepend a BOS token, which
  appears as an extra first row in token-level heatmaps; GPT-J and GPT-2 have
  no BOS. The corrupted span covers only the definition tokens in all models.
- **Comparability**: report layer positions as normalised depth (layer / L),
  as Chapter 4 does, when comparing localisation bands across models.
- **Analysis**: `analysis/dsra_topk_mass.py` computes recovery-mass shares by
  globbing `results/<run_dir>/causal_trace/cases/`; point its run-dir list at
  the new `ns3_r0_*` directories. The npz schema is unchanged.

## 5. Chapter 6 (COMPASS) model set

A second model group replicates the tracing on a sample of the models
Chapter 6 uses for COMPASS, so the DSRA localisation profiles and the
role-circuit results can be compared on identical checkpoints. GPT-J-6B
itself is the original DSRA model and is not rerun; the archive is the
reference. The sample is the three COMPASS cross-family models at
comparable (~1B) scale — Pythia-1B, Llama-3.2-1B, and Qwen2.5-1.5B — which
covers all three architecture families while skipping the two small Pythia
checkpoints (14M and 410M, used in COMPASS only for the temporal analyses;
their case entries remain in the launcher, so they can still be run by
naming them explicitly).

```bash
N_FACTS=25 bash run_dsra_models.sh ch6   # pilot
bash run_dsra_models.sh ch6              # full runs
# optional extras, by name:
bash run_dsra_models.sh EleutherAI/pythia-410m EleutherAI/pythia-14m
```

| Model | HF id | Layers | Window | IDM template | dtype | in `ch6` |
|---|---|---|---|---|---|---|
| Pythia-1B | `EleutherAI/pythia-1b` | 16 | 6 | `{} is called a "` | float32 | yes |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | 16 | 6 | `{} is called a "` | float32 | yes |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 28 | 10 | `{} is commonly known as "` | float32 | yes |
| Pythia-410M | `EleutherAI/pythia-410m` | 24 | 9 | `{} is called a "` | float32 | no (explicit) |
| Pythia-14M | `EleutherAI/pythia-14m` | 6 | 2 | `{} is called a "` | float32 | no (explicit) |

Protocol notes specific to this group:

- **Severed window.** The window preserves GPT-J's window-to-depth ratio of
  10/28 = 0.36, because that ratio sets how much of the network each severed
  restoration covers and therefore how much the layer distribution is smoothed.
  A window fixed in absolute layers smooths a shallow model far more than a
  deep one, which is exactly the quantity the cross-model comparison reads:

  ```
  w = 10                          for L >= 28
  w = max(2, round(10 * L / 28))  for L <  28
  ```

  This gives 10 for Qwen2.5-1.5B, Llama-3.2-3B, GPT-2 Large and Qwen2.5-3B,
  9 for Pythia-410M, 6 for Gemma-2B and the 16-layer models, and 2 for
  Pythia-14M.

  **Correction (October 2026).** An earlier version of this rule cut over at
  18 rather than 28 layers, which left Gemma-2B (L=18) at w = 10, i.e.
  w/L = 0.56 against 0.28–0.38 for every other run and 0.36 for GPT-J. Gemma
  was the only model in the set whose MLP layer distribution showed no early
  mode, and an over-wide window smooths an early mode away, so that null was
  confounded with the protocol. Gemma's MLP and attention traces are rerun at
  w = 6; its hidden-state traces are unaffected, because
  `trace_important_states` restores single states and never uses the window.
  Any Gemma MLP/attention npz stamped `window=10` predates the correction.

  **Reading the stored `window` on a hidden-state npz.** Every npz stores the
  `--window` the run was launched with, including the hidden-state traces,
  which do not use it. The Pythia-1B, Llama-3.2-1B and Qwen2.5-1.5B
  hidden-state files are stamped `window=0` because they were written by the
  original degenerate launch and were not retraced afterwards — correctly so,
  since `trace_important_states` restores single states and the window never
  entered the computation. Their scores are valid; only the stamp is stale.
  Judge a run's window by its `_mlp`/`_attn` files, which is what
  `analysis/dsra_cross_model.py` reports.

  The tracing script takes `--window` and stores the value
  in every npz, and `WINDOW=<n>` overrides it per run — for *every* model in
  that run, replacing each per-model value, so leave `WINDOW` unset unless you
  mean exactly that. Both the launcher and the tracer refuse a window below 2:
  such a window restores no layers, and the MLP/attention maps come out flat at
  `low_score` while still writing a full set of npz, so check the stored
  `window` field (and the "center of interval of N restored" axis label) before
  trusting a severed-window result. When comparing
  severed-window profiles across models, state the per-model window and use
  normalised depth, and treat the Pythia-14M MLP/attention decomposition as
  indicative only.
- **Screening yield for Pythia-14M** (only if run explicitly). Zero-shot IDM
  accuracy at 14M is very low; the builder may exhaust the candidate pool
  before reaching 800 kept facts. It writes however many it found (the log reports the yield), and
  the tracer runs on all of them. Report the obtained N in the chapter; if
  the yield is too small for stable aggregates (well under ~100), report
  Pythia-14M descriptively or drop it from the quantitative comparison, as
  the model's inclusion in COMPASS relied on role datasets with behavioural
  filtering, not open-vocabulary recall.
- **Pythia support** was already present in the tracing code: the GPT-NeoX
  tree (`gpt_neox.embed_in`, `gpt_neox.layers.{n}.attention`/`.mlp`) is
  handled by `layername()`, and the layer-count regex covers `gpt_neox`.
- **Environment.** All five run in the pinned rome environment
  (transformers 4.46.3 covers Pythia, Qwen2.5, and Llama-3.2; Llama-3.2-1B
  is gated like the 3B and needs the accepted licence plus `HF_TOKEN`).
- **Runtime.** These models are far smaller than the CAP set: expect a few
  seconds per case for Pythia-14M, tens of seconds for the ~1B models, and
  about half the Qwen2.5-3B time for Qwen2.5-1.5B — roughly 2–10 hours per
  model for 800 cases in float32, and minutes for the whole Pythia-14M run
  if the screening yield is small.
- **Noise.** `s3` recomputes 3 sigma per model as usual; record the printed
  value for each of the five models.

## 6. Sanity checks after the pilot

1. The facts log reports the zero-shot screening rate; expect roughly
   10–35 per cent of screened candidates kept (cf. the chapter 4 IDM
   baselines).
2. Every traced npz has `correct_prediction = True` (screened facts).
3. `low_score < high_score` in nearly all cases (corruption hurts).
4. Heatmaps show definition tokens marked with `*` (the corrupted span) and
   plausible late-layer recovery at the final token.
