# Analysis

This module aggregates the causal traces: how much of the recovery the top-K states hold, where those states sit in the network, and whether a run is sound.

The module is imported as `analysis`.

## Recovery score

For each restored state, the recovery score is

```
T = clip((p_patched - p_corrupted) / (p_clean - p_corrupted), 0, 1)
```

computed from a trace's `scores`, `low_score` and `high_score`. Cases whose clean-vs-corrupted gap is at or below `min_gap`, or whose total recovery is zero, are skipped. The trace heatmaps show the raw patched probability; `T` is used for the quantitative summaries.

## Configuration

```python
from analysis import AnalysisConfig

config = AnalysisConfig(
    results_dir="results",
    topk=10,              # K for the localisation of top-K states
    mass_ks=(10, 50),     # K values for the recovery-mass shares
    early=0.25,           # normalised depth below this is "early"
    upper=0.60,           # normalised depth above this is "upper"
    correct_only=True,
    min_gap=1e-6,
)
```

## Top-K recovery mass

Top-K selection is magnitude-based, so the share of a case's total recovery that its K best states hold is the statistic that justifies K:

```python
from analysis import mass_stats

stats = mass_stats("results/ns3_r0_gpt2-large")
# {"hidden": {"n_cases": ..., "top10_mass_pct_mean": ..., "top50_mass_pct_mean": ...,
#             "states_for_50pct_mass_median": ..., "states_for_80pct_mass_median": ...}, ...}
```

`archived_mass_stats(archive_root)` summarises the archived GPT-J-6B and BERTIN runs under `<archive_root>/results` (`ARCHIVED_RUNS`).

## Localisation

`analyse(run_root, config)` reports, per trace kind, the recovery-mass shares and where the top-K states of each case sit:

| Field | Meaning |
| --- | --- |
| `n_layers`, `window` | depth of the model and the stored severed window |
| `depth_mean`, `depth_median` | normalised depth `layer / (L - 1)` of the top-K states |
| `pct_early`, `pct_middle`, `pct_upper` | shares below `early`, between, above `upper` |
| `final_token_pct_mean` | share of the top-K states at the final prompt token |
| `peak_layer`, `layer_pct` | the layer distribution of the top-K states |

`cross_model(runs, config)` analyses every run of `runs` (results folder to display label, e.g. `CROSS_MODEL_RUNS`) found under `config.results_dir` and adds the derived fields: the run's window (from the MLP traces), its window-to-depth ratio, and a `bimodality` summary of each layer distribution (dip, balance, early mass, mode depths). A run with mixed windows in one trace kind is refused.

```python
from analysis import AnalysisConfig, CROSS_MODEL_RUNS, cross_model, plot_cross_model

res = cross_model(CROSS_MODEL_RUNS, AnalysisConfig(results_dir="results"))
plot_cross_model(res, "analysis/outputs/dsra_cross_model_layers.pdf")
```

The figure has one row per run and one column per kind, with normalised depth on the x axis, the GPT-J-6B bands reported in the paper in grey, and the mean depth of the top-K states as a dashed line.

## Grid shares

`grid_stats(data, tokenizer, layers=28)` describes the top-10 and top-50 thresholds as shares of the token x layer grid over the definition pool, with prompt-length statistics. It describes selectivity; the recovery mass is the justification for K.

## Run checks

```python
from analysis import check_run, problems

summary = check_run("results/ns3_r0_gpt2-large")
problems(summary)     # [] when the run is sound
```

Per kind: npz count, share with a correct clean prediction (all, for screened fact files), share where corruption lowers `p(answer)`, stored windows, and MLP / attention maps flat at the corrupted baseline. `problems` warns about wrong predictions, weak corruption, mixed windows, windows below 2 and flat maps.

## Command line

From the repository root:

```bash
python -m analysis.summarise cross-model --results results
python -m analysis.summarise topk-mass --results results --runs ns3_r0_gpt2-large
python -m analysis.summarise topk-mass --archive /path/to/rome-main
python -m analysis.summarise grid-stats
python -m analysis.summarise plot
python -m analysis.summarise check --results results
```

| Command | Purpose | Output |
| --- | --- | --- |
| `cross-model` | localisation and recovery mass per run and kind (`--topk`, `--derive-only`) | `analysis/outputs/dsra_cross_model.json` |
| `topk-mass` | recovery-mass shares of `--runs`, or of the archived runs with `--archive` (`--all-cases`) | `analysis/outputs/dsra_topk_mass.json` |
| `grid-stats` | grid shares over `--definitions`, prompt lengths with `--tokenizer` (default `gpt2`) | `analysis/outputs/dsra_grid_stats.json` |
| `plot` | figure from `--json` | `analysis/outputs/dsra_cross_model_layers.pdf` and `.png` |
| `check` | sanity checks of `--runs` | printed |

`--results` is the folder holding the runs (default `results`); without `--runs`, every registered cross-model run found there is used. `--out` changes the output file.
