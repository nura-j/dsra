# Reproducibility notes

This page records the experimental constants of the paper, what the archived results contain, and what this repository does not include.

## Constants

- **Analysis prompt (GPT-J-6B):** `"{definition} is often referred to as:"`, with the definition title-cased.
- **Corruption:** Gaussian noise on the definition-token embeddings, 3σₑ = 0.12922460585832596 for GPT-J-6B (ROME convention); 10 corrupted runs and 1 clean run per input.
- **Windows:** 10 layers for MLP and attention restorations; single states for hidden states.
- **Seed:** 42 for Python, NumPy, PyTorch and CUDA.
- **Heatmaps:** the colour is the raw patched-run probability `p(answer)`, with `vmin` at the corrupted baseline. The normalised recovery score `T` is used for the top-K summaries, not for the heatmap colours.

## Top-K thresholds

Top-K selection is magnitude-based, so the share of the recovery mass the top states capture justifies K. From the archived trace tensors (`python -m analysis.summarise topk-mass --archive <rome-main>`; stored in `analysis/outputs/dsra_topk_mass.json`):

- **English:** two runs (801 all-correct and 864 correct cases) agree within about 2 points. The top-10 states capture 78-84 % of the total recovery mass (means by component type) and the top-50 capture 97-99 %; the median case reaches 50 % of its mass within 4-7 states and 80 % within 9-11.
- **Spanish** (BERTIN final run, 108 usable correct cases): more diffuse; top-50 means of 82-89 %.

The grid shares (`analysis/outputs/dsra_grid_stats.json`: top-10 about 2.3 % and top-50 about 11.3 % of the token x layer cells, for prompts of 17 tokens on average over 28 layers) describe selectivity, not the justification for K.

## Recovered counts

- **Spanish WordNet:** n = 111 correctly predicted and traced definitions (of 229 traced in the final BERTIN run).
- **English WordNet:** the archived runs hold 801 (all-correct curated) and 864 (main-run correct) cases; neither reproduces the published n = 818 exactly, and the run behind that count was not located.

## Not included

- The exact published English split of 8,348 examples. The legacy cleaning preset (`CleaningConfig.legacy()`) reproduces an early selection script (4-14 words, 80/10/10 split) that does not yield 8,348.
- The DSR span labelling and the DSR / part-of-speech grouped aggregation of the top-K states.
- The prompts and filter outputs that generated the alternative definitions; their trace tensors exist in the archive as `knowledge_<N>_<M>.npz`.

## Provenance

The tracing code derives from ROME ([Meng et al., 2022](https://github.com/kmeng01/rome)); the helpers it uses are kept unchanged in `legacy/rome` under ROME's MIT licence. The full experiment archive (trace tensors, the Spanish pipeline data, fine-tuning logs) is kept outside this repository.
