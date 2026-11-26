"""Aggregate DSRA causal traces.

Run from the repository root:

    # localisation and recovery mass for the cross-model runs under results/
    python -m analysis.summarise cross-model --results results

    # recovery-mass shares of chosen runs (or of the archived GPT-J / BERTIN runs)
    python -m analysis.summarise topk-mass --results results --runs ns3_r0_gpt2-large
    python -m analysis.summarise topk-mass --archive /path/to/rome-main

    # top-K thresholds as grid shares, over the definition pool
    python -m analysis.summarise grid-stats

    # figure from the cross-model summary
    python -m analysis.summarise plot

    # sanity checks after a (pilot) run
    python -m analysis.summarise check --results results --runs ns3_r0_gpt2-large

Summaries are written to analysis/outputs/ unless --out is given.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .checks import check_run, problems
from .config import CROSS_MODEL_RUNS, AnalysisConfig
from .grid_stats import grid_stats
from .localisation import add_derived, cross_model
from .recovery import archived_mass_stats, mass_stats

OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFINITIONS = Path(__file__).resolve().parents[1] / "data" / "wordnet_data_definitions.json"


def to_json(value):
    """NumPy scalars as plain numbers (shortest repr, so rounded values stay rounded)."""
    if isinstance(value, np.floating):
        return float(str(value))
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def write(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=to_json)
    print("wrote", path)


def selected_runs(args):
    """Results folder -> label, for --runs or for every registered run found."""
    if args.runs:
        return {run: CROSS_MODEL_RUNS.get(run, run) for run in args.runs}
    return {run: label for run, label in CROSS_MODEL_RUNS.items()
            if (Path(args.results) / run).is_dir()}


def cmd_cross_model(args):
    out = args.out or OUTPUTS / "dsra_cross_model.json"
    if args.derive_only:
        with open(out) as f:
            write(add_derived(json.load(f)), out)
        return
    config = AnalysisConfig(results_dir=args.results, topk=args.topk)
    res = cross_model(selected_runs(args), config)
    for v in res.values():
        h, m = v.get("hidden", {}), v.get("mlp", {})
        print(f"{v['label']:14s} L={m.get('n_layers', 0):2d} w={m.get('window') or 0:2d} "
              f"n={h.get('n_cases', 0):4d} hidden depth={h.get('depth_mean')} "
              f"final-tok={h.get('final_token_pct_mean')}%")
    write(res, out)


def cmd_topk_mass(args):
    out = args.out or OUTPUTS / "dsra_topk_mass.json"
    if args.archive:
        res = archived_mass_stats(args.archive)
    else:
        config = AnalysisConfig(correct_only=not args.all_cases)
        res = {run: mass_stats(str(Path(args.results) / run), config)
               for run in selected_runs(args)}
    print(json.dumps(res, indent=1, default=to_json))
    write(res, out)


def cmd_grid_stats(args):
    from transformers import AutoTokenizer

    with open(args.definitions) as f:
        data = json.load(f)
    res = grid_stats(data, AutoTokenizer.from_pretrained(args.tokenizer), layers=args.layers)
    for variant, row in res.items():
        print(variant, row)
    write(res, args.out or OUTPUTS / "dsra_grid_stats.json")


def cmd_plot(args):
    from .visualization import plot_cross_model

    with open(args.json) as f:
        data = json.load(f)
    out = plot_cross_model(data, args.out)
    print("wrote", out, "and", out.with_suffix(".png"))


def cmd_check(args):
    ok = True
    for run in selected_runs(args):
        summary = check_run(str(Path(args.results) / run))
        print(f"== {run}")
        for kind, s in summary.items():
            print(f"  {kind:6s} npz={s['npz_files']:4d} correct={s['correct_pct']}% "
                  f"corruption-lowers={s['corruption_lowers_pct']}% windows={s['windows']}"
                  + (f" flat={s['flat_maps']}" if s["flat_maps"] is not None else ""))
        for p in problems(summary):
            ok = False
            print("  WARNING", p)
    if ok:
        print("no problems found")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def runs_args(p):
        p.add_argument("--results", default="results", help="Folder holding the runs.")
        p.add_argument("--runs", nargs="+", default=None,
                       help="Run folders (default: every registered cross-model run found).")

    p = sub.add_parser("cross-model", help="Localisation and recovery mass per run and kind.")
    runs_args(p)
    p.add_argument("--topk", type=int, default=10, help="K for the top-K states (default: 10).")
    p.add_argument("--derive-only", action="store_true",
                   help="Recompute the derived fields of an existing summary (no npz reads).")
    p.add_argument("--out", default=None, help="Output JSON.")
    p.set_defaults(func=cmd_cross_model)

    p = sub.add_parser("topk-mass", help="Share of the recovery mass in the top-10 / top-50 states.")
    runs_args(p)
    p.add_argument("--archive", default=None,
                   help="rome-main archive root: summarise the archived GPT-J / BERTIN runs.")
    p.add_argument("--all-cases", action="store_true",
                   help="Include cases with a wrong clean prediction.")
    p.add_argument("--out", default=None, help="Output JSON.")
    p.set_defaults(func=cmd_topk_mass)

    p = sub.add_parser("grid-stats", help="Top-K thresholds as shares of the token x layer grid.")
    p.add_argument("--definitions", default=str(DEFINITIONS), help="Definitions JSON.")
    p.add_argument("--tokenizer", default="gpt2",
                   help="Tokenizer for prompt lengths (GPT-2 shares GPT-J's vocabulary).")
    p.add_argument("--layers", type=int, default=28, help="Layers of the traced model (default: 28).")
    p.add_argument("--out", default=None, help="Output JSON.")
    p.set_defaults(func=cmd_grid_stats)

    p = sub.add_parser("plot", help="Layer-distribution figure from the cross-model summary.")
    p.add_argument("--json", default=str(OUTPUTS / "dsra_cross_model.json"), help="Summary JSON.")
    p.add_argument("--out", default=str(OUTPUTS / "dsra_cross_model_layers.pdf"),
                   help="Output PDF (a PNG is written next to it).")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("check", help="Sanity checks after a run.")
    runs_args(p)
    p.set_defaults(func=cmd_check)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
