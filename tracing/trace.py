"""Trace a fact file: corrupt the definition, restore one state at a time.

Run from the repository root:

    python -m tracing.trace --model_name gpt2-large --fact_file data/facts_gpt2-large_n800.json

    # GPT-J-6B as in the paper: fixed noise 3 sigma_e and a 10-layer window
    python -m tracing.trace --model_name EleutherAI/gpt-j-6B \
        --fact_file data/known_data_wordNet.json --noise_level 0.12922460585832596

Writes results/n<noise>_r<replace>_<model>/causal_trace/cases/knowledge_<id>[_mlp|_attn].npz
and heatmaps under .../pdfs. Existing npz files are reused, so an interrupted
run can be resumed.
"""

import argparse

from transformers import AutoConfig

from .config import DTYPES, KINDS, TraceConfig, parse_noise, window_for
from .runner import load_facts, trace_facts


def num_layers(model_name):
    """Transformer layer count from the model config (no weights loaded)."""
    return AutoConfig.from_pretrained(model_name).num_hidden_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6B", help="Hugging Face id.")
    parser.add_argument("--fact_file", default="data/known_data_wordNet.json",
                        help="Fact records (known_id, subject, attribute, prompt).")
    parser.add_argument("--output_dir", default="results/{model_name}/causal_trace",
                        help="Results folder; {model_name} becomes n<noise>_r<replace>_<model>.")
    parser.add_argument("--noise_level", default="s3", type=parse_noise,
                        help='Noise rule: a float, "s<k>" (k x embedding std), "m", "t<df>" '
                             'or "u<scale>" (default: s3).')
    parser.add_argument("--replace", default=0, type=int,
                        help="1 replaces the definition embeddings with noise instead of adding it.")
    parser.add_argument("--num_cases", default=-1, type=int,
                        help="Trace only the first N facts (default: all).")
    parser.add_argument("--torch_dtype", default="auto", choices=DTYPES,
                        help="Model precision (auto: float32, float16 for 20B models).")
    parser.add_argument("--window", default=None, type=int,
                        help="Severed-window width for MLP / attention (default: 10 for 28+ "
                             "layers, scaled to depth below that).")
    parser.add_argument("--samples", default=10, type=int,
                        help="Corrupted runs per input (default: 10).")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS,
                        help="Trace kinds to run (default: all three).")
    parser.add_argument("--max_plot_id", default=200, type=int,
                        help="Draw heatmaps for facts with known_id up to this.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    return parser.parse_args()


def main():
    args = parse_args()
    window = args.window if args.window is not None else window_for(num_layers(args.model_name))
    overrides = {"device": args.device} if args.device else {}
    config = TraceConfig(
        model_name=args.model_name,
        noise=args.noise_level,
        samples=args.samples,
        window=window,
        kinds=tuple(args.kinds),
        replace=bool(args.replace),
        torch_dtype=args.torch_dtype,
        num_cases=args.num_cases,
        max_plot_id=args.max_plot_id,
        output_dir=args.output_dir.replace("{model_name}", "{run_name}"),
        **overrides,
    )
    print(f"Tracing {config.model_name}: noise {config.noise}, window {config.window}, "
          f"kinds {', '.join(config.kinds)} -> {config.run_dir}")
    correct = trace_facts(config, load_facts(args.fact_file))
    print(f"Done: {len(correct)} correctly predicted facts traced.")


if __name__ == "__main__":
    main()
