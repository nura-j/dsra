"""Build a per-model DSRA fact file.

Run from the repository root:

    python -m dataset_construction.facts.build_facts --model_name gpt2-large
    python -m dataset_construction.facts.build_facts --model_name GPT2_large --max_facts 25

The prompt comes from the model's registered template unless --template is
given. Output: data/facts_<model>_n<max_facts>.json (or --out).
"""

import argparse

from .builder import build_facts
from .config import DTYPES, FactConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_name", required=True,
                        help="Hugging Face id or CAP/CARMA alias.")
    parser.add_argument("--template", default=None,
                        help="Prompt with {} for the definition (default: registered prompt).")
    parser.add_argument("--title_case", choices=["yes", "no"], default=None,
                        help="Title-case the definition (default: the model's setting).")
    parser.add_argument("--defs", default=None,
                        help="Definitions JSON (default: data/wordnet_data_definitions.json).")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/facts_<model>_n<max_facts>.json).")
    parser.add_argument("--max_facts", type=int, default=800, help="Facts to keep (default: 800).")
    parser.add_argument("--max_words", type=int, default=25,
                        help="Longest definition kept, in words (default: 25).")
    parser.add_argument("--max_screen", type=int, default=40000,
                        help="Upper bound on candidates screened before giving up.")
    screen = parser.add_mutually_exclusive_group()
    screen.add_argument("--screen", dest="screen", action="store_true", default=True,
                        help="Keep only candidates the model answers correctly zero-shot (default).")
    screen.add_argument("--no_screen", dest="screen", action="store_false",
                        help="Keep every single-token candidate.")
    parser.add_argument("--torch_dtype", default="float32", choices=DTYPES,
                        help="Precision of the screening model.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in dict(
        template=args.template, definitions=args.defs, out=args.out, device=args.device,
        title_case=None if args.title_case is None else args.title_case == "yes",
    ).items() if v is not None}
    config = FactConfig(model_name=args.model_name, max_facts=args.max_facts,
                        max_words=args.max_words, max_screen=args.max_screen,
                        screen=args.screen, torch_dtype=args.torch_dtype, seed=args.seed,
                        **overrides)
    build_facts(config)


if __name__ == "__main__":
    main()
