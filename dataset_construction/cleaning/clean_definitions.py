"""Clean and select WordNet definitions.

Run from the repository root:

    # paper settings, on the extraction output
    python -m dataset_construction.cleaning.clean_definitions

    # with gloss repairs and WordNet / spaCy / NER annotations
    python -m dataset_construction.cleaning.clean_definitions --repair --annotate

    # the early selection (4-14 words, 80/10/10 split), on data/WordNet_filtered_data.json
    python -m dataset_construction.cleaning.clean_definitions --legacy
"""

import argparse

from .clean import clean
from .config import CleaningConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy", action="store_true",
                        help="Reproduce the early selection script.")
    parser.add_argument("--input", default=None, help="Definitions JSON to clean.")
    parser.add_argument("--out", default=None, help="Output JSON.")
    parser.add_argument("--min_words", type=int, default=None, help="Shortest definition kept.")
    parser.add_argument("--max_words", type=int, default=None, help="Longest definition kept.")
    parser.add_argument("--repair", action="store_true",
                        help="Fix gloss-merging artefacts before filtering.")
    parser.add_argument("--annotate", action="store_true",
                        help="Add WordNet, spaCy and NER annotations to the kept records.")
    parser.add_argument("--split", type=float, nargs=3, default=None,
                        metavar=("TRAIN", "TEST", "VALID"),
                        help="Also write train / test / validation files with these shares.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in dict(
        input=args.input, out=args.out, min_words=args.min_words, max_words=args.max_words,
        split=args.split,
    ).items() if v is not None}
    overrides.update(repair=args.repair, annotate=args.annotate, seed=args.seed)
    if args.shuffle:
        overrides["shuffle"] = True
    config = CleaningConfig.legacy(**overrides) if args.legacy else CleaningConfig(**overrides)
    clean(config)


if __name__ == "__main__":
    main()
