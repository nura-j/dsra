"""Extract WordNet definitions to JSON.

Run from the repository root todo: add this to readme and requirements.txt

    python -m dataset_construction.scripts.extract_definitions --language en
    python -m dataset_construction.scripts.extract_definitions --language es

The English run writes data/wordnet_data_definitions.json; the Spanish run
writes data/wordnet_es_data_definitions.json.
"""

import argparse

from dataset_construction.extraction import ExtractionConfig, extract_definitions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=["en", "es"], default="en",
                        help="English (EWN) or Spanish (SWN) WordNet.")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: data/<language file name>).")
    parser.add_argument("--no-masked", action="store_true",
                        help="Do not add masked definitions.")
    parser.add_argument("--n-mask", type=int, default=2,
                        help="Words masked per definition (default: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the masked positions (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = ExtractionConfig.english if args.language == "en" else ExtractionConfig.spanish
    config = preset(out=args.out, add_masked=not args.no_masked,
                    n_mask=args.n_mask, seed=args.seed)
    extract_definitions(config)


if __name__ == "__main__":
    main()
