"""Annotate every WordNet synset with an alphabetic name.

Run from the repository root: todo: add to requirement.txt

    python -m dataset_construction.cleaning.annotate_wordnet

Writes data/WordNet_filtered_data.json: per synset, the definiendum's part of
speech, frequencies, category and abstraction / generality scores, and per
definition token its part of speech, dependency label, NER tag and WordNet
attributes. Needs spaCy en_core_web_sm and the NLTK wordnet and brown corpora.
"""

import argparse

from .annotate import WordNetAnnotator
from .config import DATA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DATA / "WordNet_filtered_data.json"),
                        help="Output JSON (default: data/WordNet_filtered_data.json).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Annotate only the first N synsets (for a quick check).")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy pipeline.")
    parser.add_argument("--ner_model", default="dbmdz/bert-large-cased-finetuned-conll03-english",
                        help="Token-classification model for NER.")
    parser.add_argument("--ner_revision", default="4c53496", help="Model revision.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()
    annotator = WordNetAnnotator(args.spacy_model, args.ner_model, args.ner_revision,
                                 seed=args.seed)
    annotator.annotate_wordnet(out=args.out, limit=args.limit)


if __name__ == "__main__":
    main()
