"""Cleaning pipeline: load, repair (optional), filter, annotate (optional), write.

With ``config.split`` the kept records are also written as train / test /
validation files (and text files with ``config.text_template``).
"""

import json
from pathlib import Path

from .config import CleaningConfig
from .filters import filter_records
from .repair import repair_records
from .splits import split_records, write_splits


def load_records(path):
    with open(path) as f:
        return json.load(f)


def clean(config: CleaningConfig = None, records=None, annotator=None, write=True):
    """Run the pipeline; returns the kept records (and writes them).

    ``records`` replaces reading ``config.input``; ``annotator`` is a
    ``WordNetAnnotator`` to reuse (one is created when ``config.annotate``).
    """
    config = config or CleaningConfig()
    if records is None:
        records = load_records(config.input)
    print(f"Loaded {len(records)} records")

    if config.repair:
        records, changed = repair_records(records, config.definition_field, config.repair_rules)
        print(f"Repaired {changed} definitions ({', '.join(config.repair_rules)})")

    kept = filter_records(records, config)
    print(f"Kept {len(kept)} records after filtering")

    if config.annotate:
        if annotator is None:
            from .annotate import WordNetAnnotator

            annotator = WordNetAnnotator(config.spacy_model, config.ner_model,
                                         config.ner_revision, seed=config.seed)
        kept = annotator.annotate_records(kept, definition_field=config.definition_field)

    if write:
        out = Path(config.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(kept, f, indent=4)
        print(f"Wrote {out}")
        if config.split is not None:
            parts = split_records(kept, config.split, config.shuffle, config.seed)
            write_splits(parts, config)
            print(", ".join(f"{k}: {len(v)}" for k, v in parts.items()))
    return kept
