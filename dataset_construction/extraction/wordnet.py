"""Extract all WordNet definitions into a JSON list of records.

Each record holds ``word`` (the definiendum), ``definition``, ``wordnet_id``
and ``masked_definition`` (None when masking is off). The English output is
``data/wordnet_data_definitions.json`` (84,423 records).
"""

import json
import random
from pathlib import Path

from .config import ExtractionConfig
from .glosses import combine_wordnet_definitions, corrupt_definition


def load_source(language):
    """The ``saf_datasets`` WordNet set for ``language`` (downloads on first use)."""
    if language == "en":
        from saf_datasets import WordNetFilteredDataSet

        return WordNetFilteredDataSet()
    from saf_datasets.data_access.wordnet_spanish import SpanishWordNetFilteredDataSet

    return SpanishWordNetFilteredDataSet()


def to_record(sentence, config: ExtractionConfig, rng=None):
    """One record from a dataset sentence, or None if it cannot be masked."""
    definition = str(sentence.surface)
    if config.combine_glosses:
        definition = combine_wordnet_definitions(definition)
    masked = None
    if config.add_masked:
        masked = corrupt_definition(definition, n_mask=config.n_mask, mask=config.mask, rng=rng)
        if masked is None:
            return None
    return {
        "word": sentence.annotations["definiendum"],
        "definition": definition,
        "wordnet_id": sentence.annotations["id"],
        "masked_definition": masked,
    }


def extract_definitions(config: ExtractionConfig = None, dataset=None, write=True):
    """Build the definition records and (optionally) write them to ``config.out``.

    ``dataset`` defaults to the ``saf_datasets`` set for ``config.language``.
    """
    config = config or ExtractionConfig()
    if dataset is None:
        dataset = load_source(config.language)
    rng = random.Random(config.seed) if config.seed is not None else None
    records = []
    for sentence in dataset:
        record = to_record(sentence, config, rng)
        if record is not None:
            records.append(record)
    if write:
        out = Path(config.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Wrote {len(records)} definitions to {out}")
    return records


def load_definitions(path):
    """Records of a definitions JSON file."""
    with open(path) as f:
        return json.load(f)
