"""Cleaning and selection of definitions (stage 2 of dataset construction).

- ``filters``: record filters and the candidate pool (paper settings by default).
- ``repair``: optional fixes of gloss-merging artefacts.
- ``annotate``: WordNet, spaCy and NER annotations.
- ``splits``: train / test / validation files.
- ``clean``: the pipeline; ``clean_definitions`` and ``annotate_wordnet``: command lines.
"""

from .annotate import WordNetAnnotator
from .clean import clean, load_records
from .config import REPAIR_RULES, SELF_REFERENCE, SPLIT_NAMES, CleaningConfig
from .filters import build_pool, filter_records, is_alphabetic, self_referencing, word_count_ok
from .repair import FUNCTION_WORDS, repair_definition, repair_records
from .splits import split_records, write_splits

__all__ = [
    "WordNetAnnotator",
    "clean",
    "load_records",
    "REPAIR_RULES",
    "SELF_REFERENCE",
    "SPLIT_NAMES",
    "CleaningConfig",
    "build_pool",
    "filter_records",
    "is_alphabetic",
    "self_referencing",
    "word_count_ok",
    "FUNCTION_WORDS",
    "repair_definition",
    "repair_records",
    "split_records",
    "write_splits",
]
