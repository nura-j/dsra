"""WordNet definition extraction (stage 1 of dataset construction)."""

from .config import LANGUAGES, OUTPUT_NAMES, ExtractionConfig
from .glosses import DOMAIN_HANDLING, combine_wordnet_definitions, corrupt_definition
from .wordnet import extract_definitions, load_definitions, load_source, to_record

__all__ = [
    "LANGUAGES",
    "OUTPUT_NAMES",
    "ExtractionConfig",
    "DOMAIN_HANDLING",
    "combine_wordnet_definitions",
    "corrupt_definition",
    "extract_definitions",
    "load_definitions",
    "load_source",
    "to_record",
]
