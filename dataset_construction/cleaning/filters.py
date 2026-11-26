"""Record filters and the per-model candidate pool.

Filters are applied in this order, which matters for
``first_definition_only`` (a word keeps the first definition that passes the
other filters, not simply its first definition):

1. alphabetic definiendum;
2. word already kept;
3. empty or too short / too long definition;
4. definiendum inside its own definition.
"""

import re
from .config import CleaningConfig


def is_alphabetic(word):
    """Lower-case letters only, as in the DSRA candidate pool."""
    return re.fullmatch(r"[a-z]+", word) is not None


def word_count_ok(definition, min_words=None, max_words=None):
    n = len(definition.split())
    if min_words is not None and n < min_words:
        return False
    if max_words is not None and n > max_words:
        return False
    return True


def self_referencing(word, definition, mode="token"):
    """True when the definiendum appears in its definition.

    ``"token"``: as one of its space-separated words; ``"substring"``:
    anywhere, also inside longer words; None: never.
    """
    if mode is None:
        return False
    if mode == "token":
        return word in definition.split()
    return word in definition


def filter_records(records, config: CleaningConfig = None):
    """Kept records, in input order; definitions are stripped of outer spaces."""
    config = config or CleaningConfig()
    wf, df = config.word_field, config.definition_field
    kept, seen = [], set()
    for rec in records:
        word = rec[wf]
        if config.alphabetic and not is_alphabetic(word):
            continue
        if config.first_definition_only and word in seen:
            continue
        definition = rec[df].strip()
        if not definition or not word_count_ok(definition, config.min_words, config.max_words):
            continue
        if self_referencing(word, definition, config.self_reference):
            continue
        kept.append({**rec, df: definition})
        seen.add(word)
    return kept


def build_pool(records, max_words=25):
    """``{word: {"definition", "wordnet_id"}}`` under the default (paper) filters,
    the candidate pool the fact builder draws from."""
    config = CleaningConfig(max_words=max_words)
    return {
        rec["word"]: {"definition": rec["definition"], "wordnet_id": rec.get("wordnet_id", "")}
        for rec in filter_records(records, config)
    }
