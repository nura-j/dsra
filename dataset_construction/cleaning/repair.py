"""Optional repairs of gloss-merging artefacts (off by default).

``combine_wordnet_definitions`` writes domain labels as ``"in <label>, "``.
Three patterns in the English extraction output come from that step:

- ``double_in``: the label already starts with "in" -> ``"in in ..."``
  (6 definitions), e.g. ``"in in India and Malaysia, a warehouse"``.
- ``repeated_label``: a part matching both a prefix and a postfix domain is
  written twice (29), e.g. ``"in British informal, pretty and neat and British informal"``.
- ``function_word_prefix``: a usage note such as "of business" or "used of
  persons" is treated as a domain (82), e.g. ``"in of business, not active or brisk"``;
  the added "in" is dropped.

The data used in the paper were built without these repairs.
"""

import re
from .config import REPAIR_RULES

# A domain label never starts with these words; a usage note can.
FUNCTION_WORDS = {
    "of", "for", "by", "with", "as", "on", "a", "an", "the", "to", "from", "at",
    "used", "especially", "usually", "often", "sometimes", "or", "and", "when",
    "after", "before", "followed", "preceded", "also", "not", "that", "which",
    "who", "being", "is", "are",
}

_PREFIX = re.compile(r"^in ([^,]+), ")
_REPEATED = re.compile(r"^in (.+?), (.*) and (.+)$")


def repair_double_in(definition):
    return "in " + definition[len("in in "):] if definition.startswith("in in ") else definition


def repair_repeated_label(definition):
    m = _REPEATED.match(definition)
    if m and m.group(3) == m.group(1):
        return f"in {m.group(1)}, {m.group(2)}"
    return definition


def repair_function_word_prefix(definition):
    m = _PREFIX.match(definition)
    if m and m.group(1).split()[0].lower() in FUNCTION_WORDS:
        return definition[len("in "):]
    return definition


RULES = {
    "double_in": repair_double_in,
    "repeated_label": repair_repeated_label,
    "function_word_prefix": repair_function_word_prefix,
}


def repair_definition(definition, rules=REPAIR_RULES):
    for name in rules:
        definition = RULES[name](definition)
    return definition


def repair_records(records, field="definition", rules=REPAIR_RULES):
    """``(records, n_changed)`` with the definitions repaired."""
    out, changed = [], 0
    for rec in records:
        fixed = repair_definition(rec[field], rules)
        changed += fixed != rec[field]
        out.append({**rec, field: fixed})
    return out, changed
