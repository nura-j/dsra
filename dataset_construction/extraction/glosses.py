"""Merging multi-part WordNet glosses into one sentence, and masking words.
so all are combined to one defintion

WordNet glosses can hold several ';'-separated parts: the core definition,
usage domains ("in music", "slang", ...), and cross-references. The core is
kept first, domain labels are placed before it (prefix) or after it (postfix)
according to ``DOMAIN_HANDLING``, and cross-references are dropped.
"""

import random
import re

# Domain labels and where they go in the merged sentence.
DOMAIN_HANDLING = {
    "photography": "prefix",
    "Hebrew": "prefix",
    "Ethiopia": "prefix",
    "slang": "postfix",
    "genetics": "prefix",
    "cytology": "prefix",
    "Judaism": "prefix",
    "astronomy": "prefix",
    "British": "prefix",
    "language": "postfix",
    "dentistry": "prefix",
    "anatomy": "prefix",
    "Italian": "prefix",
    "computing": "prefix",
    "physics": "prefix",
    "philosophy": "prefix",
    "Christianity": "prefix",
    "French": "prefix",
    "phonology": "prefix",
    "radiology": "prefix",
    "theology": "prefix",
    "electricity": "prefix",
    "rhetoric": "prefix",
    "mineralogy": "prefix",
    "neuroscience": "prefix",
    "psychoanalysis": "prefix",
    "antiquity": "prefix",
    "architecture": "prefix",
    "printing": "prefix",
    "dressage": "prefix",
    "computer": "prefix",
    "biochemistry": "prefix",
    "metaphor": "postfix",
    "geology": "prefix",
    "Yiddish": "prefix",
    "colloquial": "postfix",
    "Zoroastrianism": "prefix",
    "agriculture": "prefix",
    "embryology": "prefix",
    "military": "prefix",
    "economics": "prefix",
    "ballet": "prefix",
    "ethics": "prefix",
    "vertebrates": "prefix",
    "chess": "prefix",
    "ecology": "prefix",
    "biology": "prefix",
    "cards": "prefix",
    "archeology": "prefix",
    "England": "prefix",
    "neurophysiology": "prefix",
    "construction": "prefix",
    "thermodynamics": "prefix",
    "mythology": "prefix",
    "Gnosticism": "prefix",
    "microbiology": "prefix",
    "film": "prefix",
    "golf": "prefix",
    "physiology": "prefix",
    "craps": "prefix",
    "bookkeeping": "prefix",
    "Britain": "prefix",
    "music": "prefix",
    "rugby": "prefix",
    "microscopy": "prefix",
    "euphemism": "postfix",
    "furniture": "prefix",
    "cosmology": "prefix",
    "voodooism": "prefix",
    "nautical": "prefix",
    "Hinduism": "prefix",
    "postpositive": "postfix",
    "boxing": "prefix",
    "weapons": "prefix",
    "counterpoint": "prefix",
    "taxes": "prefix",
    "baseball": "prefix",
    "accounting": "prefix",
    "fencing": "prefix",
    "pharmacology": "prefix",
    "epidemiology": "prefix",
    "bowling": "prefix",
    "mechanics": "prefix",
    "color": "postfix",
    "soccer": "prefix",
    "Apocrypha": "prefix",
    "geometry": "prefix",
    "statistics": "prefix",
    "diplomacy": "prefix",
    "Luke": "prefix",
    "obstetrics": "prefix",
    "grammar": "prefix",
    "zoology": "prefix",
    "trademark": "postfix",
    "politics": "prefix",
    "India": "prefix",
    "chemistry": "prefix",
    "technology": "prefix",
    "horseshoes": "prefix",
    "Russia": "prefix",
    "theater": "prefix",
    "logic": "prefix",
    "astrology": "prefix",
    "tennis": "prefix",
    "immunology": "prefix",
    "games": "prefix",
    "poker": "prefix",
    "finance": "prefix",
    "polo": "prefix",
    "linguistics": "prefix",
    "formerly": "postfix",
    "Brazil": "prefix",
    "Latin": "prefix",
    "Marxism": "prefix",
    "metonymy": "postfix",
    "medicine": "prefix",
    "religion": "prefix",
    "ophthalmology": "prefix",
    "phonetics": "prefix",
    "business": "prefix",
    "meteorology": "prefix",
    "Haiti": "prefix",
    "psychology": "prefix",
    "mathematics": "prefix",
    "obsolete": "postfix",
    "meat": "prefix",
    "poetic": "postfix",
    "verse": "prefix",
    "game": "prefix",
    "folklore": "prefix",
    "angling": "prefix",
    "archaic": "postfix",
    "formal": "postfix",
    "ornithology": "prefix",
    "Buddhism": "prefix",
    "Akkadian": "prefix",
    "elections": "prefix",
    "bridge": "prefix",
    "spiritualism": "prefix",
    "paleontology": "prefix",
    "corporation": "prefix",
    "prosody": "prefix",
    "pathology": "prefix",
    "historically": "postfix",
    "broadcasting": "prefix",
    "sport": "prefix",
    "law": "prefix",
    "parliament": "prefix",
    "figurative": "postfix",
    "falconry": "prefix",
    "sociology": "prefix",
    "forestry": "prefix",
    "cricket": "prefix",
    "surgery": "prefix",
    "facetious": "postfix",
    "toxicology": "prefix",
    "basketball": "prefix",
    "Briticism": "postfix",
    "orthopedics": "prefix",
    "pregnancy": "prefix",
    "legend": "prefix",
    "sports": "prefix",
    "government": "prefix",
    "rare": "postfix",
    "Fungi": "prefix",
    "psychiatry": "prefix",
    "botany": "prefix",
    "electronics": "prefix",
    "neurology": "prefix",
    "football": "prefix",
    "optics": "prefix",
    "Islam": "prefix",
    "plural": "postfix",
    "art": "prefix",
    "anthropology": "prefix",
    "metaphysics": "prefix",
    "Bible": "prefix",
    "poetry": "prefix",
    "informal": "postfix",
    "histology": "prefix",
    "mining": "prefix",
    "mountaineering": "prefix",
    "literary": "postfix",
    "heraldry": "prefix",
}


def combine_wordnet_definitions(definition):
    """Merge a multi-part WordNet definition string into one readable sentence.

    Splits on ';', drops empty / cross-reference fragments, and re-orders the
    remaining pieces into prefix / core / postfix order per ``DOMAIN_HANDLING``.
    """
    parts = re.split(r"[;]+", definition)

    # Remove empty parts and parts starting with '-'
    parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith("-")]

    # Keep the first part as the core, and process the rest
    core = parts.pop(0).strip() if parts else ""

    # Process remaining parts, ignoring any that contain 'followed by'
    parts = [p.strip() for p in parts if "followed by" not in p]

    prefixes = [
        p
        for p in parts
        if any(d in p for d in DOMAIN_HANDLING if DOMAIN_HANDLING[d] == "prefix")
    ]
    postfixes = [
        p
        for p in parts
        if any(d in p for d in DOMAIN_HANDLING if DOMAIN_HANDLING[d] == "postfix")
    ]

    # Remaining parts that are not prefixes or postfixes
    others = [p for p in parts if p not in prefixes + postfixes]
    others_part = (
        " and ".join(others).strip() if len(others) > 1 else " ".join(others).strip()
    )

    prefix_part = f"in {', '.join(prefixes)}, " if prefixes else ""
    postfix_part = " ".join(postfixes).strip()

    combined_sentence = f"{prefix_part}{core}"
    if others_part:
        combined_sentence += f" and {others_part}"
    if postfix_part:
        combined_sentence += f" and {postfix_part}"

    return combined_sentence.strip()


def corrupt_definition(defn, n_mask=2, mask="_", rng=None):
    """Replace ``n_mask`` randomly chosen words of ``defn`` with ``mask``.

    Returns None for one-word definitions. ``n_mask`` is lowered for short
    definitions so at least one word stays unmasked. ``rng`` is a
    ``random.Random``; without it the global ``random`` state is used.
    """
    words = defn.split()
    if len(words) == 1:
        return None
    if len(words) <= n_mask:
        while len(words) <= n_mask:
            n_mask -= 1
            if n_mask <= 0:
                return None
    mask_indices = (rng or random).sample(range(len(words)), k=n_mask)
    for i in mask_indices:
        words[i] = mask
    return " ".join(words)
