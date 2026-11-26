"""Configuration for cleaning and selecting definitions.

The defaults are the DSRA paper settings, applied to the extraction output
(``data/wordnet_data_definitions.json``):

- alphabetic, lower-case definiendum;
- one definition per word (the first that passes the filters);
- definitions of at most 25 words;
- the definiendum is not one of the words of its own definition.

``CleaningConfig.legacy()`` reproduces the early selection script
(``data_selection.py``): 4-14 words, definiendum not a substring of the
definition, and an 80/10/10 train / test / validation split in file order,
read from the annotated WordNet file (``data/WordNet_filtered_data.json``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"

SELF_REFERENCE = ("token", "substring", None)
REPAIR_RULES = ("double_in", "repeated_label", "function_word_prefix")
SPLIT_NAMES = ("train", "test", "validation")


@dataclass
class CleaningConfig:
    """Settings for one cleaning run.

    Attributes:
        input: definitions JSON (a list of records).
        out: output JSON of the kept records.
        word_field, definition_field: record keys of the definiendum and definition.
        min_words, max_words: inclusive definition-length bounds (None = no bound).
        alphabetic: keep definienda made of lower-case letters only.
        first_definition_only: keep one definition per word (the first kept).
        self_reference: drop items whose definiendum is one of the definition's
            words (``"token"``), occurs anywhere in it (``"substring"``), or keep
            them (None).
        repair: fix gloss-merging artefacts before filtering (see ``repair.py``).
        repair_rules: which repairs to apply.
        annotate: add WordNet, spaCy and NER annotations (``annotate.py``) to
            every kept record, looked up by ``wordnet_id``.
        spacy_model, ner_model, ner_revision: annotation models.
        split: (train, test, validation) shares; None writes one file.
        shuffle: shuffle before splitting (the legacy split keeps file order).
        split_name: file name of each split, ``{stem}`` = output name, ``{part}`` = split.
        text_template: also write each split as text, one
            ``template.format(definition=..., word=...)`` line per record.
        seed: seed for shuffling.
    """

    input: str = str(DATA / "wordnet_data_definitions.json")
    out: str = str(DATA / "wordnet_definitions_clean.json")
    word_field: str = "word"
    definition_field: str = "definition"
    min_words: Optional[int] = None
    max_words: Optional[int] = 25
    alphabetic: bool = True
    first_definition_only: bool = True
    self_reference: Optional[str] = "token"
    repair: bool = False
    repair_rules: Tuple[str, ...] = REPAIR_RULES
    annotate: bool = False
    spacy_model: str = "en_core_web_sm"
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ner_revision: Optional[str] = "4c53496"
    split: Optional[Tuple[float, float, float]] = None
    shuffle: bool = False
    split_name: str = "{stem}_{part}"
    text_template: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.self_reference not in SELF_REFERENCE:
            raise ValueError(f"self_reference must be one of {SELF_REFERENCE}")
        unknown = set(self.repair_rules) - set(REPAIR_RULES)
        if unknown:
            raise ValueError(f"unknown repair rules {sorted(unknown)}; use {REPAIR_RULES}")
        self.repair_rules = tuple(self.repair_rules)
        if (self.min_words is not None and self.max_words is not None
                and not 1 <= self.min_words <= self.max_words):
            raise ValueError("need 1 <= min_words <= max_words")
        if self.split is not None:
            self.split = tuple(self.split)
            if len(self.split) != 3 or any(s < 0 for s in self.split) \
                    or abs(sum(self.split) - 1) > 1e-9:
                raise ValueError("split must be three non-negative shares summing to 1")

    @classmethod
    def legacy(cls, **overrides) -> "CleaningConfig":
        """The early selection: 4-14 words, substring self-reference check,
        80/10/10 split in file order, plus text files."""
        settings = dict(
            input=str(DATA / "WordNet_filtered_data.json"),
            out=str(DATA / "WordNet_definitions_filtered_data.json"),
            word_field="name",
            min_words=4, max_words=14,
            alphabetic=False, first_definition_only=False,
            self_reference="substring",
            split=(0.8, 0.1, 0.1), shuffle=False,
            split_name="WordNet_definitions_filtered_{part}_data",
            text_template="{definition} is commonly known as: {word}",
        )
        settings.update(overrides)
        return cls(**settings)

    def split_path(self, part, suffix=".json"):
        out = Path(self.out)
        return str(out.parent / (self.split_name.format(stem=out.stem, part=part) + suffix))
