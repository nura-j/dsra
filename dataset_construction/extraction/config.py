"""Configuration for extracting WordNet definitions.

The definitions come from ``saf_datasets``: ``WordNetFilteredDataSet``
(English WordNet, EWN) and ``SpanishWordNetFilteredDataSet`` (Spanish
WordNet, SWN). Both are filtered to definienda that are a single LLaMA token (todo: pass tokeniser of selected model),
and are cached under ``~/.saf_data/WordNet/`` after the first download.

Presets:

- ``ExtractionConfig.english()``: ``data/wordnet_data_definitions.json``, with
  merged glosses and a masked variant of each definition.
- ``ExtractionConfig.spanish()``: ``data/wordnet_es_data_definitions.json``,
  glosses kept as they are (the merge rules insert English words).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGES = ("en", "es")
OUTPUT_NAMES = {"en": "wordnet_data_definitions.json",
                "es": "wordnet_es_data_definitions.json"}


@dataclass
class ExtractionConfig:
    """Settings for one extraction run.

    Attributes:
        language: ``"en"`` (EWN) or ``"es"`` (SWN).
        out: output JSON path; defaults to ``data/<OUTPUT_NAMES[language]>``.
        combine_glosses: merge multi-part glosses with ``combine_wordnet_definitions``.
        add_masked: add a ``masked_definition`` and drop definitions that
            cannot be masked (one word long).
        n_mask: words masked per definition.
        mask: the mask string.
        seed: seed for the masked positions; None uses the global ``random``
            state, as the original script did.
    """

    language: str = "en"
    out: Optional[str] = None
    combine_glosses: bool = True
    add_masked: bool = True
    n_mask: int = 2
    mask: str = " <extra_id_0> "
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        if self.n_mask < 1:
            raise ValueError("n_mask must be at least 1")
        if self.out is None:
            self.out = str(REPO_ROOT / "data" / OUTPUT_NAMES[self.language])

    @classmethod
    def english(cls, **overrides) -> "ExtractionConfig":
        """EWN with merged glosses and masked definitions."""
        settings = dict(language="en", combine_glosses=True)
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def spanish(cls, **overrides) -> "ExtractionConfig":
        """SWN, glosses as they are."""
        settings = dict(language="es", combine_glosses=False)
        settings.update(overrides)
        return cls(**settings)
