"""Configuration for building per-model fact files.

A fact file lists the definitions to trace for one model. Each record holds
the definition as it appears in the prompt (``subject``), the definiendum
(``attribute``) and the prompt. Prompts come from
``dataset_construction.prompts``; candidates come from the cleaned pool
(``dataset_construction.cleaning.build_pool``). With screening on (the
default), only definitions the model completes correctly zero-shot are kept.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataset_construction.prompts import get_template, resolve_model, uses_title_case

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
DTYPES = ("float32", "float16", "bfloat16")


@dataclass
class FactConfig:
    """Settings for one fact file.

    Attributes:
        model_name: Hugging Face id or CAP/CARMA alias; its tokenizer decides
            the single-token test.
        template: prompt with ``{}`` for the definition (default: the model's
            registered prompt).
        title_case: title-case the definition (default: the model's setting).
        definitions: extraction output (``wordnet_data_definitions.json``).
        out: output path (default: ``data/facts_<model>_n<max_facts>.json``).
        max_facts: stop after this many kept facts.
        max_words: longest definition kept, in space-separated words.
        max_screen: upper bound on candidates screened before giving up.
        screen: keep only candidates the model answers correctly zero-shot.
        torch_dtype: precision of the screening model.
        device: torch device of the screening model.
        seed: shuffle seed for the candidate order.
    """

    model_name: str
    template: Optional[str] = None
    title_case: Optional[bool] = None
    definitions: str = str(DATA / "wordnet_data_definitions.json")
    out: Optional[str] = None
    max_facts: int = 800
    max_words: int = 25
    max_screen: int = 40000
    screen: bool = True
    torch_dtype: str = "float32"
    device: str = field(default_factory=lambda: _default_device())
    seed: int = 42

    def __post_init__(self):
        self.model_name = resolve_model(self.model_name)
        if self.template is None:
            self.template = get_template(self.model_name)
        if "{}" not in self.template:
            raise ValueError("template must contain {}")
        if self.title_case is None:
            self.title_case = uses_title_case(self.model_name)
        if self.torch_dtype not in DTYPES:
            raise ValueError(f"torch_dtype must be one of {DTYPES}")
        if self.max_facts < 1 or self.max_words < 1:
            raise ValueError("max_facts and max_words must be positive")
        if self.out is None:
            tag = self.model_name.replace("/", "_")
            self.out = str(DATA / f"facts_{tag}_n{self.max_facts}.json")


def _default_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
