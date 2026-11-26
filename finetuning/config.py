"""Configuration for fine-tuning on the inverse dictionary task.
I added the following settings:
- ``config``: ``FinetuneConfig`` .
CAP and CARMA settings for consistency with other repos
- ``config``: ``FinetuneConfig`` .
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

METHODS = ("lora", "full")
LOSSES = ("sequence", "target")
DTYPES = ("float32", "float16", "bfloat16")

SPANISH_LEGACY_FIELDS = ("spanish_definition", "spanish_name")
# Tokenizer settings of the original LoRA scripts.
PEFT_TOKENIZER = {"padding_side": "left", "add_eos_token": True, "add_bos_token": True}


@dataclass
class FinetuneConfig:
    """Settings for one fine-tuning run.

    Attributes:
        model_name: Hugging Face id or alias.
        method: ``"lora"`` (adapters) or ``"full"`` (all weights).
        loss: ``"sequence"`` (whole text) or ``"target"`` (definiendum only).
        data_path: definitions JSON; None reads raw glosses from ``saf_datasets``.
        language: ``saf_datasets`` set to read when ``data_path`` is None.
        fields: (definition key, definiendum key) of the JSON records.
        template, title_case: prompt; None takes the model's registered prompt.
        min_words, max_words: inclusive definition-length bounds (None = no bound).
        single_token: keep definienda that are one tokenizer-vocabulary entry.
        alphabetic: keep alphabetic definienda only.
        exclude_in_definition: drop items whose definiendum occurs in the definition.
        test_size: held-out share of the first split.
        valid_share: share of the held-out part used for validation; None
            evaluates on the test part during training (the LoRA scripts).
        max_len_ratio: target loss only; padded length covers this share of prompts.
        r, lora_alpha, lora_dropout, target_modules: LoRA settings
            (``target_modules=None`` lets PEFT choose for the architecture).
        adapter_path: existing LoRA adapter, merged into the model before use.
        epochs, batch_size, learning_rate, weight_decay: optimisation.
        eval_strategy, save_strategy: ``"steps"`` or ``"epoch"``.
        save_steps, logging_steps, save_total_limit, load_best_model_at_end: checkpointing.
        tokenizer_kwargs: extra ``AutoTokenizer.from_pretrained`` arguments.
        torch_dtype: precision of the model weights.
        train, test: run training and / or the next-token evaluation.
        output_dir: root folder for checkpoints, weights and results.
        device: torch device (``"auto"`` spreads the model over the GPUs).
        seed: seed for the split, data order and initialisation.
    """

    model_name: str
    method: str = "lora"
    loss: str = "sequence"
    data_path: Optional[str] = None
    language: str = "en"
    fields: Tuple[str, str] = ("definition", "word")
    template: Optional[str] = None
    title_case: Optional[bool] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    single_token: bool = False
    alphabetic: bool = False
    exclude_in_definition: bool = False
    test_size: float = 0.2
    valid_share: Optional[float] = None
    max_len_ratio: float = 0.95
    r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[Tuple[str, ...]] = None
    adapter_path: Optional[str] = None
    epochs: int = 10
    batch_size: int = 16
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_steps: int = 2000
    logging_steps: int = 2000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    tokenizer_kwargs: dict = field(default_factory=dict)
    torch_dtype: str = "float32"
    train: bool = True
    test: bool = True
    output_dir: str = "output"
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.torch_dtype not in DTYPES:
            raise ValueError(f"torch_dtype must be one of {DTYPES}")
        if self.language not in ("en", "es"):
            raise ValueError('language must be "en" or "es"')
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if self.valid_share is not None and not 0 < self.valid_share < 1:
            raise ValueError("valid_share must be in (0, 1)")
        if (self.min_words is not None and self.max_words is not None
                and not 1 <= self.min_words <= self.max_words):
            raise ValueError("need 1 <= min_words <= max_words")
        if not 0 < self.max_len_ratio <= 1:
            raise ValueError("max_len_ratio must be in (0, 1]")
        self.fields = tuple(self.fields)
        if self.target_modules is not None:
            self.target_modules = tuple(self.target_modules)

    @classmethod
    def gptj_english(cls, **overrides) -> "FinetuneConfig":
        """GPT-J-6B on English WordNet (paper): LoRA r 64, alpha 32, dropout
        0.1, 10 epochs, batch 16, 80/20 split, definitions of 4-25 words."""
        settings = dict(
            model_name="EleutherAI/gpt-j-6B", method="lora", loss="sequence",
            data_path=None, language="en",
            min_words=4, max_words=25, single_token=True, alphabetic=True,
            exclude_in_definition=True, test_size=0.2,
            r=64, lora_alpha=32, lora_dropout=0.1,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            epochs=10, batch_size=16, tokenizer_kwargs=dict(PEFT_TOKENIZER),
        )
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def bertin_spanish(cls, data_path, **overrides) -> "FinetuneConfig":
        """BERTIN GPT-J-6B on Spanish WordNet (paper): LoRA r 16, alpha 32,
        dropout 0.3, 20 epochs, batch 16, 30/70 train/test, 3-24 words."""
        settings = dict(
            model_name="bertin-project/bertin-gpt-j-6B", method="lora", loss="sequence",
            data_path=data_path, fields=SPANISH_LEGACY_FIELDS,
            min_words=3, max_words=24, test_size=0.7,
            r=16, lora_alpha=32, lora_dropout=0.3, target_modules=None,
            epochs=20, batch_size=16, tokenizer_kwargs=dict(PEFT_TOKENIZER),
        )
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def cap(cls, model_name, data_path="data/wordnet_data_definitions.json",
            **overrides) -> "FinetuneConfig":
        """CAP / CARMA setting - not really needed for DSRA paper code: all weights, loss on the definiendum only,
        20 / 8 / 72 train / validation / test, 3 epochs, lr 5e-5, weight decay 0.01."""
        settings = dict(
            model_name=model_name, method="full", loss="target", data_path=data_path,
            test_size=0.8, valid_share=0.1, epochs=3, batch_size=16,
            learning_rate=5e-5, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="epoch",
            save_steps=1000, logging_steps=200, save_total_limit=1,
            load_best_model_at_end=False, output_dir="models", device="auto",
        )
        settings.update(overrides)
        return cls(**settings)

    @property
    def run_name(self) -> str:
        """Output sub-folder, e.g. ``EleutherAI_gpt_j_6B/lora``."""
        model = self.model_name.strip("/").replace("/", "_").replace("-", "_").replace(".", "_")
        return f"{model}/{self.method}"
