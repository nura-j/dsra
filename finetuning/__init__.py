"""Fine-tuning on the inverse dictionary task.

- ``config``: ``FinetuneConfig`` .
- ``data``: loading, filtering, splitting, tokenisation.
- ``models``: base model, tokenizer and LoRA adapters.
- ``trainer``: training arguments, training and saving.
- ``evaluation``: single next-token accuracy and the report of correct items.
- ``pipeline``: ``run(config)``.
- ``finetune``: the command line.
"""

from .config import FinetuneConfig
from .data import (
    TargetOnlyDataset,
    filter_records,
    load_records,
    split_records,
    target_datasets,
    target_max_length,
    test_prompts,
    tokenize_sequences,
    tokenize_targets,
)
from .evaluation import evaluate, write_report
from .pipeline import run, set_seed
from .models import load_model, load_tokenizer
from .trainer import save, train, training_arguments

__all__ = [
    "FinetuneConfig",
    "TargetOnlyDataset",
    "filter_records",
    "load_records",
    "split_records",
    "target_datasets",
    "target_max_length",
    "test_prompts",
    "tokenize_sequences",
    "tokenize_targets",
    "evaluate",
    "write_report",
    "run",
    "set_seed",
    "load_model",
    "load_tokenizer",
    "save",
    "train",
    "training_arguments",
]
