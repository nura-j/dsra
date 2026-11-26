"""Fine-tuning pipeline: load, filter, split, train, save, evaluate.

Outputs go to <output_dir>/<model>/<method>/<timestamp>_*: checkpoints,
output_results, and weights_unmerged_lora + weights_merged_lora (LoRA) or
weights (full fine-tuning).
"""

import os
import random
from datetime import datetime

import numpy as np
import torch

from dataset_construction.prompts import PromptConfig

from .config import FinetuneConfig
from .data import (
    filter_records,
    load_records,
    split_records,
    target_datasets,
    test_prompts,
    tokenize_sequences,
)
from .evaluation import evaluate, write_report
from .models import load_model, load_tokenizer
from .trainer import save, train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def output_paths(config: FinetuneConfig, timestamp):
    base = os.path.join(config.output_dir, config.run_name, f"{timestamp}_")
    lora = config.method == "lora"
    paths = {
        "checkpoints": base + "checkpoints",
        "unmerged": base + "weights_unmerged_lora" if lora else None,
        "merged": base + ("weights_merged_lora" if lora else "weights"),
        "results": base + "output_results",
    }
    for p in paths.values():
        if p is not None:
            os.makedirs(p, exist_ok=True)
    return paths


def run(config: FinetuneConfig):
    """Train and / or evaluate; returns ``(accuracy, correct, paths)``."""
    set_seed(config.seed)
    prompts = PromptConfig(config.model_name, config.template, config.title_case)
    tokenizer = load_tokenizer(config)
    model = load_model(config)

    records = filter_records(load_records(config), tokenizer, config)
    train_part, valid_part, test_part = split_records(records, config)
    sizes = [len(p) for p in (train_part, valid_part, test_part) if p is not None]
    print("train / valid / test:" if valid_part is not None else "train / test:", sizes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = output_paths(config, timestamp)

    if config.train:
        if config.loss == "sequence":
            train_data = tokenize_sequences(train_part, tokenizer, prompts)
            eval_data = tokenize_sequences(valid_part if valid_part is not None else test_part,
                                           tokenizer, prompts)
        else:
            train_data, valid_data, test_data = target_datasets(
                [train_part, valid_part, test_part], tokenizer, prompts, config.max_len_ratio
            )
            eval_data = valid_data if valid_data is not None else test_data
        model, _ = train(model, tokenizer, train_data, eval_data, config, paths["checkpoints"])
        model = save(model, paths)

    accuracy, correct = None, []
    if config.test:
        accuracy, correct = evaluate(model, tokenizer, test_prompts(test_part, prompts))
        write_report(config, paths, timestamp, accuracy, correct, prompts.template)
        print(f"accuracy={accuracy:.4f}  correct={len(correct)}")
    return accuracy, correct, paths
