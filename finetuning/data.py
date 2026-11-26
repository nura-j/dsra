"""Data for fine-tuning: loading, filtering, splitting and tokenisation.

Two training-data forms:

- **sequence loss**: each example is the tokenised text
  ``"<Definition><prompt><definiendum>"``; the collator uses every token.
- **target loss**: the prompt is padded to a shared length and the
  definiendum is appended after the prompt's last real token; prompt and
  padding positions get label -100, so only the definiendum is learned.
"""

import json
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from dataset_construction.prompts import PromptConfig

from .config import FinetuneConfig


def load_records(config: FinetuneConfig):
    """``[{"definition", "definiendum"}]`` from the JSON file, or raw glosses
    from ``saf_datasets`` when ``config.data_path`` is None."""
    if config.data_path is None:
        if config.language == "en":
            from saf_datasets import WordNetFilteredDataSet as Source
        else:
            from saf_datasets.data_access.wordnet_spanish import (
                SpanishWordNetFilteredDataSet as Source,
            )
        return [
            {"definition": x.surface, "definiendum": x.annotations["definiendum"]}
            for x in Source()
        ]
    definition_key, word_key = config.fields
    with open(config.data_path) as f:
        data = json.load(f)
    return [{"definition": x[definition_key], "definiendum": x[word_key]} for x in data]


def filter_records(records, tokenizer, config: FinetuneConfig):
    """Length, single-token, alphabetic and self-reference filters."""
    vocab = set(tokenizer.get_vocab()) if config.single_token else None
    kept = []
    for x in records:
        definition, word = x["definition"], x["definiendum"]
        n_words = len(definition.split())
        if config.min_words is not None and n_words < config.min_words:
            continue
        if config.max_words is not None and n_words > config.max_words:
            continue
        if config.single_token and word not in vocab:
            continue
        if config.alphabetic and not word.isalpha():
            continue
        if config.exclude_in_definition and word in definition:
            continue
        kept.append(x)
    return kept


def split_records(records, config: FinetuneConfig):
    """``(train, valid, test)``; ``valid`` is None without ``valid_share``.

    Uses the same ``train_test_split`` calls as the original scripts, so the
    assignment of items to splits is unchanged.
    """
    train, held_out = train_test_split(records, test_size=config.test_size,
                                       random_state=config.seed)
    if config.valid_share is None:
        return train, None, held_out
    valid, test = train_test_split(held_out, test_size=1 - config.valid_share,
                                   random_state=config.seed)
    return train, valid, test


# --- sequence loss ---------------------------------------------------------

def tokenize_sequences(records, tokenizer, prompts: PromptConfig):
    """Tokenised ``"<Definition><prompt><definiendum>"`` texts."""
    return [tokenizer(prompts.example(x["definition"], x["definiendum"])) for x in records]


# --- target loss -----------------------------------------------------------

def target_max_length(records, tokenizer, prompts: PromptConfig, ratio=0.95):
    """Shortest prompt length (tokens) that covers ``ratio`` of the prompts."""
    lengths = Counter()
    max_len = 0
    for x in records:
        n = tokenizer(prompts.prompt(x["definition"]), return_tensors="pt").input_ids.shape[1]
        lengths[n] += 1
        max_len = max(max_len, n)
    if ratio >= 1:
        return max_len
    total, current = sum(lengths.values()), 0
    for length, count in sorted(lengths.items()):
        current += count
        if current / total >= ratio:
            return length
    return max_len


def tokenize_targets(records, tokenizer, prompts: PromptConfig, max_len):
    """``(pairs, end_before_padding)``: each pair is the prompt padded to
    ``max_len`` and the definiendum ids; ``end_before_padding`` is the number
    of real prompt tokens (at most ``max_len - 1``).

    Prompts are padded on the right, which the prompt-end index assumes.
    """
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    pairs, ends = [], []
    try:
        for x in records:
            prompt_ids = tokenizer(
                prompts.prompt(x["definition"]),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            ).input_ids
            target_ids = tokenizer(
                x["definiendum"], return_tensors="pt", padding=False, add_special_tokens=False
            ).input_ids
            pairs.append([prompt_ids, target_ids])
            n_real = int((prompt_ids.squeeze() != tokenizer.pad_token_id).long().sum())
            ends.append(max_len - 1 if n_real >= max_len else n_real)
    finally:
        tokenizer.padding_side = side
    return pairs, ends


class TargetOnlyDataset(TorchDataset):
    """``{input_ids, attention_mask, labels}`` with the loss on the definiendum only.

    ``input_ids`` is the prompt up to its last real token, then the
    definiendum, truncated and right-padded to ``max_len``. Labels are -100
    on the prompt and the padding.
    """

    def __init__(self, tokenized_data, end_before_padding, max_len, tokenizer):
        self.tokenized_data = tokenized_data
        self.end_before_padding = end_before_padding
        self.max_len = max_len
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        prompt_end_idx = self.end_before_padding[idx]
        prompt = tokens[0][0][:prompt_end_idx]
        target = tokens[1][0]
        input_ids = torch.cat((prompt, target))
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat(
                (input_ids, torch.full((padding_length,), self.pad_token_id, dtype=torch.long))
            )

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        if padding_length > 0:
            attention_mask[-padding_length:] = 0

        labels = input_ids.clone()
        labels[:prompt_end_idx] = -100
        if padding_length > 0:
            labels[-padding_length:] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def target_datasets(splits, tokenizer, prompts: PromptConfig, ratio=0.95):
    """``TargetOnlyDataset`` per split, sharing one ``max_len`` over all items."""
    every = [x for part in splits if part is not None for x in part]
    max_len = target_max_length(every, tokenizer, prompts, ratio)
    out = []
    for part in splits:
        if part is None:
            out.append(None)
            continue
        pairs, ends = tokenize_targets(part, tokenizer, prompts, max_len)
        out.append(TargetOnlyDataset(pairs, ends, max_len, tokenizer))
    return out


def test_prompts(records, prompts: PromptConfig):
    """``[{"text": prompt, "label": definiendum}]`` for the next-token evaluation."""
    return [{"text": prompts.prompt(x["definition"]), "label": x["definiendum"]} for x in records]
