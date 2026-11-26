"""Train / test / validation splits and their files."""

import json
import random
from pathlib import Path

from .config import SPLIT_NAMES, CleaningConfig


def split_records(records, shares=(0.8, 0.1, 0.1), shuffle=False, seed=42):
    """``{"train", "test", "validation"}`` by consecutive slices.

    Cut points are ``int(n * share)`` and ``int(n * (train + test))``, as in
    the early selection script; ``shuffle`` shuffles a copy first.
    """
    records = list(records)
    if shuffle:
        random.Random(seed).shuffle(records)
    n = len(records)
    first = int(n * shares[0])
    second = int(n * round(shares[0] + shares[1], 10))
    parts = (records[:first], records[first:second], records[second:])
    return dict(zip(SPLIT_NAMES, parts))


def write_splits(parts, config: CleaningConfig):
    """Write each split as JSON (and text, with ``text_template``); returns the paths."""
    paths = []
    for name, part in parts.items():
        path = Path(config.split_path(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(part, f, indent=4)
        paths.append(str(path))
        if config.text_template:
            text_path = Path(config.split_path(name, ".txt"))
            with open(text_path, "w") as f:
                for rec in part:
                    f.write(config.text_template.format(
                        definition=rec[config.definition_field], word=rec[config.word_field]) + "\n")
            paths.append(str(text_path))
    return paths
