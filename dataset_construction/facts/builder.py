"""Build a per-model fact file from the cleaned definition pool.

For every candidate (word, definition) pair the builder keeps only definienda
that form a single continuation token after the model's prompt and, with
screening on, only candidates whose argmax next token is that token (the
single-token correctness criterion of the tracing protocol).

The records use the schema of ``data/known_data_wordNet.json`` (``known_id``,
``subject``, ``attribute``, ``template``, ``prompt``, ``relation_id``), so the
tracer reads them unchanged.
"""

import json
import random
from pathlib import Path

import torch
import transformers
from packaging import version
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_construction.cleaning import build_pool
from dataset_construction.extraction import load_definitions
from dataset_construction.prompts import PromptConfig

from .config import FactConfig

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
# transformers 4.56 renamed from_pretrained's ``torch_dtype`` to ``dtype``.
DTYPE_KWARG = (
    "dtype"
    if version.parse(transformers.__version__) >= version.parse("4.56")
    else "torch_dtype"
)


def continuation_ids(tokenizer, prompt, word):
    """Token ids that ``word`` adds when appended to ``prompt``, or None if the
    prompt's tokenisation is not a prefix of the combined tokenisation."""
    base = tokenizer(prompt, add_special_tokens=True).input_ids
    full = tokenizer(prompt + word, add_special_tokens=True).input_ids
    if len(full) <= len(base) or full[: len(base)] != base:
        return None
    return full[len(base):]


def single_token_candidates(pool, tokenizer, prompts: PromptConfig):
    """Pool entries whose definiendum is one continuation token after the prompt.

    ``subject`` is the definition as written in the prompt (title-cased for
    models that title-case), so the tracer can find it there.
    """
    candidates = []
    for word, rec in pool.items():
        subject = rec["definition"].title() if prompts.title_case else rec["definition"]
        prompt = prompts.prompt(rec["definition"])
        if subject not in prompt:
            continue
        cont = continuation_ids(tokenizer, prompt, word)
        if cont is None or len(cont) != 1:
            continue
        candidates.append(dict(word=word, prompt=prompt, answer_id=cont[0],
                               subject=subject, **rec))
    return candidates


def load_screening_model(config: FactConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, **{DTYPE_KWARG: DTYPES[config.torch_dtype]}
    )
    model.eval().to(config.device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def screen_candidates(candidates, model, tokenizer, config: FactConfig):
    """Keep candidates in order until ``max_facts`` are found.

    Returns ``(kept, n_screened)``; without screening every candidate is kept.
    """
    kept, screened = [], 0
    for cand in tqdm(candidates[: config.max_screen], desc="screening"):
        if len(kept) >= config.max_facts:
            break
        if config.screen:
            screened += 1
            ids = torch.tensor(
                [tokenizer(cand["prompt"], add_special_tokens=True).input_ids]
            ).to(config.device)
            with torch.no_grad():
                logits = model(input_ids=ids).logits
            if int(logits[0, -1].argmax()) != cand["answer_id"]:
                continue
        kept.append(cand)
    return kept, screened


def to_records(kept, template):
    """Fact records in the tracer's schema."""
    return [
        dict(
            known_id=i,
            subject=c["subject"],
            attribute=c["word"],
            template=template,
            prompt=c["prompt"],
            relation_id="P01",
            wordnet_id=c["wordnet_id"],
        )
        for i, c in enumerate(kept)
    ]


def build_facts(config: FactConfig, model=None, tokenizer=None, write=True):
    """Run the whole pipeline for one model and (optionally) write the file."""
    pool = build_pool(load_definitions(config.definitions), config.max_words)
    print(f"Candidate pool after definition filters: {len(pool)} words")

    prompts = PromptConfig(config.model_name, config.template, config.title_case)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    candidates = single_token_candidates(pool, tokenizer, prompts)
    print(f"Single-continuation-token candidates: {len(candidates)}")

    random.Random(config.seed).shuffle(candidates)

    if config.screen and model is None:
        model = load_screening_model(config)
    kept, screened = screen_candidates(candidates, model, tokenizer, config)
    if config.screen and screened:
        print(
            f"Screened {screened} candidates; kept {len(kept)} "
            f"({len(kept) / screened:.1%} zero-shot correct)"
        )

    records = to_records(kept, config.template)
    if write:
        out = Path(config.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(records, f, indent=1)
        print(f"Wrote {len(records)} facts to {out}")
    return records
