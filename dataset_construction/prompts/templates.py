"""Inverse-dictionary prompt templates, one per model.

``{}`` is replaced by the definition; the model is expected to continue with
the definiendum. The GPT-J and BERTIN entries are the DSRA paper prompts; the
others follow the CARMA prompt table (CAP's table has the same wording, but
six of its entries lack ``{}`` and would drop the definition).

The GPT-J prompt appears in several forms in the archived code:
``"{} is often referred to as:"`` (paper), ``" is often referred to as: "``
with a trailing space (``main.py``), ``" is often reffered to as:"`` (English
fine-tuning default) and ``"{} is commonly known as:"`` (early traced subset).
"""

from .models import resolve_model

GPTJ_TEMPLATE = "{} is often referred to as:"
BERTIN_TEMPLATE = "{} se suele denominar como:"
CALLED_A = '{} is called a "'
CALLED_A_NO_QUOTE = "{} is called a "
COMMONLY_KNOWN_AS = '{} is commonly known as "'

TEMPLATES = {
    "EleutherAI/gpt-j-6B": GPTJ_TEMPLATE,
    "bertin-project/bertin-gpt-j-6B": BERTIN_TEMPLATE,
    "gpt2": CALLED_A,
    "gpt2-medium": CALLED_A,
    "gpt2-large": CALLED_A,
    "EleutherAI/pythia-14m": CALLED_A,
    "EleutherAI/pythia-410m": CALLED_A,
    "EleutherAI/pythia-1b": CALLED_A,
    "google/gemma-2b": CALLED_A,
    "meta-llama/Llama-2-7b-hf": CALLED_A,
    "meta-llama/Llama-2-7b-chat-hf": CALLED_A_NO_QUOTE,
    "meta-llama/Meta-Llama-3-8B": CALLED_A,
    "meta-llama/Meta-Llama-3-70B": CALLED_A,
    "meta-llama/Llama-3.1-8B": CALLED_A,
    "meta-llama/Llama-3.2-1B": CALLED_A,
    "meta-llama/Llama-3.2-3B": CALLED_A,
    "mistralai/Mistral-7B-v0.1": CALLED_A_NO_QUOTE,
    "mistralai/Mistral-7B-v0.3": CALLED_A_NO_QUOTE,
    "Qwen/Qwen2-0.5B": CALLED_A,
    "Qwen/Qwen2-1.5B": CALLED_A,
    "Qwen/Qwen2-7B": CALLED_A,
    "Qwen/Qwen2.5-0.5B": COMMONLY_KNOWN_AS,
    "Qwen/Qwen2.5-1.5B": COMMONLY_KNOWN_AS,
    "Qwen/Qwen2.5-3B": COMMONLY_KNOWN_AS,
}

# Models whose definitions are title-cased in the prompt (the GPT-J protocol).
TITLE_CASE = {"EleutherAI/gpt-j-6B", "bertin-project/bertin-gpt-j-6B"}


def get_template(model_name):
    """Registered template of a model (by id or CAP/CARMA alias)."""
    name = resolve_model(model_name)
    if name not in TEMPLATES:
        raise KeyError(f"no template registered for {model_name}; pass one explicitly")
    return TEMPLATES[name]


def uses_title_case(model_name):
    return resolve_model(model_name) in TITLE_CASE
