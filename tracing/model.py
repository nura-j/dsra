"""Model loading, module names and token helpers for causal tracing.

Supports three module trees: GPT-2 / GPT-J (``transformer.h``), GPT-NeoX /
Pythia (``gpt_neox.layers``) and the Llama family, which also covers Qwen2,
Gemma and Mistral (``model.layers``).
"""

import re

import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy.rome import nethook

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


def resolve_dtype(name, model_name=""):
    """``"auto"`` keeps the model's default precision (float16 for 20B models)."""
    if name == "auto":
        return torch.float16 if "20b" in model_name else None
    return DTYPES[name]


class ModelAndTokenizer:
    """Holds a causal LM and its tokenizer, and counts the transformer layers."""

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device="cuda",
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, **{DTYPE_KWARG: torch_dtype}
            )
            nethook.set_requires_grad(False, model)
            model.eval().to(device)
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    """Module name of layer ``num``; ``kind`` is None, "mlp", "attn" or "embed"."""
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama-family module tree (Llama, Qwen2, Gemma, Mistral, ...).
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    raise ValueError("unknown transformer structure")


def guess_subject(prompt):
    """First capitalised span of a prompt, used when no subject is given."""
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def make_inputs(tokenizer, prompts, device="cuda"):
    """Left-pad a list of prompts into ``input_ids`` / ``attention_mask``."""
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """Decode each token id on its own (nested for 2-D arrays)."""
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """Token span ``(start, end)`` covering ``substring`` in the prompt."""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    if substring in whole_string:
        char_loc = whole_string.index(substring)
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(toks):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(substring):
                tok_end = i + 1
                break
        return (tok_start, tok_end)
    # Fallback for tokenizers whose per-token decode drops word-boundary spaces
    # (SentencePiece, e.g. Gemma): locate the substring on incremental decodes,
    # which restore the spacing that single-token decoding loses.
    ids = [int(t) for t in token_array]
    prefixes = [tokenizer.decode(ids[: i + 1]) for i in range(len(ids))]
    whole = prefixes[-1]
    char_loc = whole.index(substring)
    tok_start, tok_end = None, None
    for i, p in enumerate(prefixes):
        if tok_start is None and len(p) > char_loc:
            tok_start = i
        if tok_end is None and len(p) >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_from_input(model, inp):
    """Argmax next token and its probability for every row of a batch.

    This is the correctness criterion of the protocol: a case counts as
    correct only if the argmax next token decodes to the definiendum.
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def predict_token(mt, prompts, return_p=False):
    """Decoded argmax next token for each prompt (optionally with its probability)."""
    inp = make_inputs(mt.tokenizer, prompts, device=mt.device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result
