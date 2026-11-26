# Prompts

This module holds the inverse-dictionary prompt of every supported model. The fact builder, fine-tuning and tracing all read their prompt from here, so a model always gets the same wording.

The module is imported as `dataset_construction.prompts`.

## Supported models

Models are registered by their Hugging Face id. Each entry also accepts the short names used by the CAP and CARMA code, so `GPT2_large`, `gemma_2b` or `Qwen2_5_3B` resolve to the same model.

| Family | Models | Template |
| --- | --- | --- |
| GPT-J (paper, English) | `EleutherAI/gpt-j-6B` | `{} is often referred to as:` (title-cased) |
| GPT-J (paper, Spanish) | `bertin-project/bertin-gpt-j-6B` | `{} se suele denominar como:` (title-cased) |
| GPT-2 | `gpt2`, `gpt2-medium`, `gpt2-large` | `{} is called a "` |
| Pythia | `EleutherAI/pythia-14m`, `-410m`, `-1b` | `{} is called a "` |
| Gemma | `google/gemma-2b` | `{} is called a "` |
| Llama | `meta-llama/Llama-2-7b-hf`, `Meta-Llama-3-8B`, `Meta-Llama-3-70B`, `Llama-3.1-8B`, `Llama-3.2-1B`, `Llama-3.2-3B` | `{} is called a "` |
| Llama-2 chat, Mistral | `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-v0.1`, `-v0.3` | `{} is called a ` (no quote) |
| Qwen2 | `Qwen/Qwen2-0.5B`, `-1.5B`, `-7B` | `{} is called a "` |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B`, `-1.5B`, `-3B` | `{} is commonly known as "` |

`{}` is replaced by the definition; the model is expected to continue with the definiendum. The non-paper prompts follow the CARMA prompt table. T5 and instruction-tuned models are not registered: the tracer does not support encoder-decoder models, and the instruction prompts had very low accuracy in CAP.

## Using a prompt

```python
from dataset_construction.prompts import PromptConfig

prompts = PromptConfig("GPT2_large")
prompts.model_name                                  # "gpt2-large"
prompts.prompt("a building where books are kept")
# 'a building where books are kept is called a "'
prompts.example("a building where books are kept", "library")
# 'a building where books are kept is called a "library'
```

`PromptConfig(model_name, template=None, title_case=None)` takes the registered template and title-casing of the model; either can be overridden, for example to trace a local checkpoint or another wording:

```python
PromptConfig("path/to/checkpoint", template='{} is called a "')
```

Lower-level helpers:

```python
from dataset_construction.prompts import format_prompt, get_template, model_info, resolve_model

resolve_model("Qwen2_5_3B")                 # "Qwen/Qwen2.5-3B"
model_info("gpt2-large").family             # "gpt2"
get_template("EleutherAI/gpt-j-6B")         # "{} is often referred to as:"
format_prompt("a domestic animal", '{} is called a "', title_case=True)
```

## Adding a model

Add an entry to `MODELS` in `models.py` (family, language, aliases) and its template to `TEMPLATES` in `templates.py`. The template must contain `{}`. Models whose definitions are title-cased are listed in `TITLE_CASE`.

## GPT-J prompt history

The GPT-J prompt appears in several forms in the archived code: `"{} is often referred to as:"` (the paper), `" is often referred to as: "` with a trailing space (the old tracing driver), `" is often reffered to as:"` (the old English fine-tuning default) and `"{} is commonly known as:"` (an early traced subset, `data/known_data_wordNet.json`). The registry uses the paper form.
