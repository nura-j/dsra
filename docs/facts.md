# Fact files

A fact file lists the definitions to trace for one model. This module builds it from the cleaned candidate pool: it keeps definienda that are a single continuation token after the model's prompt and, with screening on, only definitions the model already completes correctly zero-shot, so every traced case is productive.

The module is imported as `dataset_construction.facts`.

## Pipeline

1. **Pool:** `cleaning.build_pool` on the extraction output (lower-case alphabetic definiendum, first definition per word, at most `max_words` words, definiendum not in its definition).
2. **Prompt:** the model's registered prompt from [prompts](prompts.md) (or `template`).
3. **Single token:** the definiendum must add exactly one token after the prompt, and the prompt's tokenisation must be a prefix of the prompt-plus-word tokenisation.
4. **Order:** candidates are shuffled with `seed` (default 42).
5. **Screening:** candidates are kept, in order, while the model's argmax next token is the definiendum, until `max_facts` are kept or `max_screen` candidates were tried.

## Records

The records use the schema of `data/known_data_wordNet.json`, so the tracer reads them unchanged:

```json
{
  "known_id": 0,
  "subject": "a building where books are kept",
  "attribute": "library",
  "template": "{} is called a \"",
  "prompt": "a building where books are kept is called a \"",
  "relation_id": "P01",
  "wordnet_id": "library.n.01"
}
```

`subject` is the definition as it appears in the prompt, title-cased for models that title-case (GPT-J, BERTIN), so the tracer can locate the corrupted span.

## Configuration

```python
from dataset_construction.facts import FactConfig

config = FactConfig(
    model_name="gpt2-large",         # Hugging Face id or CAP/CARMA alias
    template=None,                   # default: the registered prompt
    title_case=None,                 # default: the model's setting
    definitions="data/wordnet_data_definitions.json",
    out=None,                        # default: data/facts_<model>_n<max_facts>.json
    max_facts=800,
    max_words=25,
    max_screen=40000,
    screen=True,
    torch_dtype="float32",           # "float32", "float16" or "bfloat16"
    device="cuda",                   # default: cuda if available
    seed=42,
)
```

Fact files are named by size (`facts_<model>_n<N>.json`), so a pilot file is never reused for a full run by accident.

## Building a fact file

```python
from dataset_construction.facts import FactConfig, build_facts

records = build_facts(FactConfig("gpt2-large", max_facts=25))
```

`build_facts(config, model=None, tokenizer=None, write=True)` accepts an already loaded model and tokenizer. The log reports the pool size, the single-token candidates and the screening yield.

From the command line, from the repository root:

```bash
python -m dataset_construction.facts.build_facts --model_name gpt2-large
python -m dataset_construction.facts.build_facts --model_name GPT2_large --max_facts 25
```

| Argument | Purpose |
| --- | --- |
| `--model_name` | Hugging Face id or CAP/CARMA alias (required). |
| `--template` | Prompt with `{}` for the definition (default: the registered prompt). |
| `--title_case` | `yes` or `no` (default: the model's setting). |
| `--defs` | Definitions JSON (default: `data/wordnet_data_definitions.json`). |
| `--out` | Output path (default: `data/facts_<model>_n<max_facts>.json`). |
| `--max_facts` | Facts to keep (default: `800`). |
| `--max_words` | Longest definition kept, in words (default: `25`). |
| `--max_screen` | Upper bound on candidates screened (default: `40000`). |
| `--screen` / `--no_screen` | Keep only zero-shot correct candidates (default) or every single-token candidate. |
| `--torch_dtype` | Precision of the screening model (default: `float32`). |
| `--device` | Torch device (default: `cuda` if available). |
| `--seed` | Shuffle seed (default: `42`). |

## Screening yield

The share of screened candidates a model answers zero-shot varies widely: roughly 10-35 % for the 1B-3B models, and very low for Pythia-14M, where the builder may exhaust the pool before reaching `max_facts`. It writes however many facts it found; report the obtained N.
