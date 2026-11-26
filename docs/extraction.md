# Extraction

This module reads WordNet definitions from `saf_datasets` and writes them as a list of JSON records, the starting point of every DSRA dataset. Multi-part WordNet glosses are merged into one sentence, and each definition can also get a masked variant.

The module is imported as `dataset_construction.extraction`.

## Sources

| Language | Dataset class | Output |
| --- | --- | --- |
| English (EWN) | `saf_datasets.WordNetFilteredDataSet` | `data/wordnet_data_definitions.json` (84,423 records) |
| Spanish (SWN) | `saf_datasets.data_access.wordnet_spanish.SpanishWordNetFilteredDataSet` | `data/wordnet_es_data_definitions.json` |

Both sets are filtered by `saf_datasets` to definienda that are a single LLaMA token. They are downloaded on first use and cached under `~/.saf_data/WordNet/`.

Each record holds:

```json
{
    "word": "able",
    "definition": "having the necessary means or skill or know-how or authority to do something",
    "wordnet_id": "able.a.01",
    "masked_definition": "having the necessary means or <extra_id_0> or know-how or authority to <extra_id_0> something"
}
```

## Merging glosses

A WordNet gloss can hold several `;`-separated parts: the core definition, usage domains such as `in music` or `slang`, and cross-references. `combine_wordnet_definitions` keeps the core first, places each domain label before it (prefix) or after it (postfix) as listed in `DOMAIN_HANDLING`, and drops cross-references:

```python
from dataset_construction.extraction import combine_wordnet_definitions

combine_wordnet_definitions("a small stringed instrument; music")
# 'in music, a small stringed instrument'
```

The merge rules insert English words (`in ..., `, ` and `), so the Spanish preset keeps glosses as they are. The merge also leaves a few artefacts in the English output, such as `in in ...`; the cleaning step can repair them (see [cleaning](cleaning.md)).

## Masked definitions

`corrupt_definition` replaces `n_mask` randomly chosen words with the mask string; definitions of one word cannot be masked and are dropped when masking is on. The masked positions use `random.Random(seed)`; `seed=None` falls back to the global `random` state, which is how the original script ran (its masks are not reproducible). The definitions themselves do not depend on the seed.

## Configuration

```python
from dataset_construction.extraction import ExtractionConfig

config = ExtractionConfig(
    language="en",               # "en" (EWN) or "es" (SWN)
    out=None,                    # default: data/<file name of the language>
    combine_glosses=True,        # merge multi-part glosses
    add_masked=True,             # add masked_definition, drop one-word definitions
    n_mask=2,                    # words masked per definition
    mask=" <extra_id_0> ",
    seed=42,                     # None: global random state, as in the original script
)
```

Presets:

```python
ExtractionConfig.english()   # EWN, merged glosses, masked definitions
ExtractionConfig.spanish()   # SWN, glosses kept as they are
```

## Extracting definitions

```python
from dataset_construction.extraction import ExtractionConfig, extract_definitions, load_definitions

records = extract_definitions(ExtractionConfig.english())
records = load_definitions("data/wordnet_data_definitions.json")
```

`extract_definitions(config, dataset=None, write=True)` also accepts any iterable of `saf_datasets` sentences as `dataset`, for example a subset.

From the command line, from the repository root:

```bash
python -m dataset_construction.extraction.extract_definitions --language en
python -m dataset_construction.extraction.extract_definitions --language es
```

| Argument | Purpose |
| --- | --- |
| `--language` | `en` (English WordNet) or `es` (Spanish WordNet); default `en`. |
| `--out` | Output JSON (default: `data/wordnet_data_definitions.json` or `data/wordnet_es_data_definitions.json`). |
| `--no-masked` | Do not add masked definitions. |
| `--n-mask` | Words masked per definition (default: `2`). |
| `--seed` | Seed for the masked positions (default: `42`). |
