# Cleaning

This module filters the extracted definitions into the candidate pool used for tracing, optionally repairs gloss-merging artefacts, annotates definienda and definition tokens with WordNet, spaCy and NER attributes, and writes train / test / validation splits.

The module is imported as `dataset_construction.cleaning`.

## Default filters

The defaults are the DSRA paper settings:

1. the definiendum is made of lower-case letters only;
2. one definition per word, the first that passes the other filters;
3. definitions of at most 25 words;
4. the definiendum is not one of the words of its own definition.

On `data/wordnet_data_definitions.json` these keep 53,798 of 84,423 records. The same filters build the candidate pool of the [fact builder](facts.md):

```python
from dataset_construction.cleaning import build_pool
from dataset_construction.extraction import load_definitions

pool = build_pool(load_definitions("data/wordnet_data_definitions.json"), max_words=25)
# {"word": {"definition": "...", "wordnet_id": "..."}, ...}
```

## Configuration

```python
from dataset_construction.cleaning import CleaningConfig

config = CleaningConfig(
    input="data/wordnet_data_definitions.json",
    out="data/wordnet_definitions_clean.json",
    word_field="word",
    definition_field="definition",
    min_words=None,
    max_words=25,
    alphabetic=True,                 # lower-case letters only
    first_definition_only=True,
    self_reference="token",          # "token", "substring" or None
    repair=False,                    # see "Repairs"
    annotate=False,                  # see "Annotation"
    split=None,                      # e.g. (0.8, 0.1, 0.1) for train / test / validation
    shuffle=False,
    text_template=None,              # also write each split as text lines
    seed=42,
)
```

`CleaningConfig.legacy()` reproduces the early selection script: it reads the annotated WordNet file `data/WordNet_filtered_data.json` (field `name`), keeps definitions of 4-14 words whose definiendum does not occur anywhere in them, and splits 80/10/10 in file order. It writes `WordNet_definitions_filtered_data.json` and, for each split, `WordNet_definitions_filtered_<split>_data.json` and `.txt` with lines `"<definition> is commonly known as: <word>"`.

## Repairs

`combine_wordnet_definitions` writes domain labels as `"in <label>, "`. Three patterns in the English extraction output come from that step; each repair can be switched on separately (`repair_rules`):

| Rule | Definitions | Example | Repaired |
| --- | --- | --- | --- |
| `double_in` | 6 | `in in India and Malaysia, a warehouse` | `in India and Malaysia, a warehouse` |
| `repeated_label` | 29 | `in British informal, pretty and neat and British informal` | `in British informal, pretty and neat` |
| `function_word_prefix` | 82 | `in of business, not active or brisk` | `of business, not active or brisk` |

Repairs are off by default, because the paper data were built without them.

```python
from dataset_construction.cleaning import repair_definition

repair_definition("in in India and Malaysia, a warehouse")
```

## Annotation

`WordNetAnnotator` adds, for the definiendum, its WordNet id, name and part of speech, Brown-corpus and WordNet frequency, lexname category, and abstraction and generality scores; and for each token of the definition, its spaCy part of speech (mapped to WordNet tags), dependency label, NER tag and, when WordNet knows the token, the same WordNet attributes.

It needs spaCy with `en_core_web_sm` (installed by `requirements.txt`), the NLTK WordNet and Brown corpora, and a token-classification model for NER:

```bash
python -m nltk.downloader wordnet brown
```

The NER model is pinned to `dbmdz/bert-large-cased-finetuned-conll03-english` at revision `4c53496`, the default of `pipeline("ner")`, so a newer transformers release cannot switch it. Models are loaded when the annotator is created, not on import.

```python
from dataset_construction.cleaning import WordNetAnnotator

annotator = WordNetAnnotator()
records = annotator.annotate_records(records)        # adds "annotation", looked up by wordnet_id
data = annotator.annotate_wordnet(out="data/WordNet_filtered_data.json", limit=100)
```

`annotate_wordnet` covers every synset with an alphabetic name (117,659 synsets without `limit`), which runs spaCy and NER on each definition and takes a long time.

## Running the pipeline

```python
from dataset_construction.cleaning import CleaningConfig, clean

kept = clean(CleaningConfig(repair=True))
```

`clean(config, records=None, annotator=None, write=True)` loads `config.input` (or uses `records`), repairs, filters, annotates, writes `config.out` and, with `config.split`, the split files.

From the command line, from the repository root:

```bash
python -m dataset_construction.cleaning.clean_definitions
python -m dataset_construction.cleaning.clean_definitions --repair --annotate
python -m dataset_construction.cleaning.clean_definitions --legacy
python -m dataset_construction.cleaning.annotate_wordnet --limit 100
```

`clean_definitions`:

| Argument | Purpose |
| --- | --- |
| `--legacy` | Reproduce the early selection script. |
| `--input`, `--out` | Input and output JSON. |
| `--min_words`, `--max_words` | Definition-length bounds. |
| `--repair` | Fix gloss-merging artefacts before filtering. |
| `--annotate` | Add WordNet, spaCy and NER annotations to the kept records. |
| `--split TRAIN TEST VALID` | Also write split files with these shares. |
| `--shuffle`, `--seed` | Shuffle before splitting (seed default: `42`). |

`annotate_wordnet`:

| Argument | Purpose |
| --- | --- |
| `--out` | Output JSON (default: `data/WordNet_filtered_data.json`). |
| `--limit` | Annotate only the first N synsets. |
| `--spacy_model`, `--ner_model`, `--ner_revision` | Annotation models. |
| `--seed` | Random seed (default: `42`). |

## Notes

- `clean_definition`, used for the annotated WordNet glosses, removes every character that is not a letter or a space, hyphens included.
- A word whose first definition fails a filter keeps a later definition that passes it (`first_definition_only` counts kept definitions).
