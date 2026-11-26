# Fine-tuning

This module fine-tunes a model on the inverse dictionary task and measures its single next-token accuracy on the held-out definitions. Only correctly predicted test definitions are traced.

The module is imported as `finetuning`.

## Settings

| | DSRA paper (`gptj_english`, `bertin_spanish`) | CAP (`cap`) |
| --- | --- | --- |
| Weights | LoRA adapters | all weights |
| Loss | whole text `"<Definition><prompt><definiendum>"` | definiendum tokens only (prompt and padding masked) |
| Split | train / test | train / validation / test (20 / 8 / 72) |
| Settings | Table 4 of the paper | 3 epochs, learning rate 5e-5, weight decay 0.01, evaluation per epoch |

Paper settings (Table 4); the learning rate, optimiser and target modules are not reported there, so the presets keep the values of the original scripts:

| | GPT-J-6B (English) | BERTIN GPT-J-6B (Spanish) |
| --- | --- | --- |
| Epochs | 10 | 20 |
| Batch size | 16 | 16 |
| LoRA rank | 64 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.3 |
| Data | English WordNet, raw `saf_datasets` glosses | hand-made Spanish JSON (`spanish_definition`, `spanish_name`) |
| Filters | 4-25 words, single-token, alphabetic, definiendum not in definition | 3-24 words |
| Split | 80 / 20 | 30 / 70 |

The prompt comes from [prompts](prompts.md): the GPT-J and BERTIN prompts title-case the definition.

## Configuration

```python
from finetuning import FinetuneConfig

config = FinetuneConfig.gptj_english()
config = FinetuneConfig.bertin_spanish(
    "data/WordNet_filtered_data_es_with_replacement_with_back_transl_cleaned.json")
config = FinetuneConfig.cap("gpt2-large")          # uses data/wordnet_data_definitions.json
```

Every field can be overridden in the preset call or set directly:

```python
config = FinetuneConfig(
    model_name="EleutherAI/gpt-j-6B",
    method="lora",                   # "lora" or "full"
    loss="sequence",                 # "sequence" or "target"
    data_path=None,                  # None: raw glosses from saf_datasets (language)
    language="en",
    fields=("definition", "word"),   # JSON keys of definition and definiendum
    template=None, title_case=None,  # default: the model's registered prompt
    min_words=4, max_words=25,
    single_token=True, alphabetic=True, exclude_in_definition=True,
    test_size=0.2, valid_share=None,
    r=64, lora_alpha=32, lora_dropout=0.1,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    adapter_path=None,               # existing adapter, merged before training or testing
    epochs=10, batch_size=16,
    learning_rate=None, weight_decay=None,
    train=True, test=True,
    output_dir="output",
    device="cuda",
    seed=42,
)
```

`target_modules=None` lets PEFT choose the default modules of the architecture, which is what every preset except `gptj_english` uses.

## Running

```python
from finetuning import FinetuneConfig, run

accuracy, correct, paths = run(FinetuneConfig.gptj_english())
```

From the command line, from the repository root:

```bash
python -m finetuning.finetune --preset gptj_english
python -m finetuning.finetune --preset bertin_spanish \
    --data_path data/WordNet_filtered_data_es_with_replacement_with_back_transl_cleaned.json
python -m finetuning.finetune --preset cap --model_name gpt2-large
python -m finetuning.finetune --preset cap --model_name path/to/model --template '{} is called a "'
```

| Argument | Purpose |
| --- | --- |
| `--preset` | `gptj_english`, `bertin_spanish` or `cap` (required). |
| `--model_name` | Model id or CAP/CARMA alias (required for `cap`). |
| `--data_path` | Definitions JSON (required for `bertin_spanish`). |
| `--template`, `--title_case` | Prompt override, for checkpoints outside the registry. |
| `--adapter_path` | Existing LoRA adapter to merge before training or testing. |
| `--epochs`, `--batch_size` | Override the preset. |
| `--no_train` / `--no_test` | Only evaluate / only train. |
| `--output_dir`, `--device`, `--seed` | Output root, torch device (`auto` spreads over GPUs), seed (default `42`). |

Gated models need an accepted licence on their Hugging Face page and `HF_TOKEN` in the environment.

## Evaluation and outputs

A test definition counts as correct when the single generated token, stripped of spaces, equals the definiendum.

Outputs go to `<output_dir>/<model>/<method>/<timestamp>_*`:

| Folder | Content |
| --- | --- |
| `checkpoints` | Trainer checkpoints. |
| `weights_unmerged_lora`, `weights_merged_lora` | LoRA adapter and merged model (LoRA runs). |
| `weights` | Fine-tuned model (full fine-tuning). |
| `output_results` | `correct_output_model_<model>.txt`: settings, accuracy and the correctly predicted test items. |

## Notes

- The `gptj_english` preset keeps the original script's target modules, `q_proj`, `k_proj`, `v_proj`, `o_proj`. GPT-J names its output projection `out_proj`, so only the query, key and value projections get adapters.
- The CAP preset reproduces the split CAP ran with (`test_size=0.8`, then 10 % of the held-out part for validation); prompts are right-padded for the definiendum-only loss.
