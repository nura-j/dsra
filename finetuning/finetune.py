"""Fine-tune a model on the inverse dictionary task and evaluate it.

Run from the repository root:

    # DSRA paper models (LoRA)
    python -m finetuning.finetune --preset gptj_english
    python -m finetuning.finetune --preset bertin_spanish --data_path data/WordNet_filtered_data_es_with_replacement_with_back_transl_cleaned.json

    # CAP / CARMA (all weights, loss on the definiendum) for any registered model - carma not fully integrated.
    python -m finetuning.finetune --preset cap
    python -m finetuning.finetune --preset cap --model_name gpt2-large

    # a local checkpoint needs its prompt
    python -m finetuning.finetune --preset cap --model_name path/to/model --template '{} is called a "'

Outputs go to <output_dir>/<model>/<method>/<timestamp>_*: checkpoints,
output_results, and weights_unmerged_lora + weights_merged_lora (LoRA) or
weights (full fine-tuning).
"""

import argparse

from .config import FinetuneConfig
from .pipeline import run

PRESETS = {
    "gptj_english": FinetuneConfig.gptj_english,
    "bertin_spanish": FinetuneConfig.bertin_spanish,
    "cap": FinetuneConfig.cap,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True,
                        help="gptj_english / bertin_spanish (paper, LoRA) or cap (CAP/CARMA recipe).")
    parser.add_argument("--model_name", default=None,
                        help="Model id or CAP/CARMA alias (required for --preset cap).")
    parser.add_argument("--data_path", default=None,
                        help="Definitions JSON (required for bertin_spanish; default for cap: "
                             "data/wordnet_data_definitions.json).")
    parser.add_argument("--template", default=None,
                        help='Prompt with {} for the definition (default: the model\'s registered prompt).')
    parser.add_argument("--title_case", choices=["yes", "no"], default=None,
                        help="Title-case the definition (default: the model's setting).")
    parser.add_argument("--adapter_path", default=None,
                        help="Existing LoRA adapter to merge before training or testing.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the preset's epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the batch size.")
    parser.add_argument("--no_train", action="store_true", help="Only evaluate.")
    parser.add_argument("--no_test", action="store_true", help="Only train.")
    parser.add_argument("--output_dir", default=None, help="Root folder for outputs.")
    parser.add_argument("--device", default=None, help='Torch device, or "auto".')
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in dict(
        template=args.template,
        title_case=None if args.title_case is None else args.title_case == "yes",
        adapter_path=args.adapter_path, epochs=args.epochs, batch_size=args.batch_size,
        output_dir=args.output_dir, device=args.device,
    ).items() if v is not None}
    overrides.update(seed=args.seed, train=not args.no_train, test=not args.no_test)
    if args.preset == "cap":
        if not args.model_name:
            raise SystemExit("--model_name is required with --preset cap")
        if args.data_path:
            overrides["data_path"] = args.data_path
        config = FinetuneConfig.cap(args.model_name, **overrides)
    elif args.preset == "bertin_spanish":
        if not args.data_path:
            raise SystemExit("--data_path is required with --preset bertin_spanish")
        config = FinetuneConfig.bertin_spanish(args.data_path, **overrides)
    else:
        if args.model_name:
            overrides["model_name"] = args.model_name
        if args.data_path:
            overrides["data_path"] = args.data_path
        config = FinetuneConfig.gptj_english(**overrides)
    run(config)


if __name__ == "__main__":
    main()
