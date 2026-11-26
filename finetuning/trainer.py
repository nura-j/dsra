"""Training with the Hugging Face ``Trainer`` and saving the weights."""

import inspect
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from .config import FinetuneConfig


def training_arguments(config: FinetuneConfig, output_dir):
    """``TrainingArguments`` valid on old and new transformers.

    ``evaluation_strategy`` became ``eval_strategy`` (4.41) and some options
    were removed later; options the installed version does not know are left out.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=config.train,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        use_cpu=config.device == "cpu",
    )
    kwargs[strategy] = config.eval_strategy
    if config.learning_rate is not None:
        kwargs["learning_rate"] = config.learning_rate
    if config.weight_decay is not None:
        kwargs["weight_decay"] = config.weight_decay
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


def train(model, tokenizer, train_data, eval_data, config: FinetuneConfig, output_dir):
    """Train and return the trained model.

    Sequence-loss data are lists of tokenised texts, batched by the
    language-modelling collator; target-loss data are ``TargetOnlyDataset``
    items, already padded and labelled.
    """
    collator = None
    if config.loss == "sequence":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_arguments(config, output_dir),
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )
    trainer.train()
    return model, trainer


def save(model, paths):
    """Save the weights. LoRA runs keep the adapter (``unmerged``) and the
    merged model (``merged``); full fine-tuning saves the model to ``merged``
    (a ``weights`` folder).
    Returns the merged model."""
    if hasattr(model, "merge_and_unload") and paths.get("unmerged"):
        model.save_pretrained(paths["unmerged"])
        model = model.merge_and_unload()
    model.save_pretrained(paths["merged"])
    return model
