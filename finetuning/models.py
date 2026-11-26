"""Loading the base model and tokenizer, and attaching or merging LoRA adapters."""

import os

import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import FinetuneConfig

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
# transformers 4.56 renamed from_pretrained's ``torch_dtype`` to ``dtype``.
DTYPE_KWARG = (
    "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
)


def load_tokenizer(config: FinetuneConfig, name=None):
    """Tokenizer with the EOS token as padding token."""
    tokenizer = AutoTokenizer.from_pretrained(
        name or config.model_name, token=os.environ.get("HF_TOKEN"), **config.tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: FinetuneConfig, name=None):
    """Base model; merges ``adapter_path`` if given, else adds fresh LoRA
    adapters when ``method == "lora"``."""
    device_map = "auto" if config.device == "auto" else config.device
    model = AutoModelForCausalLM.from_pretrained(
        name or config.model_name,
        device_map=device_map,
        token=os.environ.get("HF_TOKEN"),
        **{DTYPE_KWARG: DTYPES[config.torch_dtype]},
    )
    if config.adapter_path:
        from peft import PeftModel

        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, config.adapter_path).merge_and_unload()
    if config.method == "lora" and config.train:
        from peft import LoraConfig, get_peft_model

        kwargs = dict(task_type="CAUSAL_LM", inference_mode=False, r=config.r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        if config.target_modules is not None:
            kwargs["target_modules"] = list(config.target_modules)
        model = get_peft_model(model, LoraConfig(**kwargs))
        model.print_trainable_parameters()
    return model
