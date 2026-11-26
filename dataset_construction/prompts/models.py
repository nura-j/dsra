"""Models supported for prompting, fine-tuning and tracing.

Keys are Hugging Face ids. ``aliases`` are the short names used by the CAP and
CARMA code (``GPT2_large``, ``gemma_2b``, ``Qwen2_5_3B``, ...) -> commands
from those repositories resolve to the same model. ``family`` decides the
module tree for tracing and the default LoRA target modules.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelInfo:
    family: str                 # "gptj", "gpt2", "neox", "gemma", "llama", "mistral", "qwen2"
    language: str = "en"
    aliases: Tuple[str, ...] = ()


MODELS = {
    # DSRA paper models.
    "EleutherAI/gpt-j-6B": ModelInfo("gptj", aliases=("gpt-j-6B", "gpt_j_6B", "GPTJ")),
    "bertin-project/bertin-gpt-j-6B": ModelInfo("gptj", "es", aliases=("bertin-gpt-j-6B",)),
    # GPT-2 (CAP, CARMA).
    "gpt2": ModelInfo("gpt2", aliases=("GPT2",)),
    "gpt2-medium": ModelInfo("gpt2", aliases=("GPT2-medium", "GPT2_medium")),
    "gpt2-large": ModelInfo("gpt2", aliases=("GPT2-large", "GPT2_large")),
    # Pythia (CARMA).
    "EleutherAI/pythia-14m": ModelInfo("neox", aliases=("pythia_14m",)),
    "EleutherAI/pythia-410m": ModelInfo("neox", aliases=("pythia_410m",)),
    "EleutherAI/pythia-1b": ModelInfo("neox", aliases=("pythia_1b",)),
    # Gemma (CAP, CARMA).
    "google/gemma-2b": ModelInfo("gemma", aliases=("gemma-2b", "gemma_2b", "google_gemma_2b")),
    # Llama (CAP, CARMA).
    "meta-llama/Llama-2-7b-hf": ModelInfo(
        "llama", aliases=("Llama-2-7b", "Llama-2-7b-hf", "Llama_2_7b", "Llama_2_7b_hf")),
    "meta-llama/Llama-2-7b-chat-hf": ModelInfo("llama", aliases=("Llama-2-7b-chat-hf",)),
    "meta-llama/Meta-Llama-3-8B": ModelInfo(
        "llama", aliases=("Meta-Llama-3-8B", "Meta_Llama_3_8B")),
    "meta-llama/Meta-Llama-3-70B": ModelInfo(
        "llama", aliases=("Meta-Llama-3-70B", "Meta_Llama_3_70B")),
    "meta-llama/Llama-3.1-8B": ModelInfo(
        "llama", aliases=("Llama_3_1_8B", "meta_llama_Llama_3_1_8B")),
    "meta-llama/Llama-3.2-1B": ModelInfo(
        "llama", aliases=("Llama_3_2_1B", "meta_llama_Llama_3_2_1B")),
    "meta-llama/Llama-3.2-3B": ModelInfo(
        "llama", aliases=("Llama_3_2_3B", "meta_llama_Llama_3_2_3B")),
    # Mistral (CAP, CARMA).
    "mistralai/Mistral-7B-v0.1": ModelInfo("mistral", aliases=("mistral-7b", "mistral_7b")),
    "mistralai/Mistral-7B-v0.3": ModelInfo("mistral"),
    # Qwen2 and Qwen2.5 (CAP, CARMA).
    "Qwen/Qwen2-0.5B": ModelInfo("qwen2", aliases=("Qwen2_0_5B",)),
    "Qwen/Qwen2-1.5B": ModelInfo("qwen2", aliases=("Qwen2_1_5B",)),
    "Qwen/Qwen2-7B": ModelInfo("qwen2", aliases=("Qwen2_7B",)),
    "Qwen/Qwen2.5-0.5B": ModelInfo(
        "qwen2", aliases=("Qwen2.5-0.5B", "qwen2.5-0.5b", "Qwen2_5_0_5B", "Qwen_Qwen2_5_0_5B")),
    "Qwen/Qwen2.5-1.5B": ModelInfo(
        "qwen2", aliases=("Qwen2.5-1.5B", "Qwen2_5_1_5B", "Qwen_Qwen2_5_1_5B")),
    "Qwen/Qwen2.5-3B": ModelInfo(
        "qwen2", aliases=("Qwen2.5-3B", "Qwen2_5_3B", "Qwen_Qwen2_5_3B")),
}

_ALIASES = {alias: name for name, info in MODELS.items() for alias in info.aliases}


def resolve_model(name):
    """Hugging Face id for an id, a CAP/CARMA alias, or a local model folder.

    Unknown names are returned unchanged, so local checkpoints still work.
    """
    if name in MODELS:
        return name
    return _ALIASES.get(name, name)


def model_info(name):
    """``ModelInfo`` for a model, or None for models outside the registry."""
    return MODELS.get(resolve_model(name))
