"""Trace a fact file and save one npz per case and trace kind.

Output layout, under ``TraceConfig.run_dir``::

    cases/knowledge_<id>.npz        hidden-state trace
    cases/knowledge_<id>_mlp.npz    MLP trace (severed window)
    cases/knowledge_<id>_attn.npz   attention trace (severed window)
    pdfs/<answer>_<id>[_mlp|_attn].pdf   heatmaps for ids up to max_plot_id

Existing npz files are reused, so an interrupted run can be resumed.
"""

import json
import os

import numpy
import torch
from tqdm import tqdm

from .config import TraceConfig
from .model import ModelAndTokenizer, resolve_dtype
from .noise import resolve_noise
from .patching import calculate_hidden_flow
from .visualization import plot_trace_heatmap

# Trace kind -> (value passed to the tracer, file suffix).
KIND_SPEC = {"hidden": (None, ""), "mlp": ("mlp", "_mlp"), "attn": ("attn", "_attn")}


def load_facts(path):
    """Fact records: ``known_id``, ``subject`` (definition), ``attribute``
    (definiendum) and ``prompt``."""
    with open(path) as f:
        return json.load(f)


def to_numpy(value):
    """Tensors to numpy. NumPy has no bfloat16, so floating tensors are cast to
    float32 first; integer tensors keep their dtype."""
    if not torch.is_tensor(value):
        return value
    value = value.detach().cpu()
    if value.is_floating_point():
        value = value.float()
    return value.numpy()


def load_model(config: TraceConfig) -> ModelAndTokenizer:
    return ModelAndTokenizer(
        config.model_name,
        torch_dtype=resolve_dtype(config.torch_dtype, config.model_name),
        device=config.device,
    )


def trace_facts(config: TraceConfig, facts, mt=None):
    """Trace every fact for every kind in ``config.kinds``.

    Returns the list of ``known_id`` values whose clean prediction was correct.
    """
    torch.set_grad_enabled(False)
    result_dir = os.path.join(config.run_dir, "cases")
    pdf_dir = os.path.join(config.run_dir, "pdfs")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    if mt is None:
        mt = load_model(config)
    if config.num_cases > 0:
        facts = facts[: config.num_cases]
    noise, uniform_noise = resolve_noise(mt, config.noise, [k["subject"] for k in facts])

    correct = []
    for knowledge in tqdm(facts):
        known_id = knowledge["known_id"]
        ok = True
        for name in config.kinds:
            kind, suffix = KIND_SPEC[name]
            filename = os.path.join(result_dir, f"knowledge_{known_id}{suffix}.npz")
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    samples=config.samples,
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=config.replace,
                    window=config.window,
                )
                numpy_result = {k: to_numpy(v) for k, v in result.items()}
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                ok = False
                continue
            if known_id > config.max_plot_id:
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            answer = str(numpy_result["answer"]).strip()
            plot_trace_heatmap(
                plot_result, savepdf=os.path.join(pdf_dir, f"{answer}_{known_id}{suffix}.pdf")
            )
        if ok:
            correct.append(known_id)
    return correct
