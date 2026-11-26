"""Causal tracing over definitions (the DSRA tracer).

Adapted from ROME's corrupt-and-restore tracing: the definition tokens are
corrupted with Gaussian noise, and clean hidden states, MLP outputs or
attention outputs are restored one (token, layer) cell at a time to measure
how much each recovers the probability of the definiendum.
"""

from .config import GPTJ_NOISE, KINDS, TraceConfig, parse_noise, window_for
from .model import (
    ModelAndTokenizer,
    decode_tokens,
    find_token_range,
    guess_subject,
    layername,
    make_inputs,
    predict_from_input,
    predict_token,
    resolve_dtype,
)
from .noise import (
    collect_embedding_gaussian,
    collect_embedding_std,
    collect_embedding_tdist,
    resolve_noise,
)
from .patching import (
    calculate_hidden_flow,
    trace_important_states,
    trace_important_window,
    trace_with_patch,
    trace_with_repatch,
)
from .runner import load_facts, load_model, trace_facts
from .visualization import plot_all_flow, plot_hidden_flow, plot_trace_heatmap

__all__ = [
    "GPTJ_NOISE",
    "KINDS",
    "TraceConfig",
    "parse_noise",
    "window_for",
    "ModelAndTokenizer",
    "decode_tokens",
    "find_token_range",
    "guess_subject",
    "layername",
    "make_inputs",
    "predict_from_input",
    "predict_token",
    "resolve_dtype",
    "collect_embedding_gaussian",
    "collect_embedding_std",
    "collect_embedding_tdist",
    "resolve_noise",
    "calculate_hidden_flow",
    "trace_important_states",
    "trace_important_window",
    "trace_with_patch",
    "trace_with_repatch",
    "load_facts",
    "load_model",
    "trace_facts",
    "plot_all_flow",
    "plot_hidden_flow",
    "plot_trace_heatmap",
]
