"""Corrupt-and-restore causal tracing (ROME), applied to definitions.

Each trace runs one batch: row 0 is the clean prompt, rows 1..n are the same
prompt with noise added to the definition-token embeddings. Restoring a
selected clean state into the corrupted rows and reading the probability of
the definiendum shows how much that state carries.

- ``trace_important_states``: restore one hidden state (token i, layer j).
- ``trace_important_window``: restore the MLP or attention output at token i
  over a window of layers centred on j.
"""

from collections import defaultdict

import numpy
import torch

from legacy.rome import nethook

from .model import decode_tokens, find_token_range, layername, make_inputs, predict_from_input


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def trace_with_patch(
    model,
    inp,
    states_to_patch,
    answers_t,
    tokens_to_mix,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    trace_layers=None,
):
    """Run one patched batch and return the mean probability of ``answers_t``.

    Args:
        model: the causal LM.
        inp: batch from ``make_inputs``; row 0 clean, rows 1.. corrupted.
        states_to_patch: ``[(token_index, layername), ...]`` restored from row 0.
        answers_t: token id(s) whose probability is reported.
        tokens_to_mix: ``(begin, end)`` token range whose embeddings get noise.
        noise: float scale, or a callable applied to the base samples.
        uniform_noise: draw base samples from U(-1, 1) instead of N(0, 1).
        replace: replace the embeddings with noise instead of adding it.
        trace_layers: also return these layers' outputs, stacked on dim 2.
    """
    rs = numpy.random.RandomState(1)  # pseudorandom noise, for reproducibility
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # Corrupt the token range on batch rows 1.. (row 0 stays clean).
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # Restore the clean state of the selected tokens in the corrupted rows.
        h = _untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # Softmax probability of the answer, averaged over the corrupted rows.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    if trace_layers is not None:
        all_traced = torch.stack(
            [_untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced
    return probs


def trace_with_repatch(
    model,
    inp,
    states_to_patch,
    states_to_unpatch,
    answers_t,
    tokens_to_mix,
    noise=0.1,
    uniform_noise=False,
):
    """Like ``trace_with_patch``, but also re-corrupts ``states_to_unpatch``.

    A first pass records the corrupted run; the second pass restores
    ``states_to_patch`` from the clean row and puts the first-pass corrupted
    values back into ``states_to_unpatch``.
    """
    rs = numpy.random.RandomState(1)
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")
    state = {}

    def patch_rep(x, layer):
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if state["first_pass"] or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        h = _untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = _untuple(state["first_pass_trace"][layer].output)[1:, t]
        return x

    for first_pass in [True, False] if states_to_unpatch else [False]:
        state["first_pass"] = first_pass
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                state["first_pass_trace"] = td

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    return probs


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    """Grid of answer probabilities, restoring one hidden state (token, layer)."""
    ntoks = inp["input_ids"].shape[1]
    table = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    """Grid of answer probabilities, restoring ``kind`` ("mlp" / "attn") at one
    token over the ``window`` layers centred on each layer."""
    ntoks = inp["input_ids"].shape[1]
    table = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """Trace every (token, layer) cell for one prompt.

    Returns ``dict(correct_prediction=False)`` when the clean argmax token is
    not ``expect``; otherwise the score grid with ``low_score`` (corrupted),
    ``high_score`` (clean), the tokens, the corrupted ``subject_range``, the
    answer, ``window`` and ``kind``. This is the schema of the saved npz.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device=mt.device)
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )
