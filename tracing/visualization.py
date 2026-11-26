"""Heatmaps of causal traces.

The heatmap colour is the raw patched-run probability p(answer) of the
definiendum, with ``vmin`` at the corrupted baseline (``low_score``). Rows are
prompt tokens (corrupted definition tokens marked ``*``), columns are layers.
"""

import os

from matplotlib import pyplot as plt

from .model import guess_subject
from .patching import calculate_hidden_flow

CMAPS = {None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    """Draw one trace (a ``calculate_hidden_flow`` result or a loaded npz)."""
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(differences, cmap=CMAPS[kind], vmin=low_score)
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # cb.ax.set_xlabel would be the natural call; it is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf) or ".", exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    """Trace one prompt and draw its heatmap."""
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, window=10):
    """Draw the MLP, attention and hidden-state traces of one prompt."""
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, noise=noise, window=window, kind=kind)
