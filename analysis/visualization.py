"""Cross-model figure: layer distribution of the top-K states per trace kind.

One row per run, one column per kind (MLP, attention, hidden states), with
the layer axis as normalised depth. Grey bands mark the GPT-J-6B ranges
reported in the paper; the dashed line is the mean depth of the top-K states.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette of the method figure: attention blue, MLP green, hidden-state amber.
CHARCOAL = "#333333"
GREY = "#666666"
GRID = "#EEEEEE"
KIND_STYLE = {
    # key: (display name, line colour, fill colour)
    "mlp": ("MLP", "#009E73", "#D5E8D4"),
    "attn": ("Attention", "#0072B2", "#DAE8FC"),
    # The method figure's amber (#D6B656) is a box stroke and reads too pale as
    # a plot line, so the deeper Okabe-Ito amber is used at the same hue.
    "hidden": ("Hidden states", "#E69F00", "#FBE3B8"),
}

# The GPT-J-6B bands reported in the paper, as normalised depth over its 28
# layers: MLP 0-5 and 17-21, attention 12-25, hidden states 20-27.
GPTJ_BANDS = {
    "mlp": [(0 / 27, 5 / 27), (17 / 27, 21 / 27)],
    "attn": [(12 / 27, 25 / 27)],
    "hidden": [(20 / 27, 27 / 27)],
}
BAND = "#F2F2F2"

# Grouped by family, ascending scale within a family.
ORDER = [
    "ns3_r0_gpt2-large",
    "ns3_r0_EleutherAI_pythia-1b",
    "ns3_r0_google_gemma-2b",
    "ns3_r0_Qwen_Qwen2.5-1.5B",
    "ns3_r0_Qwen_Qwen2.5-3B",
    "ns3_r0_meta-llama_Llama-3.2-1B",
    "ns3_r0_meta-llama_Llama-3.2-3B",
]


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": CHARCOAL,
        "axes.linewidth": 0.6,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_cross_model(data, out):
    """Draw the figure from a ``cross_model`` result; writes ``out`` (PDF) and a PNG."""
    runs = [r for r in ORDER if r in data] + [r for r in data if r not in ORDER]
    apply_style()

    nrow, ncol = len(runs), len(KIND_STYLE)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(6.30, 0.95 * nrow + 0.6), sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.16},
    )
    axes = np.atleast_2d(axes)

    for i, run in enumerate(runs):
        v = data[run]
        for j, (kind, (kname, line, fill)) in enumerate(KIND_STYLE.items()):
            ax = axes[i, j]
            s = v[kind]
            pct = np.asarray(s["layer_pct"], dtype=float)
            L = len(pct)
            x = np.arange(L) / max(L - 1, 1)

            for b0, b1 in GPTJ_BANDS[kind]:  # GPT-J reference, behind the curve
                ax.axvspan(b0, b1, color=BAND, lw=0, zorder=1)
            ax.fill_between(x, pct, step="mid", color=fill, zorder=2)
            ax.step(x, pct, where="mid", color=line, lw=1.0, zorder=3)
            # mean normalised depth of the top-K states
            ax.axvline(s["depth_mean"], color=line, lw=0.7, ls=(0, (2.5, 1.5)), zorder=4)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, max(pct.max() * 1.22, 1e-6))
            ax.set_yticks([])
            ax.grid(False)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(CHARCOAL)
            ax.tick_params(axis="x", length=2.5, width=0.6, colors=CHARCOAL)

            # peak height, set above the panel so it never sits on the curve
            ax.text(1.0, 1.02, f"peak {pct.max():.0f}%", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.0, color=GREY)

            if i == 0:
                ax.set_title(kname, color=CHARCOAL, pad=13)
            if i == nrow - 1:
                ax.set_xlabel("normalised depth", color=CHARCOAL)
                ax.set_xticks([0, 0.5, 1.0])
            if j == 0:
                ax.text(-0.075, 0.5, f"{v['label']}\n$L{{=}}${v['layers']}, $n{{=}}${s['n_cases']}",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=7, color=CHARCOAL, linespacing=1.35)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out
