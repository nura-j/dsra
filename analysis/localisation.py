"""Where the top-K restored states sit, per run and trace kind.

Layers are reported as normalised depth ``layer / (L - 1)`` so models of 16
to 36 layers can be compared. Depth bands follow the GPT-J reading: early
below 0.25, middle 0.25-0.60, upper above 0.60. ``bimodality`` checks the
two-mode MLP profile reported for GPT-J.
"""

import statistics as st

import numpy as np

from .config import KINDS, AnalysisConfig
from .recovery import case_files, mass_profile, recovery_scores


def analyse(run_root, config: AnalysisConfig = None):
    """Recovery-mass and localisation summary of one run, per trace kind."""
    config = config or AnalysisConfig()
    topk, early, upper = config.topk, config.early, config.upper
    out = {}
    for kind, suffix in KINDS:
        shares = {k: [] for k in config.mass_ks}
        n50, n80 = [], []
        layer_hist = None
        depths, final_tok = [], []
        n_layers = None
        windows = set()
        for hidden_file in case_files(run_root):
            f = hidden_file.replace(".npz", suffix + ".npz")
            try:
                z = np.load(f, allow_pickle=True)
            except FileNotFoundError:
                continue
            if "scores" not in z or not bool(z["correct_prediction"]):
                continue
            T = recovery_scores(z, config.min_gap)
            if T is None:
                continue
            prof = mass_profile(T, config.mass_ks)
            if prof is None:
                continue
            n_tok, n_layers = T.shape
            if layer_hist is None:
                layer_hist = np.zeros(n_layers, dtype=np.int64)
            if "window" in z:
                windows.add(int(z["window"]))

            for k in config.mass_ks:
                shares[k].append(prof["shares"][k])
            n50.append(prof["n50"])
            n80.append(prof["n80"])

            # Top-K cells of this case, as (token, layer) pairs.
            k = min(topk, T.size)
            idx = np.argpartition(T.ravel(), -k)[-k:]
            ti, li = np.unravel_index(idx, T.shape)
            np.add.at(layer_hist, li, 1)
            depths.extend((li / max(n_layers - 1, 1)).tolist())
            final_tok.append(float((ti == n_tok - 1).mean() * 100))

        if not n50:
            continue
        if len(windows) > 1:
            raise ValueError(
                f"{run_root} {kind}: mixed windows {sorted(windows)} in one run; "
                "delete the stale npz and retrace before aggregating"
            )
        d = np.asarray(depths)
        hist = layer_hist / layer_hist.sum() * 100
        row = {
            "n_cases": len(n50),
            "n_layers": n_layers,
            # 0 for the hidden-state traces, which never use a window.
            "window": windows.pop() if windows else None,
        }
        for k in config.mass_ks:
            row[f"top{k}_mass_pct_mean"] = round(st.mean(shares[k]), 1)
            row[f"top{k}_mass_pct_sd"] = round(st.pstdev(shares[k]), 1)
        row.update({
            "states_for_50pct_mass_median": st.median(n50),
            "states_for_80pct_mass_median": st.median(n80),
            "depth_mean": round(float(d.mean()), 3),
            "depth_median": round(float(np.median(d)), 3),
            "pct_early": round(float((d < early).mean() * 100), 1),
            "pct_middle": round(float(((d >= early) & (d <= upper)).mean() * 100), 1),
            "pct_upper": round(float((d > upper).mean() * 100), 1),
            "final_token_pct_mean": round(float(np.mean(final_tok)), 1),
            "peak_layer": int(np.argmax(hist)),
            "layer_pct": [round(float(v), 2) for v in hist],
        })
        out[kind] = row
    return out


def bimodality(layer_pct):
    """Two-mode summary of a layer distribution.

    Scans every candidate trough and keeps the split that separates the
    distribution best, reporting

        dip        = 1 - trough / min(left peak, right peak)   separation
        balance    = min(left peak, right peak) / max(...)     relative size
        early_mass = share of the top-K states below the trough

    Bimodal in the GPT-J sense needs both a high dip and a non-trivial
    balance: a high dip alone also fires on a one-cell blip next to a single
    dominant mode.
    """
    y = np.asarray(layer_pct, dtype=float)
    L = len(y)
    if L < 5 or y.sum() <= 0:
        return {"dip": 0.0, "balance": 0.0, "early_mass_pct": 0.0, "mode_depths": []}
    k = np.convolve(y, np.ones(3) / 3, mode="same")  # light smoothing

    best = None
    for t in range(1, L - 1):
        lo, hi = float(k[:t].max()), float(k[t + 1:].max())
        h = min(lo, hi)
        if h <= 0:
            continue
        score = 1 - k[t] / h
        if best is None or score > best[0]:
            best = (score, t, lo, hi)
    if best is None:
        return {"dip": 0.0, "balance": 0.0, "early_mass_pct": 0.0, "mode_depths": []}

    score, t, lo, hi = best
    p1 = int(np.argmax(k[:t]))
    p2 = int(t + 1 + np.argmax(k[t + 1:]))
    return {
        "dip": round(float(score), 3),
        "balance": round(min(lo, hi) / max(lo, hi), 3) if max(lo, hi) > 0 else 0.0,
        "early_mass_pct": round(float(y[:t].sum()), 1),
        "trough_depth": round(t / (L - 1), 3),
        "mode_depths": [round(p1 / (L - 1), 3), round(p2 / (L - 1), 3)],
        "mode_heights": [round(float(k[p1]), 2), round(float(k[p2]), 2)],
    }


def add_derived(res):
    """Fields computed from ``layer_pct`` alone (no npz reads): the run's
    severed window, layer count, window-to-depth ratio and bimodality."""
    for v in res.values():
        # The severed window is the MLP/attention one; hidden states have none.
        w = v["mlp"].get("window") if "mlp" in v else None
        v["window"] = w
        v["layers"] = v["mlp"]["n_layers"] if "mlp" in v else None
        v["window_depth_ratio"] = round(w / v["layers"], 3) if w and v["layers"] else None
        for kind, _ in KINDS:
            if kind in v:
                v[kind]["bimodality"] = bimodality(v[kind]["layer_pct"])
    return res


def cross_model(runs, config: AnalysisConfig = None):
    """Analyse every run found under ``config.results_dir``.

    ``runs`` maps results folder -> display label.
    """
    from pathlib import Path

    config = config or AnalysisConfig()
    res = {}
    for run, label in runs.items():
        root = Path(config.results_dir) / run
        if not root.is_dir():
            print(f"skip {run}: not found under {config.results_dir}")
            continue
        stats = analyse(str(root), config)
        if not stats:
            print(f"skip {run}: no usable cases")
            continue
        res[run] = {"label": label, "topk": config.topk, **stats}
    return add_derived(res)
