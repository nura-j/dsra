"""Sanity checks on a tracing run.

For every trace kind of a run, reports:

- how many npz files there are and how many have a correct clean prediction
  (with screened fact files this should be all of them);
- how often corruption lowered the answer probability (``low < high``);
- the stored severed-window widths (an MLP / attention run should have one);
- how many severed-window maps are flat at ``low_score``, the sign of a window
  that restored no layers.

Hidden-state traces restore single states and never use the window, so their
stored ``window`` is informational only; judge a run's window by its MLP and
attention files.
"""

import numpy as np

from .config import KINDS
from .recovery import case_files


def check_run(run_root):
    """Per-kind check summary of the run under ``run_root``."""
    out = {}
    for kind, suffix in KINDS:
        n_files = n_correct = n_corrupt_lower = n_flat = 0
        windows = {}
        for hidden_file in case_files(run_root):
            try:
                z = np.load(hidden_file.replace(".npz", suffix + ".npz"), allow_pickle=True)
            except FileNotFoundError:
                continue
            n_files += 1
            if not bool(z["correct_prediction"]):
                continue
            n_correct += 1
            low, high = float(z["low_score"]), float(z["high_score"])
            n_corrupt_lower += low < high
            if "window" in z:
                w = int(z["window"])
                windows[w] = windows.get(w, 0) + 1
            if kind != "hidden" and np.allclose(z["scores"], low):
                n_flat += 1
        if not n_files:
            continue
        out[kind] = {
            "npz_files": n_files,
            "correct": n_correct,
            "correct_pct": round(100 * n_correct / n_files, 1),
            "corruption_lowers_pct": round(100 * n_corrupt_lower / n_correct, 1) if n_correct else None,
            "windows": dict(sorted(windows.items())),
            "flat_maps": n_flat if kind != "hidden" else None,
        }
    return out


def problems(summary):
    """Human-readable warnings for a ``check_run`` summary (empty if none)."""
    found = []
    for kind, s in summary.items():
        if s["correct"] < s["npz_files"]:
            found.append(f"{kind}: {s['npz_files'] - s['correct']} cases with a wrong clean "
                         "prediction (fact files are expected to be screened)")
        if s["corruption_lowers_pct"] is not None and s["corruption_lowers_pct"] < 90:
            found.append(f"{kind}: corruption lowers p(answer) in only "
                         f"{s['corruption_lowers_pct']}% of cases")
        if kind != "hidden":
            if len(s["windows"]) > 1:
                found.append(f"{kind}: mixed windows {sorted(s['windows'])}; "
                             "delete the stale npz and retrace")
            if any(w < 2 for w in s["windows"]):
                found.append(f"{kind}: window below 2 restores no layers")
            if s["flat_maps"]:
                found.append(f"{kind}: {s['flat_maps']} maps flat at low_score")
    return found
