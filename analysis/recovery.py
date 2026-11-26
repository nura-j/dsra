"""Recovery scores and top-K recovery-mass shares.

Top-K selection is magnitude-based, so the statistic that justifies K is the
share of a case's total recovery that its K best states hold:
``sum(top-K T) / sum(T)``, averaged over cases. Cases with a degenerate
clean-vs-corrupted gap or zero total recovery are skipped.
"""

import glob
import re
import statistics as st
from pathlib import Path

import numpy as np

from .config import ARCHIVED_RUNS, KINDS, AnalysisConfig


def case_files(run_root):
    """Hidden-state trace files ``knowledge_<id>.npz`` of a run (the MLP and
    attention files are found from them by suffix)."""
    return sorted(
        f
        for f in glob.glob(f"{run_root}/causal_trace/cases/knowledge_*.npz")
        if re.fullmatch(r"knowledge_\d+\.npz", Path(f).name)
    )


def kind_cases(run_root, suffix, correct_only=True):
    """Yield the loaded npz of every case of one trace kind."""
    for hidden_file in case_files(run_root):
        try:
            z = np.load(hidden_file.replace(".npz", suffix + ".npz"), allow_pickle=True)
        except FileNotFoundError:
            continue
        if "scores" not in z:
            continue
        if correct_only and not bool(z["correct_prediction"]):
            continue
        yield z


def recovery_scores(z, min_gap=1e-6):
    """Per-cell recovery T in [0, 1], or None for a degenerate gap."""
    lo, hi = float(z["low_score"]), float(z["high_score"])
    if hi - lo <= min_gap:
        return None
    return np.clip((z["scores"] - lo) / (hi - lo), 0, 1)


def mass_profile(T, ks=(10, 50)):
    """Top-K mass shares (%) and the states needed for 50% / 80% of the mass.

    Returns None when the case has no recovery mass.
    """
    tot = T.sum()
    if tot <= 0:
        return None
    flat = np.sort(T.ravel())[::-1]
    c = np.cumsum(flat) / tot
    return {
        "shares": {k: flat[:k].sum() / tot * 100 for k in ks},
        "n50": int(np.searchsorted(c, 0.5) + 1),
        "n80": int(np.searchsorted(c, 0.8) + 1),
    }


def mass_stats(run_root, config: AnalysisConfig = None):
    """Recovery-mass summary of one run, per trace kind."""
    config = config or AnalysisConfig()
    out = {}
    for kind, suffix in KINDS:
        shares = {k: [] for k in config.mass_ks}
        n50, n80 = [], []
        for z in kind_cases(run_root, suffix, config.correct_only):
            T = recovery_scores(z, config.min_gap)
            if T is None:
                continue
            prof = mass_profile(T, config.mass_ks)
            if prof is None:
                continue
            for k in config.mass_ks:
                shares[k].append(prof["shares"][k])
            n50.append(prof["n50"])
            n80.append(prof["n80"])
        if not n50:
            continue
        row = {"n_cases": len(n50)}
        for k in config.mass_ks:
            row[f"top{k}_mass_pct_mean"] = round(st.mean(shares[k]), 1)
            row[f"top{k}_mass_pct_median"] = round(st.median(shares[k]), 1)
            row[f"top{k}_mass_pct_sd"] = round(st.pstdev(shares[k]), 1)
        row["states_for_50pct_mass_median"] = st.median(n50)
        row["states_for_80pct_mass_median"] = st.median(n80)
        out[kind] = row
    return out


def archived_mass_stats(archive_root, runs=ARCHIVED_RUNS):
    """Mass summaries of the archived GPT-J-6B / BERTIN runs under
    ``<archive_root>/results``."""
    return {
        name: mass_stats(
            f"{archive_root}/results/{run_dir}",
            AnalysisConfig(correct_only=correct_only),
        )
        for name, (run_dir, correct_only) in runs.items()
    }
