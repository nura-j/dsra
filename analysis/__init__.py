"""Aggregation of DSRA causal traces.

- ``recovery``: recovery scores T and top-K recovery-mass shares.
- ``localisation``: normalised-depth layer profiles, final-token share and
  the bimodality check, per run and trace kind.
- ``grid_stats``: top-K thresholds as shares of the token x layer grid.
- ``visualization``: the cross-model layer-distribution figure.
- ``checks``: sanity checks on a tracing run.
- ``summarise``: the command line.
"""

from .checks import check_run, problems
from .config import ARCHIVED_RUNS, CROSS_MODEL_RUNS, KINDS, AnalysisConfig
from .grid_stats import grid_stats
from .localisation import add_derived, analyse, bimodality, cross_model
from .recovery import (
    archived_mass_stats,
    case_files,
    mass_profile,
    mass_stats,
    recovery_scores,
)
from .visualization import plot_cross_model

__all__ = [
    "check_run",
    "problems",
    "ARCHIVED_RUNS",
    "CROSS_MODEL_RUNS",
    "KINDS",
    "AnalysisConfig",
    "grid_stats",
    "add_derived",
    "analyse",
    "bimodality",
    "cross_model",
    "archived_mass_stats",
    "case_files",
    "mass_profile",
    "mass_stats",
    "recovery_scores",
    "plot_cross_model",
]
