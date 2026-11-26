"""Configuration for aggregating causal traces.

The recovery score of a restored state is
``T = clip((p_patched - p_corrupted) / (p_clean - p_corrupted), 0, 1)``.
The analyses summarise, per run and trace kind, how much of the total
recovery the top-K states hold and where those states sit (normalised depth,
final token).
"""

from dataclasses import dataclass
from typing import Tuple

# Display names of the cross-model runs, keyed by results folder.
CROSS_MODEL_RUNS = {
    "ns3_r0_gpt2-large": "GPT-2 Large",
    "ns3_r0_google_gemma-2b": "Gemma-2B",
    "ns3_r0_Qwen_Qwen2.5-3B": "Qwen2.5-3B",
    "ns3_r0_meta-llama_Llama-3.2-3B": "Llama-3.2-3B",
    "ns3_r0_EleutherAI_pythia-1b": "Pythia-1B",
    "ns3_r0_meta-llama_Llama-3.2-1B": "Llama-3.2-1B",
    "ns3_r0_Qwen_Qwen2.5-1.5B": "Qwen2.5-1.5B",
}

# Archived GPT-J-6B / BERTIN runs: name -> (results folder, correct cases only).
ARCHIVED_RUNS = {
    "gptj_curated_all_correct": ("ns3_reffered_r0_EleutherAI_gpt-j-6B_3", False),
    "gptj_main_correct_only": ("ns3_reffered_r0_EleutherAI_gpt-j-6B", True),
    "bertin_final_correct_only": (
        "ns03_fixed_prompt_final_r0_bertin-project_bertin-gpt-j-6B",
        True,
    ),
}

# Trace kind -> npz file suffix.
KINDS = (("hidden", ""), ("mlp", "_mlp"), ("attn", "_attn"))


@dataclass
class AnalysisConfig:
    """Settings for the recovery and localisation summaries.

    Attributes:
        results_dir: folder holding the ``<run>/causal_trace/cases`` runs.
        topk: K for the localisation of top-K states.
        mass_ks: K values for the recovery-mass shares.
        early: normalised depth below which a state is "early".
        upper: normalised depth above which a state is "upper".
        correct_only: skip cases whose clean prediction was wrong.
        min_gap: skip cases with ``p_clean - p_corrupted`` at or below this.
    """

    results_dir: str = "results"
    topk: int = 10
    mass_ks: Tuple[int, ...] = (10, 50)
    early: float = 0.25
    upper: float = 0.60
    correct_only: bool = True
    min_gap: float = 1e-6

    def __post_init__(self):
        self.mass_ks = tuple(self.mass_ks)
        if self.topk < 1 or any(k < 1 for k in self.mass_ks):
            raise ValueError("K values must be positive")
        if not 0 <= self.early < self.upper <= 1:
            raise ValueError("need 0 <= early < upper <= 1")
