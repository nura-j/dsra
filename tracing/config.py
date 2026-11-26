"""Configuration for DSRA causal tracing.

``TraceConfig`` holds everything a tracing run needs: the model, the noise
rule for corrupting the definition embeddings, the number of corrupted runs,
the severed-window width for the MLP / attention traces, and where results go.

Presets:

- ``TraceConfig.gptj()``: the GPT-J-6B protocol of the paper (fixed noise
  3 sigma_e = 0.12922, 10 corrupted runs, 10-layer window).
- ``TraceConfig.for_model(name, num_layers)``: the cross-model protocol, with
  per-model 3 sigma noise and a window scaled to the model's depth.
"""

from dataclasses import dataclass, field
import re
from typing import Tuple, Union

# 3 * sigma_e for GPT-J-6B, the ROME noise convention, as used in the paper.
GPTJ_NOISE = 0.12922460585832596

# GPT-J-6B: a 10-layer window over 28 layers.
GPTJ_WINDOW, GPTJ_LAYERS = 10, 28

KINDS = ("hidden", "mlp", "attn")
DTYPES = ("auto", "float32", "float16", "bfloat16")


def window_for(num_layers: int) -> int:
    """Severed-window width that keeps GPT-J's window-to-depth ratio (10/28).

    ``w = 10`` for ``L >= 28``; ``w = max(2, round(10 * L / 28))`` below that.
    For example 6 layers -> 2, 16 -> 6, 18 -> 6, 24 -> 9.
    """
    if num_layers >= GPTJ_LAYERS:
        return GPTJ_WINDOW
    return max(2, round(GPTJ_WINDOW * num_layers / GPTJ_LAYERS))


def parse_noise(rule: Union[float, str]) -> Union[float, str]:
    """Validate a noise rule.

    A float is a fixed Gaussian scale. Strings: ``"s<k>"`` = k times the
    embedding std of the definitions (``"s3"`` is the 3 sigma rule),
    ``"m"`` = multivariate Gaussian matched to the embedding covariance,
    ``"t<df>"`` = Student-t with ``df`` degrees of freedom, ``"u<scale>"`` =
    uniform noise of the given scale.
    """
    if isinstance(rule, (int, float)):
        return float(rule)
    if rule in ("m", "s"):
        return rule
    if re.match(r"^[uts][\d.]+$", rule):
        return rule
    try:
        return float(rule)
    except ValueError:
        raise ValueError(f"unknown noise rule {rule!r}") from None


@dataclass
class TraceConfig:
    """Settings for one causal-tracing run.

    Attributes:
        model_name: Hugging Face id of the causal LM.
        noise: noise rule for the definition embeddings (see ``parse_noise``).
        samples: corrupted runs per input; one clean run is added to the batch.
        window: severed-window width (layers) for the MLP / attention traces.
            Hidden-state traces restore one state and never use it.
        kinds: which traces to run, from ``("hidden", "mlp", "attn")``.
        replace: replace the embeddings with noise instead of adding it.
        torch_dtype: ``"auto"`` (float32, or float16 for 20B models),
            ``"float32"``, ``"float16"`` or ``"bfloat16"``.
        device: torch device for the model and inputs.
        num_cases: trace only the first N facts (-1 = all).
        max_plot_id: draw heatmaps only for facts with ``known_id`` up to this.
        output_dir: results root; ``{run_name}`` is filled in.
    """

    model_name: str = "EleutherAI/gpt-j-6B"
    noise: Union[float, str] = "s3"
    samples: int = 10
    window: int = GPTJ_WINDOW
    kinds: Tuple[str, ...] = KINDS
    replace: bool = False
    torch_dtype: str = "auto"
    device: str = field(default_factory=lambda: _default_device())
    num_cases: int = -1
    max_plot_id: int = 200
    output_dir: str = "results/{run_name}/causal_trace"

    def __post_init__(self):
        self.noise = parse_noise(self.noise)
        self.kinds = tuple(self.kinds)
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        # A window under 2 restores no layers: every severed cell falls back to
        # the corrupted-run probability and the MLP/attention maps come out flat
        # at low_score while still looking like complete runs on disk.
        if self.window < 2:
            raise ValueError(
                f"window must be at least 2 (got {self.window}); a narrower "
                "window restores no layers, so the mlp/attn traces are degenerate"
            )
        unknown = set(self.kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown trace kinds {sorted(unknown)}; use {KINDS}")
        if self.torch_dtype not in DTYPES:
            raise ValueError(f"torch_dtype must be one of {DTYPES}")

    @classmethod
    def gptj(cls, **overrides) -> "TraceConfig":
        """GPT-J-6B as in the paper: noise 3 sigma_e = 0.12922, window 10."""
        settings = dict(model_name="EleutherAI/gpt-j-6B", noise=GPTJ_NOISE,
                        window=GPTJ_WINDOW)
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def for_model(cls, model_name: str, num_layers: int, **overrides) -> "TraceConfig":
        """Cross-model protocol: per-model 3 sigma noise, depth-scaled window."""
        settings = dict(model_name=model_name, noise="s3",
                        window=window_for(num_layers), torch_dtype="float32")
        settings.update(overrides)
        return cls(**settings)

    @property
    def run_name(self) -> str:
        """Results folder name, e.g. ``ns3_r0_gpt2-large``."""
        return f"n{self.noise}_r{int(self.replace)}_{self.model_name.replace('/', '_')}"

    @property
    def run_dir(self) -> str:
        return self.output_dir.format(run_name=self.run_name)


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
