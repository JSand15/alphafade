"""alphafade measures whether a trading signal's edge is dying, how fast, and why.

Measure how a signal's edge shrinks across calendar time (not forecast horizon), fit a
half-life, test for structural breaks and publication effects, and check whether the fade
lines up with crowding.
"""

from __future__ import annotations

from . import datasets
from ._errors import (
    AlignmentError,
    AlphaFadeError,
    AlphaFadeWarning,
    DataDroppedWarning,
    DownloadError,
    FitWarning,
    FrequencyError,
    InputError,
    InsufficientDataError,
)
from .breaks import BreakResult, chow_test, find_break
from .crowding import crowding_score
from .decay import DecayFit, fit_decay
from .publication import GapResult, publication_gap
from .report import CrowdingLink, FadeReport, analyze
from .rolling import forward_returns, ic_series, rolling_ic, rolling_sharpe

__version__ = "0.1.0"

__all__ = [
    "AlignmentError",
    "AlphaFadeError",
    "AlphaFadeWarning",
    "BreakResult",
    "CrowdingLink",
    "DataDroppedWarning",
    "DecayFit",
    "DownloadError",
    "FadeReport",
    "FitWarning",
    "FrequencyError",
    "GapResult",
    "InputError",
    "InsufficientDataError",
    "__version__",
    "analyze",
    "chow_test",
    "crowding_score",
    "datasets",
    "find_break",
    "fit_decay",
    "forward_returns",
    "ic_series",
    "publication_gap",
    "rolling_ic",
    "rolling_sharpe",
]
