"""alphafade measures whether a trading signal's edge is dying, how fast, and why.

Measure how a signal's edge shrinks across calendar time (not forecast horizon), fit a
half-life, test for structural breaks and publication effects, and check whether the fade
lines up with crowding.
"""

from __future__ import annotations

from ._errors import (
    AlignmentError,
    AlphaFadeError,
    AlphaFadeWarning,
    DataDroppedWarning,
    FitWarning,
    FrequencyError,
    InputError,
    InsufficientDataError,
)
from .rolling import forward_returns, ic_series, rolling_ic, rolling_sharpe

__version__ = "0.1.0"

__all__ = [
    "AlignmentError",
    "AlphaFadeError",
    "AlphaFadeWarning",
    "DataDroppedWarning",
    "FitWarning",
    "FrequencyError",
    "InputError",
    "InsufficientDataError",
    "__version__",
    "forward_returns",
    "ic_series",
    "rolling_ic",
    "rolling_sharpe",
]
