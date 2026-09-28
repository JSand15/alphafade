"""Exception and warning classes.

Every error alphafade raises is an :class:`AlphaFadeError`. Input problems are also
:class:`ValueError`, so ``except ValueError`` keeps working for users who don't care about
the finer categories.
"""

from __future__ import annotations

__all__ = [
    "AlignmentError",
    "AlphaFadeError",
    "AlphaFadeWarning",
    "DataDroppedWarning",
    "DownloadError",
    "FitWarning",
    "FrequencyError",
    "InputError",
    "InsufficientDataError",
]


class AlphaFadeError(Exception):
    """Base class for every error raised by alphafade."""


class InputError(AlphaFadeError, ValueError):
    """An input has the wrong type, shape, or values. The message says how to fix it."""


class AlignmentError(InputError):
    """Two inputs can't be lined up by date and/or asset."""


class FrequencyError(InputError):
    """The data frequency is ambiguous, mixed, or inconsistent between inputs."""


class InsufficientDataError(InputError):
    """There are too few usable observations to compute the requested statistic."""


class DownloadError(AlphaFadeError, OSError):
    """A dataset couldn't be downloaded or saved to the local cache."""


class AlphaFadeWarning(UserWarning):
    """Base class for every warning emitted by alphafade."""


class DataDroppedWarning(AlphaFadeWarning):
    """Some observations were dropped or couldn't be used (the message gives counts)."""


class FitWarning(AlphaFadeWarning):
    """A model fit was unreliable, hit a bound, or fell back to a simpler model."""
