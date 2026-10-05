"""Project when a decaying edge falls to a level you care about."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from ._errors import InputError
from ._validate import DAYS_PER_YEAR
from .decay import DecayFit

__all__ = ["LifetimeResult", "signal_lifetime"]


@dataclass(frozen=True)
class LifetimeResult:
    """Result of :func:`signal_lifetime`.

    Attributes
    ----------
    model : {"exponential", "linear"}
        Which fitted curve the projection uses (taken from the :class:`DecayFit`).
    floor : float or None
        The edge level being waited for, in the same units as the fitted series. When the
        request was a ``fraction``, this is ``fraction * initial_level``. None when it can't
        be defined.
    fraction : float or None
        The share of the starting level requested (None if an absolute ``floor`` was given).
    years_from_start : float or None
        Years after the first date at which the fitted edge reaches ``floor``. 0.0 if it
        started at or below the floor. None when no honest answer exists (see ``reason``).
    years_remaining : float or None
        Years from the last observation until then; 0.0 if it has already been reached,
        never negative. None if ``from_end=False`` or there is no answer.
    date : Timestamp or None
        Projected calendar date. None when there is no answer or it lies beyond the range
        pandas can represent.
    ci_low, ci_high : float or None
        Confidence interval for ``years_from_start``. Faster decay means an earlier date, so
        ``ci_low`` comes from the fastest plausible decay rate. See :func:`signal_lifetime`
        for the simplification behind it.
    reason : str or None
        Plain-English explanation whenever the answer is missing or special (no decay
        detected, already below the floor, never reached). None for an ordinary answer.
    alpha : float
        1 - confidence level, inherited from the fit.
    start, end : Timestamp
        First and last dates of the fitted sample.
    """

    model: str
    floor: float | None
    fraction: float | None
    years_from_start: float | None
    years_remaining: float | None
    date: pd.Timestamp | None
    ci_low: float | None
    ci_high: float | None
    reason: str | None
    alpha: float
    start: pd.Timestamp
    end: pd.Timestamp

    def summary(self) -> str:
        """Plain-English description of the projection."""
        target = (
            f"{self.fraction:.0%} of its starting level"
            if self.fraction is not None
            else f"{self.floor:.4g}"
            if self.floor is not None
            else "the requested level"
        )
        if self.years_from_start is None:
            return f"No answer for when the edge reaches {target}: {self.reason}"
        if self.years_from_start == 0.0:
            return f"The edge was already at or below {target} at the start of the sample."
        conf = f"{1 - self.alpha:.0%}"
        lo = "?" if self.ci_low is None else f"{self.ci_low:.1f}"
        hi = (
            "infinity"
            if self.ci_high is None or math.isinf(self.ci_high)
            else f"{self.ci_high:.1f}"
        )
        text = (
            f"The fitted {self.model} edge reaches {target} about "
            f"{self.years_from_start:.1f} years after the sample start ({conf} CI {lo} to {hi})"
        )
        if self.date is not None:
            text += f", around {self.date:%Y-%m}"
        else:
            text += ", a date beyond any calendar pandas can represent"
        text += "."
        if self.years_remaining is not None:
            text += (
                " That has already happened."
                if self.years_remaining == 0.0
                else f" That is {self.years_remaining:.1f} years after the last observation."
            )
        text += " The interval holds the starting level fixed and only varies the decay rate."
        return text


def _check_level(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise InputError(f"{name} must be a number, got {value!r}.")
    out = float(value)
    if not math.isfinite(out):
        raise InputError(f"{name} must be finite, got {value!r}.")
    return out


def signal_lifetime(
    fit: DecayFit,
    *,
    floor: float | None = None,
    fraction: float | None = None,
    from_end: bool = True,
) -> LifetimeResult:
    """Project when a decaying edge falls to a level you care about.

    Uses the curve behind a :class:`DecayFit`. Exponential: edge(t) = a * exp(-lambda * t),
    so it reaches ``floor`` at t = ln(a / floor) / lambda. Linear: edge(t) = b0 * (1 - r * t)
    where b0 is the starting level and r the fit's ``decay_rate`` (share of the starting
    level lost per year), so it reaches ``floor`` at t = (1 - floor / b0) / r. Here t is
    years since the first date. ``fraction=0.5`` gives exactly ``fit.half_life_years``.

    The confidence interval pushes the fit's bootstrap interval for the decay rate through the
    same formula. Simplification: the starting level ``a`` (or ``b0``) is held at its fitted
    value, so the interval reflects uncertainty in the speed of decay only, not in where the
    edge began. It is therefore a little narrower than a fully joint interval would be.

    Parameters
    ----------
    fit : DecayFit
        Output of :func:`fit_decay`.
    floor : float, optional
        Absolute edge level to wait for, in the fitted series' units (e.g. an IC of 0.02).
    fraction : float, optional
        Share of the starting level, strictly between 0 and 1. Give exactly one of ``floor``
        and ``fraction``.
    from_end : bool, default True
        Also report ``years_remaining`` measured from the last observation.

    Returns
    -------
    LifetimeResult
        "No answer" is a valid result, not an error: when no decay was detected, the starting
        level isn't positive, or an exponential edge is asked to reach zero or below, the
        years are None and ``reason`` says why. If the floor is at or above the starting level
        the answer is 0.0 years.

    Raises
    ------
    InputError
        If ``fit`` isn't a DecayFit, or not exactly one valid ``floor``/``fraction`` is given.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from alphafade import fit_decay
    >>> dates = pd.date_range("1970-01-31", periods=480, freq="ME")
    >>> t = np.arange(480) / 12
    >>> noise = np.random.default_rng(0).normal(0, 0.02, 480)
    >>> fit = fit_decay(pd.Series(0.10 * np.exp(-t / 5) + noise, index=dates), n_boot=200, rng=0)
    >>> life = signal_lifetime(fit, fraction=0.5)
    >>> life.years_from_start == fit.half_life_years
    True
    >>> round(signal_lifetime(fit, floor=0.025).years_from_start, 1)
    7.1
    """
    if not isinstance(fit, DecayFit):
        raise InputError(f"fit must be a DecayFit from fit_decay(), got {type(fit).__name__}.")
    if (floor is None) == (fraction is None):
        raise InputError("Give exactly one of floor= (an absolute level) or fraction= (0 to 1).")
    floor_v: float | None = None
    frac_v: float | None = None
    if floor is not None:
        floor_v = _check_level("floor", floor)
    else:
        frac_v = _check_level("fraction", fraction)
        if not 0 < frac_v < 1:
            raise InputError(f"fraction must be strictly between 0 and 1, got {fraction!r}.")

    def result(
        years: float | None,
        reason: str | None,
        lo: float | None = None,
        hi: float | None = None,
        level: float | None = floor_v,
    ) -> LifetimeResult:
        remaining: float | None = None
        when: pd.Timestamp | None = None
        if years is not None:
            span = (fit.end - fit.start).total_seconds() / (86_400 * DAYS_PER_YEAR)
            if from_end:
                remaining = max(0.0, years - span)
            try:
                when = fit.start + pd.Timedelta(days=years * DAYS_PER_YEAR)
            except (OverflowError, ValueError, OutOfBoundsDatetime):
                when = None
        return LifetimeResult(
            model=fit.model,
            floor=level,
            fraction=frac_v,
            years_from_start=years,
            years_remaining=remaining,
            date=when,
            ci_low=lo,
            ci_high=hi,
            reason=reason,
            alpha=fit.alpha,
            start=fit.start,
            end=fit.end,
        )

    a = fit.initial_level
    if not fit.decay_detected:
        return result(
            None,
            "no detectable decay in the fit (its confidence interval for the decay rate "
            "includes zero), so any projected date would be invented.",
            level=floor_v if floor_v is not None else a * frac_v if frac_v is not None else None,
        )
    if not a > 0:
        return result(
            None,
            f"the fitted starting level ({a:.4g}) is not positive, so a fall to a floor "
            "or a share of it isn't well defined.",
        )
    level = floor_v if floor_v is not None else a * frac_v  # type: ignore[operator]
    if level >= a:
        return result(
            0.0,
            f"the edge was already at or below {level:.4g} at the start "
            f"(fitted starting level {a:.4g}).",
            0.0,
            0.0,
            level,
        )
    if fit.model == "exponential" and level <= 0:
        return result(
            None,
            "an exponential edge shrinks toward zero but never reaches it (or a negative "
            "level), so it is never reached.",
            level=level,
        )

    def years_at(rate: float) -> float:
        if rate <= 0:
            return math.inf
        if fit.model == "exponential":
            return math.log(a / level) / rate
        return (1.0 - level / a) / rate

    rate_lo, rate_hi = fit.rate_ci
    return result(
        years_at(fit.decay_rate),
        None,
        years_at(rate_hi),  # faster decay -> earlier
        years_at(rate_lo),
        level,
    )
