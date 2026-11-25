"""Test whether a fading edge goes all the way to zero or levels off at a floor."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._errors import InputError, InsufficientDataError
from ._stats import FloatArray, RngLike, default_block_size, resolve_rng, wild_block_signs
from ._validate import as_series, dropna_series, years_since_start
from .decay import _aic, _fit_exponential, _lambda_grid
from .report import _obj_to_dict
from .rolling import WINDOW_ATTR

__all__ = ["FloorFit", "fit_floor_decay"]

FloorOutcome = Literal["survives", "may_reach_zero", "reverses", "unknown"]

_MIN_OBS = 36
_GRID = 240
_GOLDEN_ITERS = 30
_BOOT_CHUNK = 64


@dataclass(frozen=True)
class FloorFit:
    """Result of :func:`fit_floor_decay`.

    The floor-related numbers (``floor``, ``floor_ci``, ``share_remaining``,
    ``half_life_years``, ``ci_low``, ``ci_high``) are None unless ``decay_detected`` is True.

    Attributes
    ----------
    change_detected : bool
        A fading or rising curve explains the data better than a constant average, beyond
        what noise alone would produce (``p_value <= alpha``).
    floor_identified : bool
        The curve visibly levels off inside the sample: the best rate is not at the edge of
        the search range, and (nearly) none of the bootstrap refits needed a half-life as
        long as the whole sample.
    decay_detected : bool
        Change detected, floor identified, and the fitted floor is closer to zero than the
        fitted starting level (or the same distance past it, on the other side).
    floor_outcome : {"survives", "may_reach_zero", "reverses", "unknown"}
        "survives": the floor's interval is entirely on the starting level's side of zero.
        "may_reach_zero": the interval includes zero. "reverses": it is entirely on the
        other side. "unknown": no decay detected.
    p_value : float
        Bootstrap p-value for "the average never changed".
    start_level : float or None
        Fitted edge at the first date.
    floor : float or None
        Level the fitted curve is heading towards.
    floor_ci : tuple of float or None
        Bootstrap percentile interval for ``floor``.
    share_remaining : float or None
        ``floor / start_level``: the share of the starting edge the curve ends at. Negative
        if the edge reverses.
    half_life_years : float or None
        Years to close half of the gap between the starting level and the floor.
    ci_low, ci_high : float or None
        Confidence interval for ``half_life_years``.
    decay_rate : float
        Fitted rate per year at which the gap to the floor shrinks.
    r2 : float
        R-squared of the floor model.
    aic_floor, aic_exponential, aic_constant : float
        Akaike information criterion (lower is better) for the floor model (3 parameters),
        the fade-to-zero exponential of :func:`fit_decay` (2) and a constant average (1).
    n_obs : int
    start, end : Timestamp
    block_size : int
        Block length (in observations) used by the bootstrap.
    alpha : float
        1 - confidence level.
    fitted : Series
        The fitted floor curve.
    notes : tuple of str
        Caveats worth reading.
    """

    change_detected: bool
    floor_identified: bool
    decay_detected: bool
    floor_outcome: FloorOutcome
    p_value: float
    start_level: float | None
    floor: float | None
    floor_ci: tuple[float, float] | None
    share_remaining: float | None
    half_life_years: float | None
    ci_low: float | None
    ci_high: float | None
    decay_rate: float
    r2: float
    aic_floor: float
    aic_exponential: float
    aic_constant: float
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    block_size: int
    alpha: float
    fitted: pd.Series[float] = field(repr=False)
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        """Plain-English description of the fit."""
        conf = f"{1 - self.alpha:.0%}"
        lines = [
            f"Floor decay fit on {self.n_obs} observations "
            f"({self.start:%Y-%m} to {self.end:%Y-%m})."
        ]
        if self.decay_detected:
            assert self.floor is not None
            assert self.floor_ci is not None
            assert self.share_remaining is not None
            assert self.start_level is not None
            assert self.half_life_years is not None
            assert self.ci_low is not None
            assert self.ci_high is not None
            lines.append(
                f"The edge started at {self.start_level:.4g} and is fading toward a floor of "
                f"{self.floor:.4g} ({conf} CI {self.floor_ci[0]:.4g} to "
                f"{self.floor_ci[1]:.4g}), which is {self.share_remaining:.0%} of where it "
                f"started. The gap to the floor halves every {self.half_life_years:.1f} years "
                f"({conf} CI {self.ci_low:.1f} to {self.ci_high:.1f})."
            )
            lines.append(
                {
                    "survives": "The whole floor interval is on the starting side of zero, so "
                    "on this evidence some of the edge survives.",
                    "may_reach_zero": "The floor interval includes zero, so the data can't "
                    "rule out the edge fading all the way to zero.",
                    "reverses": "The whole floor interval is on the opposite sign from the "
                    "start: the edge looks to be heading past zero.",
                    "unknown": "",
                }[self.floor_outcome]
            )
            lines.append(
                "A floor describes where the fitted curve is heading if it keeps this shape. "
                "It is not a promise about the future."
            )
        elif not self.change_detected:
            lines.append(
                "No detectable change: a constant average explains the data about as well as "
                f"a fading curve (p = {self.p_value:.2f}), so there is no floor to estimate."
            )
        elif not self.floor_identified:
            lines.append(
                f"The average changed over time (p = {self.p_value:.3f}), but the data can't "
                "pin down a floor: the curve hasn't visibly levelled off, or the change looks "
                "like a sudden step. Try fit_decay (a steady fade) or find_break (a step)."
            )
        else:
            first, last = float(self.fitted.iloc[0]), float(self.fitted.iloc[-1])
            lines.append(
                f"The average changed (p = {self.p_value:.3f}), but the edge isn't shrinking "
                f"toward zero: the fitted curve goes from {first:.4g} to {last:.4g}."
            )
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self, *, include_series: bool = False) -> dict[str, Any]:
        """Results as a JSON-safe dictionary (dates as ISO strings, NaN and inf as None).

        Parameters
        ----------
        include_series : bool, default False
            Also include the ``fitted`` curve.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.date_range("2000-01-31", periods=120, freq="ME")
        >>> r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 120), index=idx)
        >>> fit_floor_decay(r, n_boot=100, rng=0).to_dict()["n_obs"]
        120
        """
        out = _obj_to_dict(self, include_series)
        assert out is not None
        return out


def fit_floor_decay(
    perf: pd.Series[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    block_size: int | None = None,
    rng: RngLike = None,
) -> FloorFit:
    """Fit a fade that levels off at a floor, to see if the edge dies or just gets smaller.

    Fits p(t) = floor + (start - floor) * exp(-lambda * t), where t is years since the first
    date. :func:`fit_decay` assumes the floor is zero; this estimates it. Three questions are
    answered in order, and a floor is reported only if the first two pass:

    1. *Did anything change?* The floor curve is compared with a constant average, using a
       wild block bootstrap under "nothing changed" for the p-value.
    2. *Is the floor pinned down?* The curve must visibly level off within the sample.
    3. *Where is the floor?* A bootstrap interval says whether it is above zero, could be
       zero, or has crossed to the other side.

    A third parameter needs much more data than a plain half-life: "can't tell" is a common
    and honest answer on noisy monthly returns. A per-date IC series usually does better.

    Parameters
    ----------
    perf : Series
        Performance over time, indexed by date: a raw per-period series such as
        :func:`ic_series` output or periodic strategy returns.
    n_boot : int, default 1000
        Bootstrap replications (at least 100).
    alpha : float, default 0.05
        Significance level; intervals cover 1 - alpha.
    block_size : int, optional
        Bootstrap block length in observations. Default: about 1.75 * n^(1/3), and never
        shorter than the rolling window when ``perf`` is a rolling series.
    rng : int, numpy Generator, or None
        Seed for reproducible results. Pass an int.

    Returns
    -------
    FloorFit

    Warns
    -----
    DataDroppedWarning
        If ``perf`` has missing values in the middle.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> dates = pd.date_range("1970-01-31", periods=480, freq="ME")
    >>> t = np.arange(480) / 12
    >>> noise = np.random.default_rng(0).normal(0, 0.01, 480)
    >>> ic = pd.Series(0.04 + 0.06 * np.exp(-t / 5) + noise, index=dates)
    >>> fit = fit_floor_decay(ic, n_boot=200, rng=0)
    >>> fit.floor_outcome, round(fit.floor, 2)
    ('survives', 0.04)
    """
    series = as_series(perf, "perf")
    window = series.attrs.get(WINDOW_ATTR)
    y_s = dropna_series(series, "perf")
    n = len(y_s)
    if n < _MIN_OBS:
        raise InsufficientDataError(
            f"fit_floor_decay needs at least {_MIN_OBS} non-missing observations, got {n}."
        )
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise InputError(f"n_boot must be an integer of at least 100, got {n_boot!r}.")
    if not 0 < alpha < 0.5:
        raise InputError(f"alpha must be between 0 and 0.5, got {alpha!r}.")
    if alpha * (n_boot + 1) < 1:
        raise InputError(
            f"alpha={alpha} is too small for n_boot={n_boot}: the smallest possible p-value "
            f"is {1 / (n_boot + 1):.4f}. Raise n_boot to at least {math.ceil(1 / alpha)}."
        )
    y: FloatArray = np.asarray(y_s.to_numpy(), dtype=np.float64)
    if np.ptp(y) == 0:
        raise InputError("perf is constant, so there is no decay (or anything else) to fit.")
    t = years_since_start(pd.DatetimeIndex(y_s.index))

    notes: list[str] = []
    min_block = 1
    if isinstance(window, (int, np.integer)) and window > 1:
        min_block = int(window)
        notes.append(
            f"perf is a rolling series (window={window}). Neighbouring values overlap, so "
            "R-squared is inflated; bootstrap blocks were widened to the window length. "
            "For cleaner inference fit the raw per-period series instead."
        )
    if block_size is None:
        block = default_block_size(n, min_block=min_block)
    else:
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
            raise InputError(f"block_size must be a positive integer, got {block_size!r}.")
        if not 1 <= block_size <= n:
            raise InputError(f"block_size must be between 1 and {n}, got {block_size}.")
        block = int(block_size)
    if n / block < 10:
        notes.append(
            f"Only about {n // block} independent blocks of data; the intervals are rough. "
            "More history would sharpen them."
        )

    mean = float(y.mean())
    sst = float(((y - mean) ** 2).sum())
    if not sst > 0:
        raise InputError(
            "perf varies too little to measure (its squared deviations underflow to zero). "
            "Check the units, or multiply the series by a constant."
        )
    search = _Search(t)
    best_fit = _fit_floor(search, y[None, :], precise=True)
    lam, floor, gap, sse = (float(v[0]) for v in best_fit)
    fitted = floor + gap * np.exp(-lam * t)
    rng_g = resolve_rng(rng)

    # 1. Did anything change? Resample under "the average is constant".
    stat = _improvement(sst, _fit_floor(search, y[None, :])[3])[0]
    null_stats = np.empty(n_boot)
    for first, ys in _wild_samples(np.full(n, mean), y - mean, block, n_boot, rng_g):
        null_sst = ((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        null_stats[first : first + len(ys)] = _improvement(null_sst, _fit_floor(search, ys)[3])
    p_value = float((1 + np.sum(null_stats >= stat)) / (n_boot + 1))
    change = p_value <= alpha

    # 2 and 3. Resample around the fitted floor curve.
    rates = np.empty(n_boot)
    floors = np.empty(n_boot)
    for first, ys in _wild_samples(fitted, y - fitted, block, n_boot, rng_g):
        boot_lam, boot_floor, _, _ = _fit_floor(search, ys)
        rates[first : first + len(ys)] = boot_lam
        floors[first : first + len(ys)] = boot_floor
    tail = [alpha / 2, 1 - alpha / 2]
    rate_lo, rate_hi = (float(q) for q in np.quantile(rates, tail))
    floor_lo, floor_hi = (float(q) for q in np.quantile(floors, tail))
    slowest, fastest = float(search.grid[0]), float(search.grid[-1])
    identified = slowest < lam < fastest and rate_lo > slowest * (1 + 1e-6)

    start_level = floor + gap
    detected = bool(change and identified and abs(floor) <= abs(start_level))

    outcome: FloorOutcome = "unknown"
    if detected:
        side = math.copysign(1.0, start_level)
        low, high = sorted((floor_lo * side, floor_hi * side))
        outcome = "survives" if low > 0 else "reverses" if high < 0 else "may_reach_zero"

    zero_grid = _lambda_grid(t)
    sse_zero = float(_fit_exponential(t, y[None, :], zero_grid)[2][0])

    return FloorFit(
        change_detected=bool(change),
        floor_identified=bool(identified),
        decay_detected=detected,
        floor_outcome=outcome,
        p_value=p_value,
        start_level=start_level if detected else None,
        floor=floor if detected else None,
        floor_ci=(floor_lo, floor_hi) if detected else None,
        share_remaining=floor / start_level if detected else None,
        half_life_years=math.log(2) / lam if detected else None,
        ci_low=math.log(2) / rate_hi if detected else None,
        ci_high=math.log(2) / rate_lo if detected else None,
        decay_rate=lam,
        r2=1 - sse / sst,
        aic_floor=_aic(sse, n, k=3),
        aic_exponential=_aic(sse_zero, n),
        aic_constant=_aic(sst, n, k=1),
        n_obs=n,
        start=pd.Timestamp(y_s.index[0]),
        end=pd.Timestamp(y_s.index[-1]),
        block_size=block,
        alpha=float(alpha),
        fitted=pd.Series(fitted, index=y_s.index, name="floor_fit"),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# Model fitting internals
# --------------------------------------------------------------------------------------
def _floor_grid(t: FloatArray) -> FloatArray:
    """Candidate rates (per year), log-spaced.

    Slowest: a half-life as long as the whole sample (any slower and the curve can't be seen
    to level off). Fastest: a half-life of 1% of the span, or 3 observations if longer.
    """
    span = float(t[-1])
    min_half_life = max(span / 100.0, 3.0 * float(np.median(np.diff(t))))
    return np.geomspace(math.log(2) / span, math.log(2) / min_half_life, _GRID)


def _profile_floor(
    t: FloatArray, y: FloatArray, lam: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Best floor, gap and SSE for each row of ``y`` at its own rate in ``lam``.

    For a fixed rate the model is a straight-line regression on exp(-rate * t), so floor and
    gap have closed forms and the whole fit is a 1-D search that can't diverge.
    """
    e = np.exp(-lam[:, None] * t[None, :])
    e_mean = e.mean(axis=1)
    ec = e - e_mean[:, None]
    y_mean = y.mean(axis=1)
    yc = y - y_mean[:, None]
    see = np.einsum("ij,ij->i", ec, ec)
    sey = np.einsum("ij,ij->i", ec, yc)
    gap = sey / see
    floor = y_mean - gap * e_mean
    sse = np.einsum("ij,ij->i", yc, yc) - gap * sey
    return floor, gap, sse


class _Search:
    """Candidate rates and their centred exponential curves, computed once per fit."""

    def __init__(self, t: FloatArray) -> None:
        self.t = t
        self.grid = _floor_grid(t)
        e = np.exp(-np.outer(self.grid, t))  # (G, n)
        self.curves: FloatArray = e - e.mean(axis=1, keepdims=True)
        self.power: FloatArray = np.einsum("ij,ij->i", self.curves, self.curves)


def _fit_floor(
    search: _Search, y: FloatArray, *, precise: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit every row of ``y``; returns (rate, floor, gap, SSE).

    A grid search finds the best candidate rate, then it is refined between its neighbours:
    by golden-section search when ``precise`` (the reported fit), otherwise by one parabola
    through the three nearest grid points (bootstrap refits, where thousands are needed and
    the difference is far below the sampling noise).
    """
    t, grid = search.t, search.grid
    yc = y - y.mean(axis=1, keepdims=True)  # (B, n)
    sse_grid = (
        np.einsum("ij,ij->i", yc, yc)[:, None]
        - (yc @ search.curves.T) ** 2 / search.power[None, :]
    )
    best = np.argmin(sse_grid, axis=1)
    rows = np.arange(len(best))
    mid = np.clip(best, 1, len(grid) - 2)
    if precise:
        lo, hi = grid[mid - 1], grid[mid + 1]
        inv_phi = (math.sqrt(5) - 1) / 2
        for _ in range(_GOLDEN_ITERS):
            c = hi - inv_phi * (hi - lo)
            d = lo + inv_phi * (hi - lo)
            left = _profile_floor(t, y, c)[2] < _profile_floor(t, y, d)[2]
            hi = np.where(left, d, hi)
            lo = np.where(left, lo, c)
        lam = (lo + hi) / 2
    else:
        before, here, after = (sse_grid[rows, mid + k] for k in (-1, 0, 1))
        bend = before - 2 * here + after
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(bend > 0, 0.5 * (before - after) / bend, 0.0)
        # The grid is evenly spaced in log(rate), so step by a power of the spacing ratio.
        lam = grid[mid] * (grid[1] / grid[0]) ** np.clip(shift, -1.0, 1.0)
    # Keep the exact grid endpoint when the optimum is at the boundary, so callers can
    # detect "hit the bound". Also never end up worse than the best grid point.
    at_edge = (best == 0) | (best == len(grid) - 1)
    worse = _profile_floor(t, y, lam)[2] > sse_grid[rows, best]
    lam = np.where(at_edge | worse, grid[best], lam)
    floor, gap, sse = _profile_floor(t, y, lam)
    return lam, floor, gap, sse


def _improvement(sst: float | FloatArray, sse: FloatArray) -> FloatArray:
    """How much better the curve fits than a constant: (SST - SSE) / SSE."""
    out: FloatArray = (sst - sse) / np.maximum(sse, 1e-12 * np.maximum(sst, 1e-300))
    return out


def _wild_samples(
    centre: FloatArray,
    resid: FloatArray,
    block: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, FloatArray]]:
    """Yield (first row number, chunk of bootstrap samples) pairs, to bound memory use."""
    for first in range(0, n_boot, _BOOT_CHUNK):
        size = min(_BOOT_CHUNK, n_boot - first)
        yield (
            first,
            centre[None, :] + resid[None, :] * wild_block_signs(len(centre), block, size, rng),
        )
