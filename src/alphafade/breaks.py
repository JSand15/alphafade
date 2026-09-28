"""Structural-break tests for a change in a series' average at a known or unknown date."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from ._errors import InputError, InsufficientDataError
from ._stats import FloatArray, ols_hac, resolve_hac_lags
from ._supwald_table import LEVELS, QUANTILES
from ._validate import as_series, as_timestamp, dropna_series, to_array
from .rolling import WINDOW_ATTR

__all__ = ["BreakResult", "chow_test", "find_break"]

BreakMethod = Literal["sup_wald", "cusum", "chow"]

_MIN_OBS = 20
_MIN_SIDE = 5
_P_FLOOR = 1 - LEVELS[-1]


@dataclass(frozen=True)
class BreakResult:
    """Result of :func:`find_break` or :func:`chow_test`.

    Attributes
    ----------
    date : Timestamp
        First date of the new regime (the estimated break for ``find_break``).
    stat : float
        Test statistic: sup-Wald, CUSUM, or the Chow-style Wald statistic.
    p_value : float
        Probability of a statistic this large if there were no break. For sup-Wald it comes
        from a simulated table and is floored at 0.0005.
    mean_before, mean_after : float
        Average of the series before and from ``date`` on.
    change : float
        ``mean_after - mean_before``.
    change_t : float
        Newey-West t-statistic of the change.
    method : {"sup_wald", "cusum", "chow"}
    n_before, n_after : int
    hac_lags : int
    trim : float or None
        Share of the sample excluded at each end when searching (unknown-date tests).
    path : Series or None
        The statistic at every candidate date (for plotting), when available.
    """

    date: pd.Timestamp
    stat: float
    p_value: float
    mean_before: float
    mean_after: float
    change: float
    change_t: float
    method: BreakMethod
    n_before: int
    n_after: int
    hac_lags: int
    trim: float | None = None
    path: pd.Series[float] | None = field(default=None, repr=False)

    @property
    def n_obs(self) -> int:
        """Total observations used."""
        return self.n_before + self.n_after

    def significant(self, alpha: float = 0.05) -> bool:
        """Whether the break is significant at level ``alpha``."""
        return self.p_value < alpha

    def summary(self) -> str:
        """Plain-English description of the result."""
        names = {
            "sup_wald": "sup-Wald search for an unknown break date",
            "cusum": "CUSUM test for parameter stability",
            "chow": "Chow-style test at a chosen date",
        }
        p_text = f"p < {_P_FLOOR:.4f}" if self.p_value <= _P_FLOOR else f"p = {self.p_value:.3f}"
        verdict = "a statistically significant" if self.significant() else "no significant"
        where = "most likely break" if self.method != "chow" else "tested break"
        direction = "fell" if self.change < 0 else "rose"
        return (
            f"{names[self.method]} ({self.n_obs} observations): {verdict} change in the "
            f"average ({p_text}). The {where} is {self.date:%Y-%m-%d}: the average {direction} "
            f"from {self.mean_before:.4g} to {self.mean_after:.4g} "
            f"(Newey-West t = {self.change_t:.2f})."
        )


def chow_test(
    returns: pd.Series[float],
    date: str | pd.Timestamp,
    *,
    hac_lags: int | None = None,
) -> BreakResult:
    """Test whether the average of a series changed at a known date.

    Regresses the series on a constant and a dummy that is 1 from ``date`` on, and tests the
    dummy with a Newey-West (autocorrelation-robust) Wald statistic, which is chi-squared with
    1 degree of freedom when there's no change. Use it when the date comes from outside the
    data (a publication date, a regulation, a fund launch). If you picked the date by looking
    at the data, use :func:`find_break` instead; its p-value accounts for the search.

    Parameters
    ----------
    returns : Series
        Per-period performance (strategy returns or a per-date IC), indexed by date.
    date : str or Timestamp
        First date of the "after" period.
    hac_lags : int, optional
        Newey-West lags. Default: floor(4 * (n/100)^(2/9)), and at least window - 1 for a
        rolling series.

    Returns
    -------
    BreakResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> r = pd.Series(np.r_[rng.normal(0.01, 0.02, 120), rng.normal(0.0, 0.02, 120)],
    ...               index=pd.date_range("1990-01-31", periods=240, freq="ME"))
    >>> res = chow_test(r, "2000-01-31")
    >>> res.p_value < 0.01, round(res.change, 3)
    (True, -0.01)
    """
    y_s, lags = _prepare(returns, hac_lags)
    when = as_timestamp(date, "date")
    k = int(np.searchsorted(y_s.index.to_numpy(), np.datetime64(when), side="left"))
    n = len(y_s)
    if k < _MIN_SIDE or n - k < _MIN_SIDE:
        raise InsufficientDataError(
            f"chow_test needs at least {_MIN_SIDE} observations on each side of {when:%Y-%m-%d};"
            f" got {k} before and {n - k} after."
        )
    y = to_array(y_s)
    change, t = _dummy_regression(y, k, lags)
    wald = t * t
    return BreakResult(
        date=pd.Timestamp(y_s.index[k]),
        stat=wald,
        p_value=float(sps.chi2.sf(wald, df=1)),
        mean_before=float(y[:k].mean()),
        mean_after=float(y[k:].mean()),
        change=change,
        change_t=t,
        method="chow",
        n_before=k,
        n_after=n - k,
        hac_lags=lags,
    )


def find_break(
    returns: pd.Series[float],
    *,
    method: Literal["sup_wald", "cusum"] = "sup_wald",
    trim: float = 0.15,
    hac_lags: int | None = None,
) -> BreakResult:
    """Search for the date at which the average of a series most likely changed.

    ``method="sup_wald"`` (default) computes a Wald statistic for a mean shift at every
    candidate date in the middle ``1 - 2 * trim`` of the sample and reports the largest
    (Andrews 1993). Because the date was searched for, the p-value comes from the
    distribution of that maximum, not from chi-squared. The noise level (a Newey-West
    long-run variance) is estimated once from the whole demeaned series. In simulations this
    keeps false alarms near the nominal 5% in samples of a few hundred observations, where
    re-estimating it around each candidate date rejected about twice as often. The reported
    date is then the least-squares break date, and ``change_t`` is the Newey-West t-statistic
    of the shift at that date.

    ``method="cusum"`` tracks the cumulative sum of deviations from the overall mean
    (Ploberger & Kramer 1992), scaled by a Newey-West long-run standard deviation. A large
    excursion means the average drifted. The reported date is where the cumulative sum peaks.

    Parameters
    ----------
    returns : Series
        Per-period performance (strategy returns or per-date IC), indexed by date. Prefer raw
        per-period data over rolling averages.
    method : {"sup_wald", "cusum"}, default "sup_wald"
    trim : float, default 0.15
        sup-Wald only: share of the sample excluded at each end, since a break can't be
        estimated from a handful of observations. One of 0.05, 0.10, 0.15, 0.20, 0.25.
    hac_lags : int, optional
        Newey-West lags; see :func:`chow_test`.

    Returns
    -------
    BreakResult

    Notes
    -----
    Both tests look for one shift in the mean. A gradual decline shows up as a break
    somewhere in the middle of the decline; use :func:`fit_decay` to describe a smooth fade.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> r = pd.Series(np.r_[rng.normal(0.01, 0.02, 150), rng.normal(0.0, 0.02, 150)],
    ...               index=pd.date_range("1980-01-31", periods=300, freq="ME"))
    >>> res = find_break(r)
    >>> res.p_value < 0.01, abs((res.date - r.index[150]).days) < 365
    (True, True)
    """
    y_s, lags = _prepare(returns, hac_lags)
    y = to_array(y_s)
    n = len(y)
    if method == "cusum":
        return _cusum(y_s, y, lags)
    if method != "sup_wald":
        raise InputError(f"method must be 'sup_wald' or 'cusum', got {method!r}.")
    if trim not in QUANTILES:
        raise InputError(
            f"trim must be one of {sorted(QUANTILES)} (p-values are tabulated for those), "
            f"got {trim!r}."
        )
    lo = max(math.ceil(trim * n), _MIN_SIDE)
    hi = min(math.floor((1 - trim) * n), n - _MIN_SIDE)
    if hi < lo:
        raise InsufficientDataError(f"Too few observations ({n}) to search for a break.")
    ks = np.arange(lo, hi + 1)
    csum = np.cumsum(y)
    before = csum[ks - 1] / ks
    after = (csum[-1] - csum[ks - 1]) / (n - ks)
    lrv = _long_run_variance(y - y.mean(), lags)
    walds = (after - before) ** 2 / (lrv * (1.0 / ks + 1.0 / (n - ks)))
    best = int(np.argmax(walds))
    k_best = int(ks[best])
    change, t_best = _dummy_regression(y, k_best, lags)
    stat = float(walds[best])
    return BreakResult(
        date=pd.Timestamp(y_s.index[k_best]),
        stat=stat,
        p_value=supwald_pvalue(stat, trim),
        mean_before=float(y[:k_best].mean()),
        mean_after=float(y[k_best:].mean()),
        change=change,
        change_t=t_best,
        method="sup_wald",
        n_before=k_best,
        n_after=n - k_best,
        hac_lags=lags,
        trim=float(trim),
        path=pd.Series(walds, index=y_s.index[ks], name="wald"),
    )


def supwald_pvalue(stat: float, trim: float) -> float:
    """Asymptotic p-value for a one-parameter sup-Wald statistic (simulated table)."""
    q = np.asarray(QUANTILES[trim])
    levels = np.asarray(LEVELS)
    if stat <= q[0]:
        return float(1 - levels[0])
    if stat >= q[-1]:
        return float(_P_FLOOR)
    return float(1 - np.interp(stat, q, levels))


def _prepare(returns: pd.Series[float], hac_lags: int | None) -> tuple[pd.Series[float], int]:
    series = as_series(returns, "returns")
    window = series.attrs.get(WINDOW_ATTR)
    y_s = dropna_series(series, "returns")
    n = len(y_s)
    if n < _MIN_OBS:
        raise InsufficientDataError(f"Need at least {_MIN_OBS} observations, got {n}.")
    if np.ptp(y_s.to_numpy()) == 0:
        raise InputError("returns is constant, so there's nothing to test.")
    min_lags = int(window) - 1 if isinstance(window, (int, np.integer)) and window > 1 else 0
    lags = resolve_hac_lags(hac_lags, n, min_lags=min(min_lags, n - 1))
    return y_s, lags


def _long_run_variance(e: FloatArray, lags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of a mean-zero series."""
    n = len(e)
    lrv = float(e @ e) / n
    for lag in range(1, lags + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag]) / n
    return max(lrv, 1e-300)


def _dummy_regression(y: FloatArray, k: int, lags: int) -> tuple[float, float]:
    """Regress y on [1, 1(t >= k)]: return the shift and its Newey-West t-statistic."""
    x = np.ones((len(y), 2))
    x[:k, 1] = 0.0
    res = ols_hac(y, x, lags)
    return float(res.params[1]), float(res.tvalues[1])


def _cusum(y_s: pd.Series[float], y: FloatArray, lags: int) -> BreakResult:
    n = len(y)
    e = y - y.mean()
    # Long-run (Newey-West) variance of the deviations, so autocorrelation doesn't fake drift.
    sigma = math.sqrt(_long_run_variance(e, lags))
    path = np.cumsum(e) / (sigma * math.sqrt(n))
    stat = float(np.abs(path).max())
    k = int(np.abs(path[:-1]).argmax()) + 1
    k = min(max(k, 1), n - 1)
    change, t = _dummy_regression(y, k, lags) if min(k, n - k) >= 2 else (math.nan, math.nan)
    return BreakResult(
        date=pd.Timestamp(y_s.index[k]),
        stat=stat,
        p_value=float(sps.kstwobign.sf(stat)),
        mean_before=float(y[:k].mean()),
        mean_after=float(y[k:].mean()),
        change=change,
        change_t=t,
        method="cusum",
        n_before=k,
        n_after=n - k,
        hac_lags=lags,
        path=pd.Series(path, index=y_s.index, name="cusum"),
    )
