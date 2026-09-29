"""Forward-return alignment, information coefficients, and rolling Sharpe ratios."""

from __future__ import annotations

import warnings
from typing import Literal, TypeVar

import numpy as np
import pandas as pd

from ._errors import DataDroppedWarning, InputError, InsufficientDataError
from ._validate import (
    PERIODS_PER_YEAR,
    align_panels,
    as_panel,
    as_series,
    check_window,
    resolve_freq,
)

__all__ = ["forward_returns", "ic_series", "rolling_ic", "rolling_sharpe"]

ICMethod = Literal["spearman", "pearson"]
_Frame = TypeVar("_Frame", pd.Series, pd.DataFrame)

# Rolling outputs record their window here so fit_decay can adjust for the overlap.
WINDOW_ATTR = "alphafade_window"


def forward_returns(returns: _Frame, periods: int = 1) -> _Frame:
    """Shift realized returns so each row holds the return earned *after* that date.

    This is the one place alphafade moves returns in time. If ``returns`` at date t is the
    return over the period ending at t, the output at date t is the compounded return over
    the next ``periods`` periods: (t, t + periods]. A signal known at t can then be compared
    with the output at t without look-ahead bias (accidentally using information from the
    future). The last ``periods`` rows are NaN because their future isn't observed yet.

    Parameters
    ----------
    returns : Series or DataFrame
        Simple (not log) returns as decimals, indexed by date. DataFrames are dates x assets.
    periods : int, default 1
        Horizon in rows. Multi-period returns are compounded.

    Returns
    -------
    Series or DataFrame
        Same shape and labels as ``returns``.

    Examples
    --------
    >>> import pandas as pd
    >>> r = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    >>> af_fwd = forward_returns(r)
    >>> af_fwd.round(4).tolist()
    [0.02, -0.01, nan]
    """
    if isinstance(returns, pd.DataFrame):
        data: pd.Series | pd.DataFrame = as_panel(returns, "returns")
    else:
        data = as_series(returns, "returns")
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 1:
        raise InputError(f"periods must be a positive integer, got {periods!r}.")
    if (data < -1).to_numpy().any():
        raise InputError(
            "returns contains values below -1 (a loss of more than 100%). alphafade expects "
            "simple returns as decimals, e.g. 0.05 for +5%. If you passed log returns or "
            "percents, convert them first."
        )
    growth = 1.0 + data
    out = growth.shift(-1)
    for step in range(2, int(periods) + 1):
        out = out * growth.shift(-step)
    return out - 1.0  # type: ignore[return-value]


def ic_series(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: ICMethod = "spearman",
    min_assets: int = 5,
) -> pd.Series[float]:
    """Per-date information coefficient (IC).

    The IC on date t is the cross-sectional correlation between the signal known at t and
    the forward returns at t (the returns realized *after* t; build them with
    :func:`forward_returns`). An IC of 0.05 means the signal ranks next period's winners and
    losers slightly better than chance.

    Feed this raw per-date series to :func:`fit_decay`. Unlike a rolling average, its
    errors aren't artificially correlated.

    Parameters
    ----------
    signal : DataFrame
        Dates x assets. Values known at each date.
    fwd_returns : DataFrame
        Dates x assets. Row t must hold returns realized after t.
    method : {"spearman", "pearson"}, default "spearman"
        Spearman correlates ranks (robust to outliers and the usual choice for IC).
        Pearson correlates raw values.
    min_assets : int, default 5
        Dates with fewer assets that have both a signal and a return get NaN.

    Returns
    -------
    Series
        IC per date, named ``"ic"``. Dates with no overlapping data are NaN.

    Warns
    -----
    DataDroppedWarning
        If the inputs only partly overlap, or some dates had data but too few assets (or a
        constant signal) to compute an IC. Dates where no asset has both a signal and a
        return (such as the last date after :func:`forward_returns`) are NaN without a
        warning here; :func:`fit_decay` warns if such gaps fall in the middle of the sample.

    Notes
    -----
    If your universe contains only stocks that exist today, the IC is subject to
    survivorship bias: stocks that were delisted (often after crashing) are missing, which
    usually flatters the signal in early years and can fake a decay.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    >>> sig = pd.DataFrame([[1, 2, 3], [3, 2, 1]], index=dates, columns=list("abc"))
    >>> fwd = pd.DataFrame([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]], index=dates,
    ...                    columns=list("abc"))
    >>> ic_series(sig, fwd, min_assets=3).tolist()
    [1.0, -1.0]
    """
    if method not in ("spearman", "pearson"):
        raise InputError(f"method must be 'spearman' or 'pearson', got {method!r}.")
    if isinstance(min_assets, bool) or not isinstance(min_assets, (int, np.integer)):
        raise InputError(f"min_assets must be an integer, got {min_assets!r}.")
    if min_assets < 2:
        raise InputError("min_assets must be at least 2 (a correlation needs 2 points).")
    sig = as_panel(signal, "signal")
    fwd = as_panel(fwd_returns, "fwd_returns")
    sig, fwd = align_panels(sig, fwd, "signal", "fwd_returns")

    both = sig.notna() & fwd.notna()
    s = sig.where(both)
    f = fwd.where(both)
    if method == "spearman":
        s = s.rank(axis=1)
        f = f.rank(axis=1)
    count = both.sum(axis=1).to_numpy()
    sv = s.to_numpy()
    fv = f.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        sd = sv - np.nanmean(sv, axis=1, keepdims=True)
        fd = fv - np.nanmean(fv, axis=1, keepdims=True)
        num = np.nansum(sd * fd, axis=1)
        den = np.sqrt(np.nansum(sd * sd, axis=1) * np.nansum(fd * fd, axis=1))
        ic = num / den
    ic = np.clip(ic, -1.0, 1.0)
    too_few = (count > 0) & (count < min_assets)
    degenerate = (count >= min_assets) & ~(den > 0)
    ic[(count < min_assets) | ~(den > 0)] = np.nan
    if too_few.any() or degenerate.any():
        parts = []
        if too_few.any():
            parts.append(f"{int(too_few.sum())} had fewer than {min_assets} usable assets")
        if degenerate.any():
            parts.append(f"{int(degenerate.sum())} had a constant signal or constant returns")
        warnings.warn(
            f"IC set to NaN on {int(too_few.sum() + degenerate.sum())} of {len(ic)} dates: "
            + "; ".join(parts)
            + ".",
            DataDroppedWarning,
            stacklevel=2,
        )
    return pd.Series(ic, index=sig.index, name="ic", dtype="float64")


def rolling_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    window: int = 36,
    method: ICMethod = "spearman",
    min_assets: int = 5,
    min_periods: int | None = None,
    freq: str | None = None,
) -> pd.Series[float]:
    """Compute the rolling mean of the per-date information coefficient.

    Parameters
    ----------
    signal, fwd_returns, method, min_assets
        As in :func:`ic_series`.
    window : int, default 36
        Number of periods (rows) in each window, e.g. 36 months.
    min_periods : int, optional
        Minimum non-NaN ICs needed in a window. Defaults to ``window``.
    freq : str, optional
        'D', 'W', 'M', 'Q' or 'A'. Inferred from the dates if omitted; mixed or irregular
        frequencies raise :class:`FrequencyError`.

    Returns
    -------
    Series
        Rolling mean IC named ``"rolling_ic"``. NaN until the window fills.

    Notes
    -----
    Neighbouring values share ``window - 1`` observations, so the series is strongly
    autocorrelated. Don't run ordinary t-tests or regressions on it; use
    :func:`ic_series` with :func:`fit_decay` instead.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> dates = pd.date_range("2000-01-31", periods=48, freq="ME")
    >>> sig = pd.DataFrame(rng.normal(size=(48, 20)), index=dates)
    >>> fwd = 0.1 * sig + pd.DataFrame(rng.normal(size=(48, 20)), index=dates)
    >>> ric = rolling_ic(sig, fwd, window=12)
    >>> int(ric.notna().sum())
    37
    """
    ic = ic_series(signal, fwd_returns, method=method, min_assets=min_assets)
    resolve_freq(pd.DatetimeIndex(ic.index), freq, "signal")
    window = check_window(window, len(ic))
    mp = window if min_periods is None else check_window(min_periods, window, "min_periods")
    out = ic.rolling(window, min_periods=mp).mean().rename("rolling_ic")
    out.attrs[WINDOW_ATTR] = window
    return out


def rolling_sharpe(
    returns: pd.Series[float],
    window: int = 252,
    freq: str | None = None,
    rf: float | pd.Series[float] = 0.0,
    min_periods: int | None = None,
) -> pd.Series[float]:
    """Annualized rolling Sharpe ratio of a return series.

    The Sharpe ratio is average excess return divided by its volatility (standard
    deviation), scaled to a year: mean / std * sqrt(periods per year).

    Parameters
    ----------
    returns : Series
        Periodic strategy returns as decimals. Long-short (zero-cost) returns are already
        excess returns, so leave ``rf`` at 0 for them.
    window : int, default 252
        Periods per window (252 is about one year of trading days).
    freq : str, optional
        'D', 'W', 'M', 'Q' or 'A', used for annualizing (252/52/12/4/1). Inferred if omitted.
    rf : float or Series, default 0.0
        Per-period risk-free rate subtracted before computing the ratio.
    min_periods : int, optional
        Minimum observations per window. Defaults to ``window``.

    Returns
    -------
    Series
        Named ``"rolling_sharpe"``. NaN until the window fills, and where volatility is 0.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> r = pd.Series(0.01 + 0.04 * rng.standard_normal(120),
    ...               index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    >>> sr = rolling_sharpe(r, window=36)
    >>> bool(sr.dropna().between(-3, 4).all())
    True
    """
    r = as_series(returns, "returns")
    f = resolve_freq(pd.DatetimeIndex(r.index), freq, "returns")
    window = check_window(window, len(r))
    if r.notna().sum() < window:
        raise InsufficientDataError(
            f"returns has only {int(r.notna().sum())} non-missing values, fewer than the "
            f"rolling window ({window}), so no Sharpe ratio can be computed."
        )
    mp = window if min_periods is None else check_window(min_periods, window, "min_periods")
    if isinstance(rf, pd.Series):
        rf_s = as_series(rf, "rf")
        missing = r.index.difference(rf_s.index)
        if len(missing):
            raise InputError(
                f"rf is missing {len(missing)} of the return dates (first: "
                f"{missing[0]:%Y-%m-%d}). Pass an rf Series covering every return date."
            )
        excess = r - rf_s.reindex(r.index)
    else:
        excess = r - float(rf)
    roll = excess.rolling(window, min_periods=mp)
    mean = roll.mean()
    std = roll.std(ddof=1)
    # A window of identical returns has zero volatility, but pandas' streaming variance can
    # leave floating-point residue there, so detect constant windows exactly (max == min)
    # and return NaN instead of a huge meaningless ratio.
    spread = roll.max() - roll.min()
    sharpe = (mean / std.where(spread > 0)) * np.sqrt(PERIODS_PER_YEAR[f])
    out: pd.Series[float] = sharpe.rename("rolling_sharpe")
    out.attrs[WINDOW_ATTR] = window
    return out
