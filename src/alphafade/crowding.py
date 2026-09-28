"""Comomentum-style crowding score (Lou & Polk)."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ._errors import AlignmentError, DataDroppedWarning, InputError
from ._stats import FloatArray
from ._validate import as_panel, check_window, same_freq

__all__ = ["crowding_score"]

CrowdingMethod = Literal["leave_one_out", "pairwise"]


def crowding_score(
    stock_returns: pd.DataFrame,
    long_members: pd.DataFrame,
    short_members: pd.DataFrame,
    *,
    factors: pd.DataFrame | None = None,
    window: int = 52,
    method: CrowdingMethod = "leave_one_out",
    min_obs: int = 26,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """Measure crowding in each leg of a trade as the co-movement of its stocks' residuals.

    For every formation date, alphafade takes the ``window`` periods of returns up to and
    including that date, removes what a factor model (default: whatever ``factors`` you
    pass, typically Fama-French 3) explains, and measures how strongly the leftover
    (residual) returns of stocks in the same leg move together. If lots of capital is
    buying the same winners and selling the same losers, their residuals become correlated
    even after controlling for the factors. Lou & Polk (2022) call this comomentum and find
    that high values predict weaker, then reversing, momentum returns.

    Only returns up to each formation date are used, so the score is known at that date
    (no look-ahead).

    Parameters
    ----------
    stock_returns : DataFrame
        Dates x stocks, simple returns (decimals). Lou & Polk use weekly returns.
    long_members, short_members : DataFrame of bool
        Formation dates x stocks; True if the stock is in that leg on that date (e.g. the
        top and bottom deciles). Formation dates can be less frequent than returns (e.g.
        monthly formation, weekly returns). Stock labels must match ``stock_returns``.
    factors : DataFrame, optional
        Factor returns at the same frequency as ``stock_returns``, covering every return
        date used, e.g. ``alphafade.datasets.load_ff3("W")``. A column named ``"RF"`` is
        treated as the risk-free rate: it's subtracted from stock returns and not used as a
        regressor. If omitted, residuals are just deviations from each stock's own mean,
        so market-wide moves will dominate the score. That's not recommended.
    window : int, default 52
        Number of return periods per estimation window (52 weeks = 1 year, as in the paper).
    method : {"leave_one_out", "pairwise"}, default "leave_one_out"
        ``"leave_one_out"`` (Lou & Polk): average, over stocks, of the correlation between
        a stock's residual and the equal-weighted residual of the rest of its leg.
        ``"pairwise"``: average correlation over all pairs of stocks in the leg. The two are
        related but not equal: leave-one-out values are larger, because a portfolio's
        residual is less noisy than a single stock's.
    min_obs : int, default 26
        A stock needs at least this many non-missing returns in the window to be used.
    min_stocks : int, default 5
        A leg needs at least this many usable stocks, or its score is NaN.

    Returns
    -------
    DataFrame
        Indexed by formation date, with columns ``long``, ``short``, ``mean`` (average of
        the two legs), ``n_long`` and ``n_short`` (stocks used). Dates without a full window
        of history are NaN.

    Warns
    -----
    DataDroppedWarning
        If member stocks are missing from ``stock_returns``, are dropped for having fewer
        than ``min_obs`` returns in a window, or a leg has fewer than ``min_stocks`` usable
        stocks.

    Notes
    -----
    Survivorship bias: if your stock universe only includes companies that exist today,
    the stocks that dropped out (often losers) are missing, which distorts the short leg
    most. Use a point-in-time universe (e.g. CRSP) when you can.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> dates = pd.date_range("2000-01-07", periods=156, freq="W-FRI")
    >>> common = rng.normal(0, 0.02, (156, 1))
    >>> rets = pd.DataFrame(rng.normal(0, 0.03, (156, 40)) + common * (np.arange(40) < 20),
    ...                     index=dates)
    >>> formation = dates[51::4]
    >>> in_long = np.tile(np.arange(40) < 20, (len(formation), 1))
    >>> long_ = pd.DataFrame(in_long, index=formation, columns=rets.columns)
    >>> crowd = crowding_score(rets, long_, ~long_, window=52)
    >>> bool((crowd["long"] > crowd["short"] + 0.1).all())
    True
    """
    if method not in ("leave_one_out", "pairwise"):
        raise InputError(f"method must be 'leave_one_out' or 'pairwise', got {method!r}.")
    for name, value, low in (("min_obs", min_obs, 3), ("min_stocks", min_stocks, 3)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise InputError(f"{name} must be an integer of at least {low}, got {value!r}.")
    rets = as_panel(stock_returns, "stock_returns")
    window = check_window(window, len(rets))
    if min_obs > window:
        raise InputError(f"min_obs ({min_obs}) can't exceed window ({window}).")
    legs = {
        "long": _membership(long_members, "long_members"),
        "short": _membership(short_members, "short_members"),
    }
    if not legs["long"].index.equals(legs["short"].index):
        raise AlignmentError(
            "long_members and short_members must have the same formation dates (index)."
        )
    formation = pd.DatetimeIndex(legs["long"].index)

    all_members = legs["long"].columns.union(legs["short"].columns)
    ever_mask = legs["long"].any().reindex(all_members, fill_value=False) | legs[
        "short"
    ].any().reindex(all_members, fill_value=False)
    ever = all_members[ever_mask.to_numpy(dtype=bool)]
    known = ever.intersection(rets.columns)
    if len(known) == 0 and len(ever) > 0:
        raise AlignmentError(
            "None of the member stocks appear as columns of stock_returns. Check that both "
            "use the same identifiers."
        )
    if len(ever) - len(known):
        warnings.warn(
            f"{len(ever) - len(known)} member stock(s) have no column in stock_returns and "
            "are ignored.",
            DataDroppedWarning,
            stacklevel=2,
        )

    x_factors: FloatArray | None = None
    if factors is not None:
        fac = as_panel(factors, "factors")
        if len(rets) >= 3 and len(fac) >= 3:
            same_freq(
                pd.DatetimeIndex(rets.index),
                pd.DatetimeIndex(fac.index),
                "stock_returns",
                "factors",
            )
        missing = rets.index.difference(fac.index)
        if len(missing):
            raise AlignmentError(
                f"factors is missing {len(missing)} of the stock_returns dates (first: "
                f"{missing[0]:%Y-%m-%d}). Factors must cover every return date, with the "
                "same date convention (e.g. both Friday-dated weekly)."
            )
        fac = fac.loc[rets.index]
        if fac.isna().to_numpy().any():
            raise InputError("factors contains missing values on stock return dates.")
        rf_col = [c for c in fac.columns if str(c).upper() == "RF"]
        if rf_col:
            rets = rets.sub(fac[rf_col[0]], axis=0)
            fac = fac.drop(columns=rf_col)
        x_factors = fac.to_numpy(dtype=np.float64)

    ret_idx = rets.index.to_numpy()
    values = rets.to_numpy(dtype=np.float64)
    col_pos = {c: i for i, c in enumerate(rets.columns)}
    out = np.full((len(formation), 4), np.nan)
    thin_stocks = 0
    thin_legs = 0
    for row, date in enumerate(formation):
        end = int(np.searchsorted(ret_idx, np.datetime64(date), side="right"))
        start = end - window
        if start < 0:
            continue  # not enough history yet (warm-up)
        x_win = None if x_factors is None else x_factors[start:end]
        for leg_i, leg in enumerate(("long", "short")):
            members = legs[leg].columns[legs[leg].loc[date].to_numpy()]
            cols = [col_pos[c] for c in members if c in col_pos]
            block = values[start:end, cols]
            enough = np.sum(~np.isnan(block), axis=0) >= min_obs
            thin_stocks += int((~enough).sum())
            block = block[:, enough]
            out[row, 2 + leg_i] = block.shape[1]
            if block.shape[1] < min_stocks:
                if len(cols) > 0:
                    thin_legs += 1
                continue
            resid = _residualize(block, x_win)
            out[row, leg_i] = (
                _leave_one_out(resid) if method == "leave_one_out" else _pairwise(resid)
            )
    if thin_stocks:
        warnings.warn(
            f"Skipped {thin_stocks} stock-window(s) with fewer than {min_obs} returns.",
            DataDroppedWarning,
            stacklevel=2,
        )
    if thin_legs:
        warnings.warn(
            f"{thin_legs} leg-date(s) had fewer than {min_stocks} usable stocks; their score "
            "is NaN.",
            DataDroppedWarning,
            stacklevel=2,
        )
    return _assemble(out, formation)


def _assemble(out: FloatArray, formation: pd.DatetimeIndex) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows in nanmean
        mean = np.nanmean(out[:, :2], axis=1) if len(out) else np.empty(0)
    counts = np.nan_to_num(out[:, 2:], nan=0.0).astype(np.int64)
    return pd.DataFrame(
        {
            "long": out[:, 0],
            "short": out[:, 1],
            "mean": mean,
            "n_long": counts[:, 0],
            "n_short": counts[:, 1],
        },
        index=pd.DatetimeIndex(formation, name="date"),
    )


def _membership(x: object, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise InputError(
            f"{name} must be a boolean DataFrame (formation dates x stocks), got "
            f"{type(x).__name__}. For a fixed list of tickers use "
            "`pd.DataFrame(True, index=dates, columns=tickers)`."
        )
    frame = x.fillna(False) if x.isna().to_numpy().any() else x
    try:
        as_bool = frame.astype(bool)
    except (TypeError, ValueError):
        raise InputError(f"{name} must contain True/False values.") from None
    if not frame.isin([True, False, 0, 1]).to_numpy().all():
        raise InputError(f"{name} must contain only True/False (or 1/0) values.")
    as_panel(as_bool.astype(float), name)  # index checks: DatetimeIndex, sorted, unique
    return as_bool


def _residualize(block: FloatArray, x: FloatArray | None) -> FloatArray:
    """Residuals of each column on [1, factors], fitted on that column's non-missing rows."""
    t = block.shape[0]
    design = np.ones((t, 1)) if x is None else np.column_stack([np.ones(t), x])
    resid = np.full_like(block, np.nan)
    complete = ~np.isnan(block).any(axis=0)
    if complete.any():
        coef, *_ = np.linalg.lstsq(design, block[:, complete], rcond=None)
        resid[:, complete] = block[:, complete] - design @ coef
    for j in np.flatnonzero(~complete):
        ok = ~np.isnan(block[:, j])
        coef_j, *_ = np.linalg.lstsq(design[ok], block[ok, j], rcond=None)
        resid[ok, j] = block[ok, j] - design[ok] @ coef_j
    return resid


def _corr_columns(a: FloatArray, b: FloatArray) -> FloatArray:
    """Column-by-column correlation of a and b using rows where both are present."""
    ok = ~(np.isnan(a) | np.isnan(b))
    n = ok.sum(axis=0)
    a0 = np.where(ok, a, 0.0)
    b0 = np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a0.sum(axis=0) / n
        mb = b0.sum(axis=0) / n
        da = np.where(ok, a0 - ma, 0.0)
        db = np.where(ok, b0 - mb, 0.0)
        r: FloatArray = (da * db).sum(axis=0) / np.sqrt(
            (da * da).sum(axis=0) * (db * db).sum(axis=0)
        )
    r[n < 3] = np.nan
    return r


def _leave_one_out(resid: FloatArray) -> float:
    """Mean correlation of each stock with the equal-weighted average of the others."""
    present = ~np.isnan(resid)
    total = np.nansum(resid, axis=1, keepdims=True)
    count = present.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        others = (total - np.where(present, resid, 0.0)) / (count - present)
    others[~np.isfinite(others)] = np.nan
    r = _corr_columns(resid, others)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def _pairwise(resid: FloatArray) -> float:
    """Mean off-diagonal pairwise correlation (pairwise-complete observations)."""
    if not np.isnan(resid).any():
        c: FloatArray = np.atleast_2d(np.corrcoef(resid, rowvar=False))
    else:
        c = pd.DataFrame(resid).corr(min_periods=3).to_numpy(dtype=np.float64)
    k = c.shape[0]
    off = c[~np.eye(k, dtype=bool)]
    return float(np.nanmean(off)) if np.isfinite(off).any() else float("nan")
